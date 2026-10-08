use std::borrow::Cow;
use gpu_api_dto::TextureType;
use gpu_api_relay::model_bindless_data::{CameraUniform, CullingTask, DrawIndexedIndirectCommand, MaterialFactors, NodeData, StaticVertex, VisibleInstanceData};
use log::info;
use wgpu::{Queue, TextureFormat, util::{DeviceExt, StagingBelt}};
use crate::{camera::CAMERA_UNIFORM_SIZE, pipeline::{model_pipeline::model::InitData, static_bindless_pipeline::{ALIGN, InstanceData, MAX_INDICES, MAX_INSTANCES, MAX_MATERIALS, MAX_MESH_INFOS, MAX_MESHLETS, MAX_NODES, MAX_TEXTURES, MAX_VERTICES, MeshInfo, NUM_FRAMES_IN_FLIGHT, StaticBindlessResources, StaticMeshletDescription, static_bindless_frame_res::{BufferRange, FrameResourceRanges, FrameResources, align_to}, static_bindless_layout::PipelineLayouts, static_bindless_pipelines::create_pipelines}}};

impl StaticBindlessResources {
    pub fn new(device: &wgpu::Device, queue: &Queue, init_data: &InitData, depth_stencil: Option<wgpu::DepthStencilState>) -> Self {
        let max_meshlets_count = 200;
        
        let layouts = PipelineLayouts::new(device);
        let (culling_compute_pipeline, render_pipeline) = create_pipelines(device, &layouts,
            "../shaders/static_bindless_culling.wgsl",
            "../shaders/static_bindless.wgsl",
        depth_stencil);

        // =====================================================================
        // БУФЕР 1: StaticGeometryBuffer (Вершины + Индексы)
        // =====================================================================
        let mut geom_offset = 0;

        let vertex_size = (MAX_VERTICES as u64) * std::mem::size_of::<StaticVertex>() as u64;
        let vertex_range = BufferRange { offset: geom_offset, size: vertex_size };
        geom_offset = align_to(geom_offset + vertex_size, ALIGN);

        let index_size = (MAX_INDICES as u64) * 4; // u32 сквозной шаблон
        let index_range = BufferRange { offset: geom_offset, size: index_size };
        geom_offset = align_to(geom_offset + index_size, ALIGN);

        let geometry_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Static Geometry Buffer (Vertices & Indices)"),
            size: geom_offset,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // =====================================================================
        // БУФЕР 2: SceneDataBuffer (Массивы структур Storage, только чтение)
        // =====================================================================
        let mut scene_offset = 0;

        let mesh_infos_size = (MAX_INSTANCES as u64) * std::mem::size_of::<MeshInfo>() as u64;
        let mesh_infos_range = BufferRange { offset: scene_offset, size: mesh_infos_size };
        scene_offset = align_to(scene_offset + mesh_infos_size, ALIGN);

        let meshlets_size = (MAX_MESHLETS as u64) * std::mem::size_of::<StaticMeshletDescription>() as u64;
        let meshlets_range = BufferRange { offset: scene_offset, size: meshlets_size };
        scene_offset = align_to(scene_offset + meshlets_size, ALIGN);

        let local_indices_size = (MAX_MESHLETS as u64) * 18 * 4; // По 18 индексов (6 трианглов) на мешлет
        let meshlet_local_indices_range = BufferRange { offset: scene_offset, size: local_indices_size };
        scene_offset = align_to(scene_offset + local_indices_size, ALIGN);

        let redirect_size = (MAX_MESHLETS as u64) * 8 * 4; // Максимум по 8 вершин перенаправления на мешлет куба
        let meshlet_vertex_redirect_range = BufferRange { offset: scene_offset, size: redirect_size };
        scene_offset = align_to(scene_offset + redirect_size, ALIGN);

        let materials_size = (MAX_MATERIALS as u64) * std::mem::size_of::<MaterialFactors>() as u64;
        let materials_range = BufferRange { offset: scene_offset, size: materials_size };
        scene_offset = align_to(scene_offset + materials_size, ALIGN);

        let instances_size = (MAX_INSTANCES as u64) * std::mem::size_of::<InstanceData>() as u64;
        let instances_range = BufferRange { offset: scene_offset, size: instances_size };
        scene_offset = align_to(scene_offset + instances_size, ALIGN);

        let nodes_size = (MAX_NODES as u64) * std::mem::size_of::<NodeData>() as u64;
        let nodes_range = BufferRange { offset: scene_offset, size: nodes_size };
        scene_offset = align_to(scene_offset + nodes_size, ALIGN);

        // Буфер задач куллинга (обычно равен числу инстансов или проходов)
        let culling_tasks_size = (MAX_INSTANCES as u64) * std::mem::size_of::<CullingTask>() as u64;
        let culling_tasks_range = BufferRange { offset: scene_offset, size: culling_tasks_size };
        scene_offset = align_to(scene_offset + culling_tasks_size, ALIGN);

        let scene_data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scene Data Buffer (Storage Bindless Arrays)"),
            size: scene_offset,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let materials_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Materials Buffer"),
            size: MAX_MATERIALS * size_of::<MaterialFactors>() as u64, 
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // =====================================================================
        // БУФЕР 3: FrameRingBuffer (Динамическое кадровое кольцо)
        // =====================================================================
        let mut ring_offset = 0;
        let mut frame_ranges = Vec::new();

        let counter_size = 4; // u32 атомарный счетчик
        let indirect_size = (MAX_MESHLETS as u64) * 20; // DrawIndexedIndirectArgs (20 байт)
        let visible_instances_size = (MAX_MESHLETS as u64) * 4; // Массив u32 индексов видимых мешлетов
        let camera_size = align_to(std::mem::size_of::<CameraUniform>() as u64, ALIGN);

        for _ in 0..NUM_FRAMES_IN_FLIGHT {
            let counter_offset = ring_offset;
            ring_offset = align_to(ring_offset + counter_size, ALIGN);

            let indirect_offset = ring_offset;
            ring_offset = align_to(ring_offset + indirect_size, ALIGN);

            let visible_instances_offset = ring_offset;
            ring_offset = align_to(ring_offset + visible_instances_size, ALIGN);

            let camera_offset = ring_offset;
            ring_offset = align_to(ring_offset + camera_size, ALIGN);

            frame_ranges.push(FrameResourceRanges {
                counter: BufferRange { offset: counter_offset, size: counter_size },
                indirect: BufferRange { offset: indirect_offset, size: indirect_size },
                visible_instances: BufferRange { offset: visible_instances_offset, size: visible_instances_size },
                camera: BufferRange { offset: camera_offset, size: camera_size },
            });
        }

        let frame_ring_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Frame Ring Buffer (Dynamic Rings)"),
            size: ring_offset,
            usage: wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::UNIFORM 
                | wgpu::BufferUsages::INDIRECT 
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // =====================================================================
        // НАРЕЗКА СРЕЗОВ (BufferBinding) И ИНИЦИАЛИЗАЦИЯ BIND GROUPS КАДРОВ
        // =====================================================================
        let mut frame_resources = Vec::new();

        for i in 0..NUM_FRAMES_IN_FLIGHT {
            let ranges = &frame_ranges[i];

            // Нарезка кадровой группы для Compute-шейдера куллинга
            let culling_compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Culling Compute Bind Group Frame {}", i)),
                layout: &layouts.culling_compute_layout,
                entries: &[                
                    wgpu::BindGroupEntry { 
                        binding: 0, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &frame_ring_buffer, offset: ranges.camera.offset, size: wgpu::BufferSize::new(ranges.camera.size)
                        })
                    },
                    wgpu::BindGroupEntry { 
                        binding: 1, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &scene_data_buffer, offset: culling_tasks_range.offset, size: wgpu::BufferSize::new(culling_tasks_range.size)
                        })
                    },                
                    wgpu::BindGroupEntry { 
                        binding: 2, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &scene_data_buffer, offset: instances_range.offset, size: wgpu::BufferSize::new(instances_range.size)
                        })
                    },
                    wgpu::BindGroupEntry { 
                        binding: 3, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &scene_data_buffer, offset: mesh_infos_range.offset, size: wgpu::BufferSize::new(mesh_infos_range.size)
                        })
                    },                
                    wgpu::BindGroupEntry { 
                        binding: 4, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &scene_data_buffer, offset: meshlets_range.offset, size: wgpu::BufferSize::new(meshlets_range.size)
                        })
                    },
                    wgpu::BindGroupEntry { 
                        binding: 5, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &frame_ring_buffer, offset: ranges.visible_instances.offset, size: wgpu::BufferSize::new(ranges.visible_instances.size)
                        })
                    },                
                    wgpu::BindGroupEntry { 
                        binding: 6, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &frame_ring_buffer, offset: ranges.indirect.offset, size: wgpu::BufferSize::new(ranges.indirect.size)
                        })
                    },
                    wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &frame_ring_buffer, offset: ranges.counter.offset, size: wgpu::BufferSize::new(ranges.counter.size)
                    })
                    },
                    ],
                    });
                    // Нарезка кадровой группы для графического Render-пайплайна
                    let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Render Bind Group Frame {}", i)),
                    layout: &layouts.render_bind_group_layout,
                    entries: &[
                    wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &frame_ring_buffer, offset: ranges.camera.offset, size: wgpu::BufferSize::new(ranges.camera.size)
                    })
                    },
                    wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &scene_data_buffer, offset: nodes_range.offset, size: wgpu::BufferSize::new(nodes_range.size)
                    })
                    },
                    wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &scene_data_buffer, offset: instances_range.offset, size: wgpu::BufferSize::new(instances_range.size)
                    })
                    },
                    wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &scene_data_buffer, offset: mesh_infos_range.offset, size: wgpu::BufferSize::new(mesh_infos_range.size)
                    })
                    },
                    wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &frame_ring_buffer, offset: ranges.visible_instances.offset, size: wgpu::BufferSize::new(ranges.visible_instances.size)
                    })
                    },
                    wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &scene_data_buffer, offset: meshlets_range.offset, size: wgpu::BufferSize::new(meshlets_range.size)
                    })
                    },
                    wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &scene_data_buffer, offset: meshlet_local_indices_range.offset, size: wgpu::BufferSize::new(meshlet_local_indices_range.size)
                    })
                    },
                    wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &scene_data_buffer, offset: meshlet_vertex_redirect_range.offset, size: wgpu::BufferSize::new(meshlet_vertex_redirect_range.size)
                    })
                    },
                    ],
            });

            frame_resources.push(FrameResources {
                culling_compute_bind_group,
                render_bind_group,
            });
        }

        let dummy_size = wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };

        let dummy_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy Texture Fallback"),
            size: dummy_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        let dummy_pixel = [255, 255, 255, 255];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                texture: &dummy_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &dummy_pixel,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            dummy_size,
        );

        let dummy_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let default_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Universal Material Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let max_textures = MAX_TEXTURES as usize;
        
        let mut base_color_views = vec![&dummy_view; max_textures];
        let mut metallic_roughness_views = vec![&dummy_view; max_textures];
        let mut normal_views = vec![&dummy_view; max_textures];
        let mut emissive_views = vec![&dummy_view; max_textures];
        
        info!("Materials total: {}", init_data.materials.len());
        
        for (material_idx, md) in init_data.materials.iter().enumerate() {
            info!("Got material {}, textures: {}", material_idx, md.textures.len());
            if material_idx >= max_textures {
                panic!("Max textures limit reached!");
            }
            for (t_type, texture_item) in &md.textures {
                let view_ref = &texture_item.view;
                match t_type {
                    TextureType::BaseColor => {
                        base_color_views[material_idx] = view_ref;
                    }
                    TextureType::Normal => {
                        normal_views[material_idx] = view_ref;
                    }
                    TextureType::MetallicRoughness => {
                        metallic_roughness_views[material_idx] = view_ref;
                    }
                    TextureType::Emissive => {
                        emissive_views[material_idx] = view_ref;
                    }
                    _ => {
                        info!("Unknown texture type found for bindless");
                    }
                }
            }
        }
        
        let samplers = vec![&default_sampler; max_textures];
        
        let materials_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Materials Bind Group"),
            layout: &layouts.materials_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(&base_color_views),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::SamplerArray(&samplers),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureViewArray(&metallic_roughness_views),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::SamplerArray(&samplers),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureViewArray(&normal_views),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::SamplerArray(&samplers),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureViewArray(&emissive_views),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::SamplerArray(&samplers),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: materials_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            geometry_buffer,
            scene_data_buffer,
            frame_ring_buffer,
            materials_buffer,
            vertex_range,
            index_range,
            mesh_infos_range,
            meshlets_range,
            meshlet_local_indices_range,
            meshlet_vertex_redirect_range,
            materials_range,
            instances_range,
            nodes_range,
            culling_tasks_range,
            frame_ranges,
            frame_resources,
            frame_index: 0,
            max_meshlets_count,
            culling_compute_pipeline,
            render_pipeline,
            materials_bind_group,
        }
    }
}