use std::borrow::Cow;
use gpu_api_dto::TextureType;
use gpu_api_relay::model_bindless_data::{CameraUniform, CullingTask, DrawIndexedIndirectCommand, MaterialFactors, NodeData, StaticVertex, VisibleInstanceData};
use log::info;
use wgpu::{TextureFormat, util::{DeviceExt, StagingBelt}};
use crate::{camera::CAMERA_UNIFORM_SIZE, pipeline::model_pipeline::model::InitData};

pub const MAX_VERTICES: u64 = 1_000_000;
pub const MAX_INDICES: u64 = 3_000_000;
pub const MAX_INSTANCES: u64 = 100_000;
pub const MAX_MESH_INFOS: u64 = 10000;
pub const MAX_MESHLETS: u64 = 100_000;
pub const MAX_MATERIALS: u64 = 1_000;
pub const MAX_TEXTURES: u32 = 256;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshInfo {
    pub start_meshlet_index: u32,
    pub meshlet_count: u32,
    pub vertex_buffer_offset: u32,
    pub base_vertex: i32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StaticMeshletDescription {
    pub aabb_min: [f32; 3],
    pub vertex_offset: u32,
    pub aabb_max: [f32; 3],
    pub index_offset: u32,
    pub index_count: u32,
    pub material_index: u32,
    pub pad0: u32,
    pub pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    pub model_matrix: [[f32; 4]; 4],
    pub is_animated: u32,       // Всегда 0u для статики, но сохраняем структуру
    pub node_index: u32,
    pub joints_offset: u32,     // Не используется
    pub material_index: u32,
    pub primitive_index: u32,   // Базовый ID indirect-команды первого мешлета этого инстанса
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
    pub aabb_min: [f32; 3],
    pub pad_aabb1: u32,
    pub aabb_max: [f32; 3],
    pub pad_aabb2: u32,
}

pub struct StaticBindlessResources {
    // Буферы геометрии и материалов
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub materials_buffer: wgpu::Buffer,
    
    // Новые буферы мешлет-архитектуры
    pub mesh_infos_buffer: wgpu::Buffer,
    pub meshlets_buffer: wgpu::Buffer,
    
    // Буферы сцены и куллинга
    pub instances_buffer: wgpu::Buffer,
    pub visible_instances_buffer: wgpu::Buffer,
    pub nodes_buffer: wgpu::Buffer,
    pub culling_tasks_buffer: wgpu::Buffer,
    pub camera_buffer: wgpu::Buffer,
    
    // Indirect Draw Буферы
    pub indirect_commands_template_buffer: wgpu::Buffer,
    pub indirect_commands_buffer: wgpu::Buffer,
    
    // Пайплайны и бинд-группы
    pub culling_compute_pipeline: wgpu::ComputePipeline,
    pub render_pipeline: wgpu::RenderPipeline,
    pub camera_bind_group: wgpu::BindGroup,
    pub culling_compute_bind_group: wgpu::BindGroup,
    pub materials_bind_group: wgpu::BindGroup,
    pub render_bind_group: wgpu::BindGroup,
}

impl StaticBindlessResources {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,        
        camera_uniform: &CameraUniform,
        depth_stencil: Option<wgpu::DepthStencilState>,
        total_meshlets_commands_count: usize,        
        init_data: &InitData,
    ) -> Self {                        
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mega Vertex Buffer"),
            size: MAX_VERTICES * size_of::<StaticVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mega Index Buffer"),
            size: MAX_INDICES * 4,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let instances_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instances Buffer"),
            size: MAX_INSTANCES * size_of::<InstanceData>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let nodes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Nodes Buffer"),
            size: MAX_INSTANCES * size_of::<NodeData>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        /*
        let joints_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Joints Buffer"),
            size: MAX_INSTANCES * 64 * 4, 
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        */

        let materials_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Materials Buffer"),
            size: MAX_MATERIALS * size_of::<MaterialFactors>() as u64, 
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let culling_tasks_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Culling Tasks Buffer"),
            size: MAX_INSTANCES * size_of::<CullingTask>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let total_visible_slots = 100 * 2; // 200 слотов под VisibleInstanceData
        let visible_buffer_size = (total_visible_slots * std::mem::size_of::<VisibleInstanceData>()) as wgpu::BufferAddress;

        let visible_instances_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Static Visible Instances Buffer"),
            size: visible_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });


        // 2. Создаем буфер метаданных мешей (MeshInfo)
        let mesh_infos_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Static Objects Mesh Infos Buffer"),
            size: MAX_MESH_INFOS * (size_of::<MeshInfo>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 3. Создаем глобальный буфер геометрии мешлетов (StaticMeshletDescription)
        let meshlets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Static Objects Global Meshlets Buffer"),
            size: MAX_MESHLETS * (size_of::<StaticMeshletDescription>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indirect_buffer_size = (total_meshlets_commands_count * std::mem::size_of::<DrawIndexedIndirectCommand>()) as u64;

        // 1. Создаем буфер-шаблон строго нужного размера (40 байт)
        let indirect_commands_template_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Static Indirect Commands Template Buffer"),
            size: indirect_buffer_size, 
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 2. Создаем рабочий буфер строго нужного размера (40 байт)
        let indirect_commands_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Static Indirect Commands Buffer"),
            size: indirect_buffer_size, 
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let culling_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Static Bindless Culling Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/static_bindless_culling.wgsl").into()),
        });
        
    let culling_compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Static Bindless Culling Compute Bind Group Layout"),
        entries: &[
            // binding(0): culling_tasks (Storage, read)
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true }, // Должно быть true!
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding(1): global_instances (Storage, read)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true }, // Должно быть true!
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding(2): global_mesh_infos (Storage, read) -> НАША ОШИБКА ЗДЕСЬ
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true }, // ОБЯЗАТЕЛЬНО TRUE!
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding(3): global_meshlets (Storage, read)
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true }, // Должно быть true!
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding(4): visible_instances (Storage, read_write)
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false }, // Здесь false, шейдер пишет сюда
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding(5): indirect_commands (Storage, read_write)
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false }, // Здесь false, шейдер пишет сюда
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });


    let camera_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(bytemuck::bytes_of(camera_uniform)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }
    );

    let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        ],
        label: Some("camera_bind_group_layout"),
    });

    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &camera_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }
        ],
        label: Some("camera_bind_group")
    });
        
        // Camera (Group 1), Instances/Nodes/Task data/Visible Indices/Indirect Commands (Group 2)
        let culling_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Culling Pipeline Layout"),
            bind_group_layouts: &[
                Some(&camera_bind_group_layout),
                Some(&culling_compute_bind_group_layout),
            ],            
            immediate_size: 0,
        });

        // 3. Создаем сам Compute Pipeline
        let culling_compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Static Bindless Culling Compute Pipeline"),
            layout: Some(&culling_pipeline_layout),
            module: &culling_shader,
            entry_point: Some("culling_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("static_bindless.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/static_bindless.wgsl")))
        });

        let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Static Bindless Render Bind Group Layout (Group 2)"),
            entries: &[
                // @binding(0): static_vertices
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(1): global_nodes
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },                                
                // @binding(2): global_instances
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(3): global_mesh_infos (НОВЫЙ БИНДИНГ: добавлен для ручного Vertex Pulling смещения)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(4): visible_instances (Сдвинут на индекс 4)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
                
        let texture_count = std::num::NonZeroU32::new(MAX_TEXTURES);

        let materials_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Materials Bind Group Layout"),
            entries: &[                
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT, // Нужны только во фрагментном шейдере
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: texture_count,
                },                
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: texture_count,
                },                
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: texture_count,
                },                
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: texture_count,
                },                
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: texture_count,
                },                
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: texture_count,
                },                
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: texture_count,
                },                
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: texture_count,
                },                
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Static Bindless Render Pipeline Layout"),
            bind_group_layouts: &[
                Some(&materials_bind_group_layout), // @group(0)
                Some(&camera_bind_group_layout),    // @group(1)
                Some(&render_bind_group_layout), // @group(2)
            ],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Static Bindless Pipeline"),
            layout: Some(&render_pipeline_layout),        
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: TextureFormat::Rgba8UnormSrgb,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })
                ],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None
        });
        
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
            layout: &materials_bind_group_layout,
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

        let culling_compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Static bindless Culling Compute Bind Group"),
            layout: &culling_compute_bind_group_layout,
            entries: &[                
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: culling_tasks_buffer.as_entire_binding(),
                },                
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instances_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mesh_infos_buffer.as_entire_binding(),
                },                
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: meshlets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: visible_instances_buffer.as_entire_binding(),
                },                
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: indirect_commands_buffer.as_entire_binding(),
                },
            ],
        });        

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Static Bindless Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[                
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: nodes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: instances_buffer.as_entire_binding(),
                },                
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mesh_infos_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: visible_instances_buffer.as_entire_binding(),
                },
            ],
        });

        Self {    
            vertex_buffer,
            index_buffer,
            camera_buffer,
            instances_buffer,
            nodes_buffer,            
            materials_buffer,
            culling_tasks_buffer,
            visible_instances_buffer,
            mesh_infos_buffer,
            meshlets_buffer,
            indirect_commands_template_buffer,
            indirect_commands_buffer,
            culling_compute_pipeline,
            render_pipeline,
            materials_bind_group,
            camera_bind_group,
            culling_compute_bind_group,
            render_bind_group,
        }        
    }

    pub fn init(
        &self,
        queue: &wgpu::Queue,
        vertices: &[StaticVertex],               // Новая структура вершин без костей
        indices: &[u32],
        meshlets: &[StaticMeshletDescription],  // Данные всех мешлетов
        mesh_infos: &[MeshInfo],                // Связующие данные мешей
        material_factors: &[MaterialFactors],
        indirect_commands: &[DrawIndexedIndirectCommand] // Массив команд (по одной на КАЖДЫЙ мешлет)
    ) {
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices));
        queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(indices));
        queue.write_buffer(&self.meshlets_buffer, 0, bytemuck::cast_slice(meshlets));
        queue.write_buffer(&self.mesh_infos_buffer, 0, bytemuck::cast_slice(mesh_infos));
        queue.write_buffer(&self.materials_buffer, 0, bytemuck::cast_slice(material_factors));
        
        // Шаблон команд содержит правильные index_count/first_index для каждого мешлета,
        // но instance_count в нем равен 0.
        queue.write_buffer(&self.indirect_commands_template_buffer, 0, bytemuck::cast_slice(indirect_commands));
        queue.write_buffer(&self.indirect_commands_buffer, 0, bytemuck::cast_slice(indirect_commands));
    }

    pub fn load_frame(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        camera_uniform: &CameraUniform,
        staging_belt: &mut StagingBelt,
        instances: &[InstanceData],
        nodes: &[NodeData],
        culling_tasks: &[CullingTask], // Задачи куллинга для объектов (чанков/LOD-групп)
    ) {
        {                                                                                                        
            let mut camera_slice = staging_belt.write_buffer(
                encoder,
                &self.camera_buffer,
                0,
                wgpu::BufferSize::new(CAMERA_UNIFORM_SIZE).expect("Failed to allocate bindless camera slice")
            );            
            camera_slice.copy_from_slice(bytemuck::bytes_of(camera_uniform));
        }

        queue.write_buffer(&self.instances_buffer, 0, bytemuck::cast_slice(instances));
        queue.write_buffer(&self.nodes_buffer, 0, bytemuck::cast_slice(nodes));

        if !culling_tasks.is_empty() {
            queue.write_buffer(&self.culling_tasks_buffer, 0, bytemuck::cast_slice(culling_tasks));
        }
    }

    pub fn clear_gpu_driven_frame(&self, encoder: &mut wgpu::CommandEncoder) {        
        encoder.copy_buffer_to_buffer(
            &self.indirect_commands_template_buffer,
            0,
            &self.indirect_commands_buffer,
            0,
            self.indirect_commands_buffer.size(), 
        );
    }

    pub fn compute_gpu_driven_frame(
        &self,
        compute_pass: &mut wgpu::ComputePass,
        total_instances_count: u32, // Передаем общее число кубов (например, 100)
    ) {        
        compute_pass.set_pipeline(&self.culling_compute_pipeline);
        compute_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        compute_pass.set_bind_group(1, &self.culling_compute_bind_group, &[]);
        
        // Распределяем группы по количеству объектов сцены
        let workgroup_count = (total_instances_count + 63) / 64;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    pub fn draw_gpu_driven_frame(
        &self,
        render_pass: &mut wgpu::RenderPass,
        commands: &[DrawIndexedIndirectCommand] // Этот массив на CPU должен содержать ровно 2 элемента
    ) {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.materials_bind_group, &[]);
        render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(2, &self.render_bind_group, &[]); 
        
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);        
        
        // ИСПРАВЛЕНИЕ: Передаем строго 2 (или commands.len() as u32), так как у нас всего 2 мешлет-команды!
        render_pass.multi_draw_indexed_indirect(
            &self.indirect_commands_buffer, 
            0, 
            commands.len() as u32
        );
    }
}
