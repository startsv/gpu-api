use std::borrow::Cow;
use gpu_api_dto::TextureType;
use gpu_api_relay::model_bindless_data::{CullingTask, DrawIndexedIndirectCommand, InstanceData, MaterialFactors, NodeData, PrimitiveMeta, SurfaceVertex, SurfaceCullingTask, SurfaceMeshletDescription, VisibleInstanceData};
use log::info;
use wgpu::{ComputePass, RenderPass, TextureFormat, util::{DeviceExt, StagingBelt}, wgt::DrawIndirectArgs};
use crate::{camera::Camera, pipeline::{clear_commands_pipeline::{self, ClearCommandsPipeline}, model_pipeline::{CAMERA_UNIFORM_SIZE, model::{InitData, MaterialData}}}};
use gpu_api_relay::model_bindless_data::CameraUniform;

pub const MAX_VERTICES: u64 = 1_000_000;
pub const MAX_INDICES: u64 = 3_000_000;
pub const MAX_INSTANCES: u64 = 100_000;
pub const MAX_MATERIALS: u64 = 1_000;
pub const MAX_TEXTURES: u32 = 256;

const MAX_MESHLETS_IN_SCENE: usize = 16384;
const MAX_TERRAIN_CULLING_TASKS: usize = 256;

pub struct SurfaceBindlessResources {    
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
        
    pub meshlets_buffer: wgpu::Buffer,        
        
    pub camera_buffer: wgpu::Buffer,
    pub materials_buffer: wgpu::Buffer,
        
    pub culling_tasks_buffer: wgpu::Buffer,
    pub active_meshlets_buffer: wgpu::Buffer,
    
    pub indirect_commands_template_buffer: wgpu::Buffer,
    pub indirect_commands_buffer: wgpu::Buffer,
    
    pub culling_compute_pipeline: wgpu::ComputePipeline,
    pub render_pipeline: wgpu::RenderPipeline,
        
    pub materials_bind_group: wgpu::BindGroup,
    pub camera_bind_group: wgpu::BindGroup,
    pub culling_compute_bind_group: wgpu::BindGroup,
    pub render_bind_group: wgpu::BindGroup,
}


impl SurfaceBindlessResources {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,        
        camera_uniform: &CameraUniform,
        depth_stencil: Option<wgpu::DepthStencilState>,        
        materials: &[MaterialData],
    ) -> Self {        
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mega Vertex Buffer"),
            size: MAX_VERTICES * size_of::<SurfaceVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mega Index Buffer"),
            size: MAX_INDICES * 4,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let meshlets_buffer_size = MAX_INSTANCES * size_of::<SurfaceMeshletDescription>() as u64; // (global_meshlets.len() * std::mem::size_of::<TerrainMeshletDescription>()) as wgpu::BufferAddress;

        let meshlets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain Meshlets Description Buffer"),
            size: meshlets_buffer_size,            
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let materials_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Materials Buffer"),
            size: MAX_MATERIALS * 64, 
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let buffer_size = (MAX_TERRAIN_CULLING_TASKS * std::mem::size_of::<SurfaceCullingTask>()) as u64;

        let culling_tasks_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain Culling Tasks Buffer"),
            size: buffer_size,            
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let active_meshlets_buffer_size = (MAX_MESHLETS_IN_SCENE * size_of::<u32>()) as wgpu::BufferAddress;
        
        let active_meshlets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain Active Meshlets Buffer"),
            size: active_meshlets_buffer_size,            
            usage: wgpu::BufferUsages::STORAGE, 
            mapped_at_creation: false,
        });

        let indirect_commands_template_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Commands Buffer"),
            size: (MAX_MESHLETS_IN_SCENE * size_of::<DrawIndexedIndirectCommand>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indirect_commands_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain Single Indirect Command Buffer"),
            size: (MAX_MESHLETS_IN_SCENE * size_of::<DrawIndexedIndirectCommand>()) as wgpu::BufferAddress,            
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let culling_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Culling Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/surface_bindless_culling.wgsl").into()),
        });

        let culling_compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Culling Compute Bind Group Layout"),
            entries: &[                
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },                
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },                
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },                
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                    
        let culling_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Culling Pipeline Layout"),
            bind_group_layouts: &[
                Some(&camera_bind_group_layout),
                Some(&culling_compute_bind_group_layout),
            ],            
            immediate_size: 0,
        });
        
        let culling_compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Culling Compute Pipeline"),
            layout: Some(&culling_pipeline_layout),
            module: &culling_shader,
            entry_point: Some("culling_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("model_bindless.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/surface_bindless.wgsl")))
        });

        let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Render Bind Group Layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer { // Было: wgpu::BindingType::Texture
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },  
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
            ],
        });
                
        let texture_count = std::num::NonZeroU32::new(MAX_TEXTURES);

        let materials_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Materials Bind Group Layout"),
            entries: &[                
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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
            label: Some("GPU Driven Render Pipeline Layout"),
            bind_group_layouts: &[
                Some(&materials_bind_group_layout), // @group(0)
                Some(&camera_bind_group_layout),    // @group(1)
                Some(&render_bind_group_layout), // @group(2)
            ],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Surface pipeline"),
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
        
        info!("Materials total: {}", materials.len());
        
        for (material_idx, md) in materials.iter().enumerate() {
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
            label: Some("Terrain Culling Compute Bind Group"),
            layout: &culling_compute_bind_group_layout,
            entries: &[                
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: culling_tasks_buffer.as_entire_binding(),
                },                
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: meshlets_buffer.as_entire_binding(),
                },                
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: active_meshlets_buffer.as_entire_binding(),
                },                
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: indirect_commands_buffer.as_entire_binding(),
                },
            ],
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain GPU Driven Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[                
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },                
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: index_buffer.as_entire_binding(),
                },                
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: meshlets_buffer.as_entire_binding(),
                },                
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: active_meshlets_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            vertex_buffer,
            index_buffer,
            meshlets_buffer,
            camera_buffer,
            materials_buffer,
            culling_tasks_buffer,
            indirect_commands_template_buffer,
            indirect_commands_buffer,
            culling_compute_pipeline,
            render_pipeline,
            materials_bind_group,
            active_meshlets_buffer,
            camera_bind_group,
            culling_compute_bind_group,
            render_bind_group,
        }
    }

    pub fn init(
        &self,
        queue: &wgpu::Queue,
        vertices: &[SurfaceVertex],
        indices: &[u32],
        meshlets: &[SurfaceMeshletDescription],
        material_factors: &[MaterialFactors],
        indirect_commands: &[DrawIndexedIndirectCommand]
    ) {        
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices));
        queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(indices));
        queue.write_buffer(&self.meshlets_buffer, 0, bytemuck::cast_slice(meshlets));
        queue.write_buffer(&self.materials_buffer, 0, bytemuck::cast_slice(material_factors));
        queue.write_buffer(&self.indirect_commands_template_buffer, 0, bytemuck::cast_slice(indirect_commands));
        queue.write_buffer(&self.indirect_commands_buffer, 0, bytemuck::cast_slice(indirect_commands));
    }

    pub fn load_frame(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        camera: &Camera,
        staging_belt: &mut StagingBelt,
        culling_tasks: &[SurfaceCullingTask],        
    ) {        
        {                                                                                            
            let camera_uniform = camera.get_uniform();
            let mut camera_slice = staging_belt.write_buffer(
                encoder,
                &self.camera_buffer,
                0,
                wgpu::BufferSize::new(CAMERA_UNIFORM_SIZE).expect("Failed to allocate bindless camera slice")
            );            
            camera_slice.copy_from_slice(bytemuck::bytes_of(&camera_uniform));
        }

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
        total_meshlets_count: u32,
    ) {
        compute_pass.set_pipeline(&self.culling_compute_pipeline);
        compute_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        compute_pass.set_bind_group(1, &self.culling_compute_bind_group, &[]);
        let workgroup_count = (total_meshlets_count + 63) / 64;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    pub fn draw_gpu_driven_frame(
        &self,
        render_pass: &mut wgpu::RenderPass,
        total_meshlets_count: u32,
    ) {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.materials_bind_group, &[]);
        render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(2, &self.render_bind_group, &[]);
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);                
        render_pass.multi_draw_indexed_indirect(
            &self.indirect_commands_buffer, 
            0, 
            total_meshlets_count
        );
    }
}
