use std::borrow::Cow;
use glam::Mat4;
use gpu_api_dto::TextureType;
use gpu_api_relay::model_bindless_data::{CullingTask, DrawIndexedIndirectCommand, InstanceData, MaterialFactors, NodeData, PrimitiveMeta, Vertex, VisibleInstanceData};
use log::info;
use wgpu::{ComputePass, RenderPass, TextureFormat, util::{DeviceExt, StagingBelt}};
use crate::{camera::Camera, pipeline::{clear_commands_pipeline::{self, ClearCommandsPipeline}, model_pipeline::{CAMERA_UNIFORM_SIZE, model::InitData}}};
use gpu_api_relay::model_bindless_data::CameraUniform;

pub const MAX_VERTICES: u64 = 1_000_000;
pub const MAX_INDICES: u64 = 3_000_000;
pub const MAX_INSTANCES: u64 = 100_000;
pub const MAX_MATERIALS: u64 = 1_000;
pub const MAX_TEXTURES: u32 = 256;

// TODO: Заменить Compute Clear Pass на аппаратный command_encoder.clear_buffer.
// Для этого потребуется:
// 1. Вынести `instance_count` из структуры `DrawIndexedIndirectCmd` в отдельный плотный `draw_counts_buffer: array<atomic<u32>>`.
// 2. Инициализировать константы (index_count, и т.д.) в indirect_commands_buffer один раз, а не перезаписывать их.
// 3. Включить фичу `wgpu::Features::MULTI_DRAW_INDIRECT_COUNT` при инициализации устройства.
// 4. Заменить `multi_draw_indexed_indirect` на `multi_draw_indexed_indirect_count`.


pub struct ModelBindlessResources {    
    pub mega_vertex_buffer: wgpu::Buffer,
    pub mega_index_buffer: wgpu::Buffer,
    
    pub camera_buffer: wgpu::Buffer,
    pub instances_buffer: wgpu::Buffer,
    pub nodes_buffer: wgpu::Buffer,
    //pub joints_buffer: wgpu::Buffer,
    pub materials_buffer: wgpu::Buffer,
    
    pub culling_tasks_buffer: wgpu::Buffer,    
    pub visible_instances_buffer: wgpu::Buffer,
    
    pub indirect_commands_template_buffer: wgpu::Buffer,
    pub indirect_commands_buffer: wgpu::Buffer,
    
    pub culling_compute_pipeline: wgpu::ComputePipeline,
    //pub clear_commands_pipeline: ClearCommandsPipeline,
    pub render_pipeline: wgpu::RenderPipeline,
    
    pub materials_bind_group: wgpu::BindGroup,
    pub camera_bind_group: wgpu::BindGroup,
    pub culling_compute_bind_group: wgpu::BindGroup,
    pub render_bind_group: wgpu::BindGroup,
}

impl ModelBindlessResources {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,        
        camera_uniform: &CameraUniform,
        depth_stencil: Option<wgpu::DepthStencilState>,
        primitives_count: usize,        
        init_data: &mut InitData,
    ) -> Self {                        
        let mega_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mega Vertex Buffer"),
            size: MAX_VERTICES * size_of::<Vertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mega_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
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

        let visible_instances_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visible Instances Buffer"),            
            size: MAX_INSTANCES * (size_of::<VisibleInstanceData>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });        

        let indirect_commands_template_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Commands Buffer"),
            size: (primitives_count * 3 * size_of::<DrawIndexedIndirectCommand>()) as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indirect_commands_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Commands Buffer"),
            size: (primitives_count * 3 * size_of::<DrawIndexedIndirectCommand>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let culling_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Culling Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/model_bindless_culling.wgsl").into()),
        });

        let culling_compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Culling Compute Bind Group Layout"),
            entries: &[
                // Binding 0: Culling Tasks
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
                // Binding 1: Instances
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
                // Binding 2: Visible Instance Indices                
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
                // Binding 3: Indirect Commands Buffer
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
            label: Some("Culling Compute Pipeline"),
            layout: Some(&culling_pipeline_layout),
            module: &culling_shader,
            entry_point: Some("culling_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("model_bindless.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/model_bindless.wgsl")))
        });

        let gpu_driven_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GPU Driven Render Bind Group Layout"),
            entries: &[
                // Binding 0: Nodes
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
                // Binding 1: Joint Matrices
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    /*
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    */
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 2: Global Instances - InstanceData
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
                // Binding 3: Visible Instance Indices
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
            label: Some("GPU Driven Render Pipeline Layout"),
            bind_group_layouts: &[
                Some(&materials_bind_group_layout), // @group(0)
                Some(&camera_bind_group_layout),    // @group(1)
                Some(&gpu_driven_bind_group_layout), // @group(2)
            ],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Model pipeline"),
            layout: Some(&render_pipeline_layout),        
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array!(                            
                            0 => Float32x3,                            
                            1 => Float32x2,                            
                            2 => Float32x3,                            
                            3 => Float32x3,                            
                            4 => Float32x3,                            
                            5 => Uint32x4,                            
                            6 => Float32x4,
                        ),                                        
                    }),
                ]
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
            label: Some("Culling Compute Bind Group"),
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
                    resource: visible_instances_buffer.as_entire_binding(),
                },                
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: indirect_commands_buffer.as_entire_binding(),
                },
            ],
        });

        let matrix_texture_view = Self::load_matrices_into_texture(device, queue, &mut init_data.joints);

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GPU Driven Render Bind Group"),
            layout: &gpu_driven_bind_group_layout, // @group(2)
            entries: &[                
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: nodes_buffer.as_entire_binding(),
                },                
                wgpu::BindGroupEntry {
                    binding: 1,
                    //resource: joints_buffer.as_entire_binding(),
                    resource: wgpu::BindingResource::TextureView(&matrix_texture_view),
                },                
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: instances_buffer.as_entire_binding(),
                },                
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: visible_instances_buffer.as_entire_binding(),
                },
            ],
        });

        //let clear_commands_pipeline = ClearCommandsPipeline::new(device, &indirect_commands_buffer, commands_count as u32);

        Self {    
            mega_vertex_buffer,
            mega_index_buffer,
            camera_buffer,
            instances_buffer,
            nodes_buffer,
            //joints_buffer,
            materials_buffer,
            culling_tasks_buffer,
            visible_instances_buffer,
            indirect_commands_template_buffer,
            indirect_commands_buffer,
            culling_compute_pipeline,
            //clear_commands_pipeline,
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
        vertices: &[Vertex],
        indices: &[u32],
        material_factors: &[MaterialFactors],
        indirect_commands: &[DrawIndexedIndirectCommand]
    ) {
        queue.write_buffer(&self.mega_vertex_buffer, 0, bytemuck::cast_slice(vertices));
        queue.write_buffer(&self.mega_index_buffer, 0, bytemuck::cast_slice(indices));
        queue.write_buffer(&self.materials_buffer, 0, bytemuck::cast_slice(material_factors));
        queue.write_buffer(&self.indirect_commands_template_buffer, 0, bytemuck::cast_slice(indirect_commands));
        queue.write_buffer(&self.indirect_commands_buffer, 0, bytemuck::cast_slice(indirect_commands));
    }
  
    pub fn load_matrices_into_texture(device: &wgpu::Device, queue: &wgpu::Queue, joint_matrices: &mut Vec<Mat4>) -> wgpu::TextureView {
        let total_matrices = joint_matrices.len() as u32;

        // JOINT_MATRICES_COUNT = 128
        let texture_width = 2048u32;   // 4 128 (512)
        let matrices_per_row = 512u32; // 2048 / 4
        
        let texture_height = ((total_matrices as f32) / (matrices_per_row as f32)).ceil() as u32;
        let texture_height = texture_height.max(1);
        
        let required_total_matrices = (matrices_per_row * texture_height) as usize;
        
        if joint_matrices.len() < required_total_matrices {
            joint_matrices.resize(required_total_matrices, Mat4::IDENTITY);
        }

        let matrix_texture_size = wgpu::Extent3d {
            width: texture_width,
            height: texture_height,
            depth_or_array_layers: 1,
        };

        // HDR
        let matrix_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Global Joint Matrix Texture"),
            size: matrix_texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let matrix_bytes: &[u8] = bytemuck::cast_slice(joint_matrices);

        let bytes_per_pixel = 16u32;
        let bytes_per_row = texture_width * bytes_per_pixel;

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                texture: &matrix_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            matrix_bytes, 
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),     
                rows_per_image: Some(texture_height),   
            },
            matrix_texture_size,
        );

        matrix_texture.create_view(&wgpu::TextureViewDescriptor::default())        
    }

    pub fn load_frame(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        camera: &Camera,
        staging_belt: &mut StagingBelt,
        instances: &[InstanceData],
        nodes: &[NodeData],
        //joints: &[Mat4],
        culling_tasks: &[CullingTask],
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

        queue.write_buffer(&self.instances_buffer, 0, bytemuck::cast_slice(instances));
        queue.write_buffer(&self.nodes_buffer, 0, bytemuck::cast_slice(nodes));
        //queue.write_buffer(&self.joints_buffer, 0, bytemuck::cast_slice(joints));

        if culling_tasks.is_empty() == false {
            queue.write_buffer(&self.culling_tasks_buffer, 0, bytemuck::cast_slice(culling_tasks));
        }
    }

    pub fn clear_gpu_driven_frame(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        //compute_pass: &mut ComputePass,        
    ) {
        //self.clear_commands_pipeline.compute(compute_pass);
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
        compute_pass: &mut ComputePass,
        culling_tasks: &[CullingTask]
    ) {        
        compute_pass.set_pipeline(&self.culling_compute_pipeline);
        compute_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        compute_pass.set_bind_group(1, &self.culling_compute_bind_group, &[]);
        compute_pass.dispatch_workgroups(culling_tasks.len() as u32, 1, 1);
    }

    pub fn draw_gpu_driven_frame(
        &self,
        render_pass: &mut RenderPass,
        commands: &[DrawIndexedIndirectCommand]
    ) {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.materials_bind_group, &[]);
        render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(2, &self.render_bind_group, &[]);        
        render_pass.set_vertex_buffer(0, self.mega_vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.mega_index_buffer.slice(..), wgpu::IndexFormat::Uint32);        
        render_pass.multi_draw_indexed_indirect(&self.indirect_commands_buffer, 0, commands.len() as u32);
    }
}
