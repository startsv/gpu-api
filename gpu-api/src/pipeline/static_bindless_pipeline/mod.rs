use std::borrow::Cow;
use glam::Mat4;
use gpu_api_dto::TextureType;
use gpu_api_relay::model_bindless_data::{CullingTask, DrawIndexedIndirectCommand, InstanceData, MaterialFactors, NodeData, Vertex, VisibleInstanceData};
use log::info;
use wgpu::{ComputePass, RenderPass, TextureFormat, util::{DeviceExt, StagingBelt}};
use crate::{camera::{CAMERA_UNIFORM_SIZE, Camera}, pipeline::model_pipeline::model::InitData};
use gpu_api_relay::model_bindless_data::CameraUniform;

pub const MAX_VERTICES: u64 = 1_000_000;
pub const MAX_INDICES: u64 = 3_000_000;
pub const MAX_INSTANCES: u64 = 100_000;
pub const MAX_MATERIALS: u64 = 1_000;
pub const MAX_TEXTURES: u32 = 256;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshletData {
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub index_offset: u32,
    pub triangle_count: u32,
    
    pub instance_id: u32,
    pub bounding_center_x: f32,
    pub bounding_center_y: f32,
    pub bounding_center_z: f32,
    
    pub bounding_radius: f32,
    // Выравнивание структуры до кратности 16 байтам (4 байта * 12 полей = 48 байт)
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VisibleMeshletData {
    pub meshlet_id: u32,
    pub material_index: u32,
}


pub struct StaticBindlessResources {    
    pub mega_vertex_buffer: wgpu::Buffer,
    pub mega_index_buffer: wgpu::Buffer,
    
    pub camera_buffer: wgpu::Buffer,
    pub instances_buffer: wgpu::Buffer,
    pub nodes_buffer: wgpu::Buffer,
    //pub joints_buffer: wgpu::Buffer,
    pub materials_buffer: wgpu::Buffer,
    
    pub culling_tasks_buffer: wgpu::Buffer,    
    
    // === ИЗМЕНЕНИЯ И НОВЫЕ БУФЕРЫ ДЛЯ МЕШЛЕТОВ ===
    
    /// Глобальный массив ВСЕХ мешлетов сцены (описания их геометрии и bounding-сфер).
    /// Шейдер куллинга читает его, чтобы знать параметры каждого мешлета.
    pub global_meshlets_buffer: wgpu::Buffer,

    /// Заменяет старый `visible_instances_buffer`. 
    /// Сюда Compute-шейдер записывает пары `(meshlet_id, material_index)` для прошедших куллинг мешлетов.
    pub visible_meshlets_buffer: wgpu::Buffer,
    
    /// НОВЫЙ БУФЕР: Атомарный счетчик на GPU (размер 4 байта / u32).
    /// Хранит текущее количество видимых мешлетов. Обнуляется перед куллингом.
    /// Используется шейдером для `atomicAdd`, а также в `multi_draw_indexed_indirect_count`.
    pub global_draw_counter_buffer: wgpu::Buffer,
    
    // ПОДСПУДНОЕ УДАЛЕНИЕ:
    // pub indirect_commands_template_buffer: wgpu::Buffer, // БОЛЬШЕ НЕ НУЖЕН: команды генерируются на GPU с нуля
    
    /// Буфер для команд Multi-Draw Indirect. 
    /// Теперь его размер должен быть равен общему числу мешлетов в сцене (`global_meshlets.len() * 20` байт).
    pub indirect_commands_buffer: wgpu::Buffer,
    
    // =============================================

    pub culling_compute_pipeline: wgpu::ComputePipeline,
    //pub clear_commands_pipeline: ClearCommandsPipeline, // Больше не нужен, заменен на encoder.clear_buffer
    pub render_pipeline: wgpu::RenderPipeline,
    
    pub materials_bind_group: wgpu::BindGroup,
    pub camera_bind_group: wgpu::BindGroup,
    pub culling_compute_bind_group: wgpu::BindGroup,
    pub render_bind_group: wgpu::BindGroup,
}

impl StaticBindlessResources {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,        
        camera_uniform: &CameraUniform,
        depth_stencil: Option<wgpu::DepthStencilState>,
        total_instances: usize,
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

        // Условное максимальное количество мешлетов, которое мы можем обрабатывать на GPU за кадр
        let max_total_meshlets = total_instances * 2;

        let indirect_commands_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Commands Buffer"),
            size: (max_total_meshlets * size_of::<DrawIndexedIndirectCommand>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Задайте максимальный лимит мешлетов, который может содержать сцена/уровень
        let max_meshlets = 500_000_u64; 

        // 1. GLOBAL MESHLETS BUFFER
        // Хранит статичные геометрические параметры и сферы всех мешлетов всех моделей.
        let global_meshlets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Global Meshlets Buffer"),
            size: max_meshlets * std::mem::size_of::<MeshletData>() as u64,
            // STORAGE: Читается в Compute и Vertex шейдерах
            // COPY_DST: Позволяет загружать данные с CPU через queue.write_buffer
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 2. VISIBLE MESHLETS BUFFER
        // Сюда Compute-шейдер записывает индексы прошедших куллинг мешлетов.
        let visible_meshlets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visible Meshlets Buffer"),
            size: max_meshlets * std::mem::size_of::<VisibleMeshletData>() as u64,
            // STORAGE: Шейдер куллинга пишет в него (Read/Write), а Vertex-шейдер читает (Read)
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // 3. GLOBAL DRAW COUNTER BUFFER
        // Атомарный счетчик на GPU. Хранит ровно одно число u32 (4 байта).
        let global_draw_counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Global Draw Counter Buffer"),
            size: 4, // 1 * std::mem::size_of::<u32>()
            // STORAGE: Шейдер увеличивает значение через atomicAdd
            // COPY_DST: Позволяет обнулять буфер на CPU перед кадром через encoder.clear_buffer
            // INDIRECT: Позволяет использовать буфер как аргумент подсчета в multi_draw_indexed_indirect_count
            usage: wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::COPY_DST 
                | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        
        let culling_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Culling Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/static_bindless_culling.wgsl").into()),
        });

        let culling_compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    label: Some("Culling Compute Bind Group Layout"),
    entries: &[
        // Binding 0: Culling Tasks (Read-only)
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
        // Binding 1: Global Instances (Read-only)
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
        // Binding 2: Global Meshlets Geometry & Spheres Data (Read-only) - НОВЫЙ СЛОТ
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        // Binding 3: Visible Meshlets Output (Read/Write) - ЗАМЕНИЛ visible_instances
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
        // Binding 4: Indirect Commands Buffer Output (Read/Write)
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        // Binding 5: Global Atomic Draw Counter (Read/Write) - НОВЫЙ СЛОТ
        wgpu::BindGroupLayoutEntry {
            binding: 5,
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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/static_bindless.wgsl")))
        });

        let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    label: Some("Bindless Render Bind Group Layout"),
    entries: &[
        // Binding 0: Nodes (Read-only)
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
        // Binding 1: Joint Baked Texture
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        // Binding 2: Global Instances - InstanceData (Read-only)
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
        // Binding 3: Visible Meshlets Data - ЗАМЕНИЛ visible_instances (Read-only)
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
        // Binding 4: Global Meshlets Geometry Reference Buffer (Read-only) - НОВЫЙ СЛОТ
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
            label: Some("GPU Driven Render Pipeline Layout"),
            bind_group_layouts: &[
                Some(&materials_bind_group_layout), // @group(0)
                Some(&camera_bind_group_layout),    // @group(1)
                Some(&render_bind_group_layout), // @group(2)
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
                // Binding 0: Задачи куллинга мешлетов
                wgpu::BindGroupEntry { 
                    binding: 0, 
                    resource: culling_tasks_buffer.as_entire_binding() 
                },                
                // Binding 1: Все инстансы объектов сцены
                wgpu::BindGroupEntry { 
                    binding: 1, 
                    resource: instances_buffer.as_entire_binding() 
                },                
                // Binding 2: Геометрия и сферы всех мешлетов (Новый)
                wgpu::BindGroupEntry { 
                    binding: 2, 
                    resource: global_meshlets_buffer.as_entire_binding() 
                },                
                // Binding 3: Выходной буфер видимых мешлетов (Заменил visible_instances)
                wgpu::BindGroupEntry { 
                    binding: 3, 
                    resource: visible_meshlets_buffer.as_entire_binding() 
                },
                // Binding 4: Выходной буфер indirect-команд для Multi-Draw
                wgpu::BindGroupEntry { 
                    binding: 4, 
                    resource: indirect_commands_buffer.as_entire_binding() 
                },
                // Binding 5: Глобальный счетчик видимых мешлетов (Новый)
                wgpu::BindGroupEntry { 
                    binding: 5, 
                    resource: global_draw_counter_buffer.as_entire_binding() 
                },
            ],
        });

        let matrix_texture_view = Self::load_matrices_into_texture(device, queue, &mut init_data.joints);

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GPU Driven Render Bind Group"),
            layout: &render_bind_group_layout, // @group(2)
            entries: &[                
                // Binding 0: Данные нод (трансформации костей скелета)
                wgpu::BindGroupEntry { 
                    binding: 0, 
                    resource: nodes_buffer.as_entire_binding() 
                },                
                // Binding 1: Текстура запеченной анимации суставов
                wgpu::BindGroupEntry { 
                    binding: 1, 
                    resource: wgpu::BindingResource::TextureView(&matrix_texture_view) 
                },                
                // Binding 2: Данные о всех инстансах объектов
                wgpu::BindGroupEntry { 
                    binding: 2, 
                    resource: instances_buffer.as_entire_binding() 
                },                
                // Binding 3: Список видимых мешлетов (Заменил visible_instances)
                wgpu::BindGroupEntry { 
                    binding: 3, 
                    resource: visible_meshlets_buffer.as_entire_binding() 
                },
                // Binding 4: Глобальный буфер геометрии и метаданных мешлетов (Новый)
                wgpu::BindGroupEntry { 
                    binding: 4, 
                    resource: global_meshlets_buffer.as_entire_binding() 
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
            global_draw_counter_buffer,
            global_meshlets_buffer,
            visible_meshlets_buffer,
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
}


impl StaticBindlessResources {
    pub fn init(
        &self,
        queue: &wgpu::Queue,
        vertices: &[Vertex],
        indices: &[u32],
        material_factors: &[MaterialFactors],
        // Передаем сгенерированный на CPU массив данных мешлетов вместо старых indirect команд
        global_meshlets: &[MeshletData] 
    ) {
        // 1. Загружаем геометрию в мега-буферы (без изменений)
        queue.write_buffer(&self.mega_vertex_buffer, 0, bytemuck::cast_slice(vertices));
        queue.write_buffer(&self.mega_index_buffer, 0, bytemuck::cast_slice(indices));
        
        // 2. Загружаем параметры материалов (без изменений)
        queue.write_buffer(&self.materials_buffer, 0, bytemuck::cast_slice(material_factors));

        // 3. Загружаем метаданные и bounding-сферы мешлетов в глобальный буфер
        if !global_meshlets.is_empty() {
            queue.write_buffer(
                &self.global_meshlets_buffer, 
                0, 
                bytemuck::cast_slice(global_meshlets)
            );
        }
        
        // ЗАМЕЧАНИЕ: 
        // self.indirect_commands_template_buffer — УДАЛЕН.
        // self.indirect_commands_buffer — больше не заполняется с CPU, так как 
        // GPU генерирует команды динамически во время фазы куллинга.
    }


    pub fn load_frame(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        camera_uniform: &CameraUniform,
        staging_belt: &mut wgpu::util::StagingBelt,
        instances: &[InstanceData],
        nodes: &[NodeData],
        culling_tasks: &[CullingTask],
        // Передаем мешлеты, если они изменились/динамические (опционально)
        // global_meshlets: &[Meshlet], 
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

    pub fn clear_gpu_driven_frame(
        &self,
        encoder: &mut wgpu::CommandEncoder,       
    ) {
        // Вместо копирования шаблона команд, мы просто обнуляем глобальный счетчик видимых мешлетов.
        // Шейдер куллинга сам запишет команды в indirect_commands_buffer, начиная с 0-го индекса.
        encoder.clear_buffer(&self.global_draw_counter_buffer, 0, None);
        
        // Опционально: если вы НЕ используете multi_draw_indexed_indirect_count, 
        // имеет смысл очистить и сам буфер команд, чтобы не отрисовать старые мешлеты с прошлого кадра,
        // хотя выставление правильного лимита в draw_gpu_driven_frame обычно это решает.
        // encoder.clear_buffer(&self.indirect_commands_buffer, 0, None);
    }

    pub fn compute_gpu_driven_frame(
        &self,
        compute_pass: &mut wgpu::ComputePass,        
        culling_tasks_count: u32, // Передаем реальное количество задач в этом кадре
    ) {        
        compute_pass.set_pipeline(&self.culling_compute_pipeline);
        compute_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        compute_pass.set_bind_group(1, &self.culling_compute_bind_group, &[]);
        
        // Количество рабочих групп теперь строго равно количеству задач куллинга (1 task = 1 workgroup)
        // Внутри группы 64 потока будут параллельно перебирать мешлеты этой задачи
        if culling_tasks_count > 0 {
            compute_pass.dispatch_workgroups(culling_tasks_count, 1, 1);
        }
    }

    pub fn draw_gpu_driven_frame(
        &self,
        render_pass: &mut wgpu::RenderPass,
        max_commands_len: u32 // Максимальная вместимость вашего indirect_commands_buffer (общее число мешлетов в сцене)
    ) {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.materials_bind_group, &[]);
        render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(2, &self.render_bind_group, &[]);        
        
        render_pass.set_vertex_buffer(0, self.mega_vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.mega_index_buffer.slice(..), wgpu::IndexFormat::Uint32);        
        
        // ВАРИАНТ А: У вас включена фича MULTI_DRAW_INDIRECT_COUNT (Рекомендуется)
        // Видеокарта сама возьмет точное число сгенерированных команд из global_draw_counter_buffer
        #[cfg(feature = "use_mdi_count")]
        {
            render_pass.multi_draw_indexed_indirect_count(
                &self.indirect_commands_buffer,
                0,
                &self.global_draw_counter_buffer,
                0,
                max_commands_len, // Ограничитель сверху во избежание переполнения GPU
            );
        }

        // ВАРИАНТ Б: Стандартный Multi-Draw Indirect (Без MDI Count расширения)
        // Если расширение недоступно, передаем max_commands_len (размер буфера). 
        // Чтобы видеокарта не рисовала «пустые» слоты, ваш Compute-шейдер при инициализации буфера 
        // (или метод clear_gpu_driven_frame) должен гарантировать, что у невидимых мешлетов instance_count == 0.
        #[cfg(not(feature = "use_mdi_count"))]
        {
            render_pass.multi_draw_indexed_indirect(
                &self.indirect_commands_buffer, 
                0, 
                max_commands_len
            );
        }
    }
}
