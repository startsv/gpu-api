use std::borrow::Cow;
use gpu_api_relay::model_bindless_data::Vertex;
use wgpu::{Device, RenderPipeline, Buffer, BindGroup, ShaderModule, BindGroupLayout, PipelineLayout, TextureFormat, RenderPass, Sampler};
use wgpu::util::DeviceExt;
use gpu_api_relay::model_bindless_data::CameraUniform;
use crate::pipeline::model_pipeline::model_instance::ModelInstance;
use crate::texture::Texture;
use model::ObjectGroup;

pub mod model;
pub mod model_instance;

pub const CAMERA_UNIFORM_SIZE: u64 = 240;
pub const INSTANCE_SIZE: u64 = 80;
pub const MAX_MODEL_INSTANCES_COUNT: u64 = 100000;
pub const JOINT_MATRICES_COUNT: usize = 128;
pub const JOINT_MATRICES_UNIFORM_SIZE: u64 = 8192;
pub const NODE_TRANSFORM_UNIFORM_SIZE: u64 = 80;
pub const MATERIAL_FACTORS_UNIFORM_SIZE: u64 = 48;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct MaterialFactorsUniform {
    pub base_color_factor: [f32; 4],
    pub emissive_factor: [f32; 3],
    pub metallic_factor: f32,
    pub padding: [u32; 3],
    pub roughness_factor: f32
}

unsafe impl bytemuck::Pod for MaterialFactorsUniform {}
unsafe impl bytemuck::Zeroable for MaterialFactorsUniform {}

pub struct Pipeline {
    pub shader: ShaderModule,
    pub material_bind_group_layout: BindGroupLayout,
    pub base_color_sampler: Sampler,
    pub normal_sampler: Sampler,
    pub metallic_roughness_sampler: Sampler,
    pub emissive_sampler: Sampler,
    pub depth_texture: Texture,    
    pub camera_buffer: Buffer,
    pub camera_bind_group_layout: BindGroupLayout,
    pub camera_bind_group: BindGroup,
    pub joint_matrices_bind_group_layout: BindGroupLayout,
    pub node_transform_bind_group_layout: BindGroupLayout,
    pub pipeline_layout: PipelineLayout,
    pub swapchain_format: TextureFormat,
    pub render_pipeline: RenderPipeline
}

impl Pipeline {
    pub fn draw(&self, render_pass: &mut RenderPass, object_groups: &[ObjectGroup]) {
        render_pass.set_pipeline(&self.render_pipeline);

        for object_group in object_groups {
            if object_group.active == false {
                continue;
            }
            
            for object in &object_group.objects {
                if object.instances_count == 0 {
                    continue;
                }
                
                render_pass.set_vertex_buffer(1, object.instance_buffer.slice(..)); // Instances
                        
                let instances_range = 0..object.instances_count;
                
                for mesh in &object.meshes {
                    for primitive in &mesh.primitives {
                        render_pass.set_bind_group(0, &object.materials[primitive.material_index].material_bind_group, &[]); // Material
                        render_pass.set_bind_group(1, &self.camera_bind_group, &[]); // Camera
                        render_pass.set_bind_group(2, &object.joint_matrices_bind_group, &[]); // Joint matrices
                        render_pass.set_bind_group(3, &mesh.node_transform_bind_group, &[]); // Node transform
                        render_pass.set_vertex_buffer(0, primitive.vertex_buffer.slice(..));
                        render_pass.set_index_buffer(primitive.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        render_pass.draw_indexed(0..primitive.num_elements, 0, instances_range.clone());
                    }
                }
            }
        }
    }
}

pub fn new(device: &Device, config: &wgpu::SurfaceConfiguration, camera_uniform: &CameraUniform, depth_stencil: Option<wgpu::DepthStencilState>) -> Pipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("model.wgsl"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/model.wgsl")))
    });    

    let material_bind_group_layout = device.create_bind_group_layout(        
        &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,                        
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,                        
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,                        
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None
                        },
                        count: None
                    },
                    /*
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::FRAGMENT,                        
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None
                    }
                    */                   
                ],
                label: Some("texture_bind_group_layout"),
            });

    let base_color_sampler = device.create_sampler(&wgpu::SamplerDescriptor {            
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,            
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });

    let normal_sampler = device.create_sampler(&wgpu::SamplerDescriptor {            
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,            
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });

    

    let metallic_roughness_sampler = device.create_sampler(&wgpu::SamplerDescriptor {            
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,            
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });

    let emissive_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,            
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
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
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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

    let joint_matrices_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        ],
        label: Some("joint_matrices_group_layout"),
    });

    let node_transform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        ],
        label: Some("node_transform_bind_group_layout"),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[
            Some(&material_bind_group_layout),
            Some(&camera_bind_group_layout),
            Some(&joint_matrices_bind_group_layout),
            Some(&node_transform_bind_group_layout),
        ],
        immediate_size: 0
    });    

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Model pipeline"),
        layout: Some(&pipeline_layout),        
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[                        
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: (std::mem::size_of::<[f32; 3]>()) as wgpu::BufferAddress,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x2
                        },
                        wgpu::VertexAttribute {
                            offset: (
                                std::mem::size_of::<[f32; 3]>() + 
                                std::mem::size_of::<[f32; 2]>()
                            ) as wgpu::BufferAddress,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x3
                        },
                        wgpu::VertexAttribute {
                            offset: (
                                std::mem::size_of::<[f32; 3]>() + 
                                std::mem::size_of::<[f32; 2]>() +
                                std::mem::size_of::<[f32; 3]>()
                            ) as wgpu::BufferAddress,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32x3
                        },
                        wgpu::VertexAttribute {
                            offset: (
                                std::mem::size_of::<[f32; 3]>() + 
                                std::mem::size_of::<[f32; 2]>() +
                                std::mem::size_of::<[f32; 3]>() +
                                std::mem::size_of::<[f32; 3]>()
                            ) as wgpu::BufferAddress,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32x3
                        },
                        wgpu::VertexAttribute {
                            offset: (
                                std::mem::size_of::<[f32; 3]>() + 
                                std::mem::size_of::<[f32; 2]>() +
                                std::mem::size_of::<[f32; 3]>() + 
                                std::mem::size_of::<[f32; 3]>() +
                                std::mem::size_of::<[f32; 3]>()
                            ) as wgpu::BufferAddress,
                            shader_location: 5,
                            format: wgpu::VertexFormat::Uint32x4
                        },
                        wgpu::VertexAttribute {
                            offset: (
                                std::mem::size_of::<[f32; 3]>() + 
                                std::mem::size_of::<[f32; 2]>() +
                                std::mem::size_of::<[f32; 3]>() + 
                                std::mem::size_of::<[f32; 3]>() + 
                                std::mem::size_of::<[f32; 3]>() +
                                std::mem::size_of::<[u32; 4]>()
                            ) as wgpu::BufferAddress,
                            shader_location: 6,
                            format: wgpu::VertexFormat::Float32x4
                        }
                    ]
                }),
                Some(ModelInstance::vertex_buffer_layout())
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

    let depth_texture = Texture::create_depth_texture(device, config, "Depth texture");    

    Pipeline {
        shader,
        material_bind_group_layout,
        base_color_sampler,        
        normal_sampler,        
        metallic_roughness_sampler,
        emissive_sampler,
        depth_texture,        
        camera_buffer,
        camera_bind_group_layout,
        camera_bind_group,
        joint_matrices_bind_group_layout,
        node_transform_bind_group_layout,
        pipeline_layout,
        swapchain_format: TextureFormat::Rgba8UnormSrgb,
        render_pipeline      
    }
}
