use std::mem;
use wgpu::{DepthStencilState, RenderPass, TextureFormat};
use image::DynamicImage;
use crate::pipeline::{solid_quad_pipeline::{Transformation, Uniforms}};

pub const MAX_IMAGE_QUADS_COUNT: u64 = 1000;

/// The properties of a quad.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ImageQuad {    
    /// The position of the [`Quad`].
    pub position: [f32; 2],

    /// The size of the [`Quad`].
    pub size: [f32; 2],

    /// The border color of the [`Quad`], in __linear RGB__.
    pub border_color: [f32; 4],

    /// The border radii of the [`Quad`].
    pub border_radius: [f32; 4],

    /// The border width of the [`Quad`].
    pub border_width: f32,

    /// The shadow color of the [`Quad`].
    pub shadow_color: [f32; 4],

    /// The shadow offset of the [`Quad`].
    pub shadow_offset: [f32; 2],

    /// The shadow blur radius of the [`Quad`].
    pub shadow_blur_radius: f32,

    /// Whether the [`Quad`] should be snapped to the pixel grid.
    pub snap: u32,

    /// Quad parts will be discarded if they are outside of component coordinates.
    pub clip: [f32; 4]    
}

unsafe impl bytemuck::Zeroable for ImageQuad {}
unsafe impl bytemuck::Pod for ImageQuad {}

#[derive(Debug)]
pub struct Pipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub image_bind_group_layout: wgpu::BindGroupLayout,
    pub image_sampler: wgpu::Sampler,
    pub uniform_bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer
}

pub struct ImageObject {
    pub name: String,
    pub is_active: bool,
    pub vertex_buffer: wgpu::Buffer,
    pub image_bind_group: wgpu::BindGroup,
    pub image_texture: crate::texture::Texture,
    pub quads: Vec<ImageQuad>
}

impl ImageObject {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, pipeline: &Pipeline, name: &str, is_active: bool, img: &DynamicImage, quads: Vec<ImageQuad>) -> ImageObject {        
        let image_texture = crate::texture::Texture::from_image(&device, &queue, img, &gpu_api_dto::ImageFormat::R8G8B8A8, false, Some("happy-tree.png")).expect("Failed to create texture from image");

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Quad instance buffer"),
            size: mem::size_of::<ImageQuad>() as u64 * MAX_IMAGE_QUADS_COUNT,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let image_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &pipeline.image_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&image_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&pipeline.image_sampler),
                    }
                ],
                label: Some("Image bind group"),
            }
        );

        ImageObject {
            name: name.to_owned(),
            is_active,
            vertex_buffer,
            image_bind_group,
            image_texture,
            quads
        }
    }
}

impl Pipeline {
    pub fn draw<'a>(&'a self, render_pass: &mut RenderPass<'a>, image_objects: &Vec<ImageObject>) {
        for image_object in image_objects {
            if image_object.is_active == false || image_object.quads.len() == 0 {
                continue;
            }            

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_bind_group(1, &image_object.image_bind_group, &[]);
            render_pass.set_vertex_buffer(0, image_object.vertex_buffer.slice(..));

            render_pass.draw(0..6, 0..image_object.quads.len() as u32);
        }        
    }

    pub fn new(device: &wgpu::Device, depth_stencil: Option<DepthStencilState>) -> Pipeline {
        let constant_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Quad uniforms layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            mem::size_of::<Uniforms>() as wgpu::BufferAddress,
                        ),
                    },
                    count: None,
                }],
            });

        let constants_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Quad uniforms buffer"),
            size: mem::size_of::<Uniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let constants = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Quad uniforms bind group"),
            layout: &constant_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: constants_buffer.as_entire_binding(),
            }],
        });        

        let image_bind_group_layout = device.create_bind_group_layout(        
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
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            // This should match the filterable field of the
                            // corresponding Texture entry above.
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        }
                    ],
                    label: Some("Image bind group layout")
                });

        let image_sampler = device.create_sampler(&wgpu::SamplerDescriptor {            
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        
        

        let layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Image pipeline"),                
                bind_group_layouts: &[
                    Some(&constant_layout),
                    Some(&image_bind_group_layout),
                ],
                immediate_size: 0
            });

        let shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Image shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    concat!(
                        include_str!("shaders/snap.wgsl"),
                        "\n",
                        include_str!("shaders/color.wgsl"),
                        "\n",
                        include_str!("shaders/quad.wgsl"),
                        "\n",
                        include_str!("shaders/vertex.wgsl"),
                        "\n",
                        include_str!("shaders/shadow.wgsl"),
                        "\n",
                        include_str!("shaders/image.wgsl")
                    ),
                )),
            });

        let pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Image pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("solid_vs_main"),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<ImageQuad>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array!(                            
                            // Position
                            0 => Float32x2,
                            // Size
                            1 => Float32x2,
                            // Border color
                            2 => Float32x4,
                            // Border radius
                            3 => Float32x4,
                            // Border width
                            4 => Float32,
                            // Shadow color
                            5 => Float32x4,
                            // Shadow offset
                            6 => Float32x2,
                            // Shadow blur radius
                            7 => Float32,
                            // Snap
                            8 => Uint32,
                            // Component coordinates
                            9 => Float32x4                            
                        )
                    })],
                    compilation_options:
                        wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("solid_fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
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
                })],
                    compilation_options:
                        wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Cw,
                    ..Default::default()
                },
                depth_stencil,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });        

        Pipeline {
            pipeline,
            image_bind_group_layout,
            image_sampler,
            uniform_bind_group: constants,
            uniform_buffer: constants_buffer            
        }
    }    
}
