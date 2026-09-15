use std::mem;
use std::ops::Mul;
use wgpu::{DepthStencilState, RenderPass, TextureFormat};
use glam::{Mat4, Vec3};

pub const MAX_QUADS_COUNT: u64 = 1000;

/// The properties of a quad.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct SolidQuad {
    /// The background color data of the quad.
    pub color: [f32; 4],

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

unsafe impl bytemuck::Zeroable for SolidQuad {}
unsafe impl bytemuck::Pod for SolidQuad {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Uniforms {
    pub transform: [f32; 16],
    pub scale: f32,
    pub _padding: f32,
    pub viewport_offset: [f32; 2],     
}

unsafe impl bytemuck::Pod for Uniforms {}
unsafe impl bytemuck::Zeroable for Uniforms {}

impl Uniforms {
    pub fn new(transformation: Transformation, scale: f32, viewport_offset: [f32; 2]) -> Uniforms {
        Self {
            transform: *transformation.as_ref(),
            scale,
            _padding: 0.0,
            viewport_offset,            
        }
    }
}
impl Default for Uniforms {
    fn default() -> Self {
        Self {
            transform: *Transformation::identity().as_ref(),
            scale: 1.0,
            _padding: 0.0,
            viewport_offset: [0.0, 0.0],            
        }
    }
}

#[derive(Debug)]
pub struct Pipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub uniform_bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub vertex_buffer: wgpu::Buffer
}

impl Pipeline {
    pub fn draw<'a>(&'a self, render_pass: &mut RenderPass<'a>, count: u32) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

        render_pass.draw(0..6, 0..count);
    }

    pub fn draw_range<'a>(&'a self, render_pass: &mut RenderPass<'a>, range_start: u32, range_end: u32) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

        render_pass.draw(0..6, range_start..range_end);
    }

    pub fn new(device: &wgpu::Device, depth_stencil: Option<DepthStencilState>) -> Pipeline {
        let constant_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Quad uniforms layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
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

        let layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Quad pipeline"),                
                bind_group_layouts: &[Some(&constant_layout)],
                immediate_size: 0
            });

        let shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Quad shader"),
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
                        include_str!("shaders/solid.wgsl")
                    ),
                )),
            });

        let pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Solid quad pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("solid_vs_main"),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<SolidQuad>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array!(                            
                            0 => Float32x4, // color                            
                            1 => Float32x2, // position                            
                            2 => Float32x2, // size                            
                            3 => Float32x4, // border color                            
                            4 => Float32x4, // border radius                            
                            5 => Float32,   // border width                            
                            6 => Float32x4, // shadow color                            
                            7 => Float32x2, // shadow offset                            
                            8 => Float32,   // shadow blur radius                            
                            9 => Uint32,    // snap                            
                            10 => Float32x4 // clip
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

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Quad instance buffer"),
            size: mem::size_of::<SolidQuad>() as u64 * MAX_QUADS_COUNT,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        Pipeline {
            pipeline,
            uniform_bind_group: constants,
            uniform_buffer: constants_buffer,        
            vertex_buffer
        }
    }    
}

/// A 2D transformation matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transformation(Mat4);

impl Transformation {
    /// Get the identity transformation.
    pub fn identity() -> Transformation {
        Transformation(Mat4::IDENTITY)
    }

    /// Creates an orthographic projection.
    #[rustfmt::skip]
    pub fn orthographic(width: u32, height: u32) -> Transformation {
        Transformation(Mat4::orthographic_rh_gl(
            0.0, width as f32,
            height as f32, 0.0,
            -1.0, 1.0
        ))
    }

    /// Creates a translate transformation.
    pub fn translate(x: f32, y: f32) -> Transformation {
        Transformation(Mat4::from_translation(Vec3::new(x, y, 0.0)))
    }

    /// Creates a scale transformation.
    pub fn scale(x: f32, y: f32) -> Transformation {
        Transformation(Mat4::from_scale(Vec3::new(x, y, 1.0)))
    }
}

impl Mul for Transformation {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Transformation(self.0 * rhs.0)
    }
}

impl AsRef<[f32; 16]> for Transformation {
    fn as_ref(&self) -> &[f32; 16] {
        self.0.as_ref()
    }
}

impl From<Transformation> for [f32; 16] {
    fn from(t: Transformation) -> [f32; 16] {
        *t.as_ref()
    }
}

impl From<Transformation> for Mat4 {
    fn from(transformation: Transformation) -> Self {
        transformation.0
    }
}
