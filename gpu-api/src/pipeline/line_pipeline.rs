use std::mem;
use wgpu::{DepthStencilState, RenderPass, TextureFormat, util::DeviceExt};
use gpu_api_relay::model_bindless_data::CameraUniform;

pub const LINE_VERTICES_COUNT: u64 = 20000;

/// The properties of a quad.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct LineVertex {
    /// The background color data of the quad.
    pub color: [f32; 4],

    /// The position of line point.
    pub pos: [f32; 3],
}

unsafe impl bytemuck::Zeroable for LineVertex {}
unsafe impl bytemuck::Pod for LineVertex {}

#[derive(Debug)]
pub struct Pipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub camera_bind_group: wgpu::BindGroup,
    pub camera_buffer: wgpu::Buffer,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
}

impl Pipeline {
    pub fn draw<'a>(&'a self, render_pass: &mut RenderPass<'a>, count: u32) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..count, 0, 0..1);        
    }    

    pub fn new(device: &wgpu::Device, camera_uniform: &CameraUniform, depth_stencil: Option<DepthStencilState>) -> Pipeline {        
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

        let layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Line pipeline"),                
                bind_group_layouts: &[
                    Some(&camera_bind_group_layout)
                ],
                immediate_size: 0
            });

        let shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Line shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    include_str!("shaders/line.wgsl")
                )),
            });

        let pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Line pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<LineVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array!(
                            // Color
                            0 => Float32x4,
                            // Position
                            1 => Float32x3,
                        )
                    })],
                    compilation_options:
                        wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
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
                    topology: wgpu::PrimitiveTopology::LineList,
                    front_face: wgpu::FrontFace::Ccw,
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
            label: Some("Line vertex buffer"),
            size: mem::size_of::<LineVertex>() as u64 * LINE_VERTICES_COUNT,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Line index buffer"),
            size: mem::size_of::<u32>() as u64 * LINE_VERTICES_COUNT,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        Pipeline {
            pipeline,
            camera_bind_group,
            camera_buffer,
            vertex_buffer,
            index_buffer,
        }
    }    
}
