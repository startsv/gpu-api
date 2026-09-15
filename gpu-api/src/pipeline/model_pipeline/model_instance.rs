use std::mem;
use glam::Mat4;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ModelInstance {
    pub model_matrix: Mat4,
    pub is_animated: u32,
    pub padding: [u32; 3], 
}

unsafe impl bytemuck::Pod for ModelInstance {}
unsafe impl bytemuck::Zeroable for ModelInstance {}

impl ModelInstance {
    pub fn vertex_buffer_layout<'a>() -> wgpu::VertexBufferLayout<'a> {        
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ModelInstance>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[                
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Uint32
                }                
            ]
        }
    }
}
