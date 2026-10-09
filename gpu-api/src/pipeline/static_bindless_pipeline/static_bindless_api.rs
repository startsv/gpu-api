use std::borrow::Cow;
use glam::Mat4;
use gpu_api_dto::TextureType;
use gpu_api_relay::model_bindless_data::{CameraUniform, CullingTask, DrawIndexedIndirectCommand, MaterialFactors, NodeData, StaticVertex, Vertex, VisibleInstanceData};
use log::info;
use wgpu::{ComputePass, RenderPass, TextureFormat, util::{DeviceExt, StagingBelt}};
use crate::{camera::CAMERA_UNIFORM_SIZE, pipeline::{model_pipeline::model::InitData, static_bindless_pipeline::{InstanceData, MeshletData, StaticBindlessResources}}};

impl StaticBindlessResources {
    /*
    /// Вспомогательный метод для получения ресурсов текущего активного кадра
    pub fn current_frame(&self) -> &FrameBindgroups {
        &self.frame_resources[self.frame_index % NUM_FRAMES_IN_FLIGHT]
    }

    /// Переключить индекс на следующий кадр (вызывается в самом конце рендер-петли)
    pub fn advance_frame(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }
    */

    pub fn init(
        &self,
        queue: &wgpu::Queue,
        vertices: &[Vertex],
        indices: &[u32],
        material_factors: &[MaterialFactors],
        indirect_commands: &[DrawIndexedIndirectCommand],
        meshlets: &[MeshletData]
    ) {
        queue.write_buffer(&self.mega_vertex_buffer, 0, bytemuck::cast_slice(vertices));
        queue.write_buffer(&self.mega_index_buffer, 0, bytemuck::cast_slice(indices));
        queue.write_buffer(&self.materials_buffer, 0, bytemuck::cast_slice(material_factors));
        queue.write_buffer(&self.indirect_commands_template_buffer, 0, bytemuck::cast_slice(indirect_commands));
        queue.write_buffer(&self.indirect_commands_buffer, 0, bytemuck::cast_slice(indirect_commands));
        queue.write_buffer(&self.global_meshlets_buffer, 0, bytemuck::cast_slice(&meshlets));
    }
      
    pub fn load_frame(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        camera_uniform: &CameraUniform,
        staging_belt: &mut StagingBelt,
        instances: &[InstanceData],
        nodes: &[NodeData],
        //joints: &[Mat4],
        culling_tasks: &[CullingTask],
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
        //queue.write_buffer(&self.joints_buffer, 0, bytemuck::cast_slice(joints));

        if culling_tasks.is_empty() == false {
            queue.write_buffer(&self.culling_tasks_buffer, 0, bytemuck::cast_slice(culling_tasks));
        }
    }

    pub fn clear_gpu_driven_frame(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) {        
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
    ) {        
        compute_pass.set_pipeline(&self.culling_compute_pipeline);
        compute_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        compute_pass.set_bind_group(1, &self.culling_compute_bind_group, &[]);
        let workgroup_count = (100 + 63) / 64;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    pub fn draw_gpu_driven_frame(
        &self,
        render_pass: &mut RenderPass,
        commands_len: u32
    ) {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.materials_bind_group, &[]);
        render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(2, &self.render_bind_group, &[]);        
        render_pass.set_vertex_buffer(0, self.mega_vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.mega_index_buffer.slice(..), wgpu::IndexFormat::Uint32);        
        render_pass.multi_draw_indexed_indirect(&self.indirect_commands_buffer, 0, commands_len);
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
