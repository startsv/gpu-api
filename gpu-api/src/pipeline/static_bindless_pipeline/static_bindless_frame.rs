use std::borrow::Cow;
use gpu_api_dto::TextureType;
use gpu_api_relay::model_bindless_data::{CameraUniform, CullingTask, DrawIndexedIndirectCommand, MaterialFactors, NodeData, StaticVertex, VisibleInstanceData};
use log::info;
use wgpu::{TextureFormat, util::{DeviceExt, StagingBelt}};
use crate::{camera::CAMERA_UNIFORM_SIZE, pipeline::{model_pipeline::model::InitData, static_bindless_pipeline::{InstanceData, MeshInfo, NUM_FRAMES_IN_FLIGHT, StaticBindlessResources, StaticMeshletDescription, static_bindless_frame_res::FrameResources}}};

impl StaticBindlessResources {
    /// Вспомогательный метод для получения ресурсов текущего активного кадра
    pub fn current_frame(&self) -> &FrameResources {
        &self.frame_resources[self.frame_index % NUM_FRAMES_IN_FLIGHT]
    }

    /// Переключить индекс на следующий кадр (вызывается в самом конце рендер-петли)
    pub fn advance_frame(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }

    /// 1. Инициализация статических данных (Вызывается один раз при загрузке сцены)
    /// Данные пишутся строго в выделенные для них регионы монолитных буферов.
    pub fn init(
        &self, 
        queue: &wgpu::Queue,
        vertices: &[StaticVertex],               
        dummy_indices_template: &[u32],         // Сквозной шаблон [0, 1, 2...]
        meshlets: &[StaticMeshletDescription],  
        mesh_infos: &[MeshInfo],                
        material_factors: &[MaterialFactors],
        meshlet_local_indices: &[u32],          
        meshlet_vertex_redirect: &[u32],         
    ) {    
        // Запись в монолитный буфер геометрии (geometry_buffer)
        queue.write_buffer(&self.geometry_buffer, self.vertex_range.offset, bytemuck::cast_slice(vertices));
        queue.write_buffer(&self.geometry_buffer, self.index_range.offset, bytemuck::cast_slice(dummy_indices_template));
        
        // Запись в монолитный буфер структур сцены (scene_data_buffer)
        queue.write_buffer(&self.scene_data_buffer, self.mesh_infos_range.offset, bytemuck::cast_slice(mesh_infos));
        queue.write_buffer(&self.scene_data_buffer, self.meshlets_range.offset, bytemuck::cast_slice(meshlets));
        queue.write_buffer(&self.scene_data_buffer, self.meshlet_local_indices_range.offset, bytemuck::cast_slice(meshlet_local_indices));
        queue.write_buffer(&self.scene_data_buffer, self.meshlet_vertex_redirect_range.offset, bytemuck::cast_slice(meshlet_vertex_redirect));
        queue.write_buffer(&self.scene_data_buffer, self.materials_range.offset, bytemuck::cast_slice(material_factors));
    }

    /// 2. Загрузка динамических данных для конкретного кадра
    /// Данные инстансов и задач пишутся в scene_data_buffer, а камера — в изолированную зону текущего кадра внутри frame_ring_buffer.
    pub fn load_frame(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        camera_uniform: &CameraUniform,
        staging_belt: &mut wgpu::util::StagingBelt,
        instances: &[InstanceData],
        nodes: &[NodeData],
        culling_tasks: &[CullingTask], 
    ) {
        let current_ranges = &self.frame_ranges[self.frame_index % NUM_FRAMES_IN_FLIGHT];

        // 1. Загружаем данные камеры в кадровый срез монолитного frame_ring_buffer через StagingBelt
        {                                                                                                        
            let raw_camera_size = std::mem::size_of::<CameraUniform>() as u64;
            let camera_size_nonzero = std::num::NonZeroU64::new(raw_camera_size)
                .expect("Invalid camera uniform size");
                
            let mut camera_slice = staging_belt.write_buffer(
                encoder,
                &self.frame_ring_buffer,
                current_ranges.camera.offset, // Выровненный оффсет кадра
                camera_size_nonzero,
            );            
            camera_slice.copy_from_slice(bytemuck::bytes_of(camera_uniform));
        }

        // 2. Обновляем динамические списки объектов сцены строго в их срезы внутри scene_data_buffer
        queue.write_buffer(&self.scene_data_buffer, self.instances_range.offset, bytemuck::cast_slice(instances));

        // ФИКС: Пишем данные нод в scene_data_buffer по правильному оффсету
        queue.write_buffer(&self.scene_data_buffer, self.nodes_range.offset, bytemuck::cast_slice(nodes)); 

        if !culling_tasks.is_empty() {
            queue.write_buffer(&self.scene_data_buffer, self.culling_tasks_range.offset, bytemuck::cast_slice(culling_tasks));
        }
    }
    
    /// 3. Сброс счетчика команд GPU перед началом Culling прохода
    /// Зануляет u32 счетчик конкретного кадра внутри frame_ring_buffer.
    pub fn clear_gpu_driven_frame(&self, _queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        let current_ranges = &self.frame_ranges[self.frame_index % NUM_FRAMES_IN_FLIGHT];
        
        // Сбрасываем только 4 байта счетчика текущего кадра
        encoder.clear_buffer(
            &self.frame_ring_buffer, 
            current_ranges.counter.offset, 
            Some(4)
        );
    }

    /// 4. Выполнение Compute-пасса куллинга
    pub fn compute_gpu_driven_frame(
        &self,
        compute_pass: &mut wgpu::ComputePass<'_>,
        total_tasks_count: u32, 
    ) {        
        let frame_res = self.current_frame();
        
        compute_pass.set_pipeline(&self.culling_compute_pipeline);
        // Привязываем кадровый BindGroup, где все смещения срезов уже настроены на CPU
        compute_pass.set_bind_group(0, &frame_res.culling_compute_bind_group, &[]);
        
        // Потоки шейдера распределяются по количеству задач culling_tasks
        let workgroup_count = (total_tasks_count + 63) / 64;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    /// 5. Графическая отрисовка отсеянных мешлетов через MultiDrawIndexedIndirectCount
    pub fn draw_gpu_driven_frame(
        &self,
        render_pass: &mut wgpu::RenderPass,
    ) {
        let frame_res = self.current_frame();
        let current_ranges = &self.frame_ranges[self.frame_index % NUM_FRAMES_IN_FLIGHT];
        
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.materials_bind_group, &[]);        
        // Кадровая бинд-группа (нарезанные срезы из frame_ring_buffer и scene_data_buffer)
        render_pass.set_bind_group(1, &frame_res.render_bind_group, &[]);
        
        // Привязываем срез сквозного индексного буфера [0, 1, 2...] из geometry_buffer
        render_pass.set_index_buffer(
            self.geometry_buffer.slice(self.index_range.offset..(self.index_range.offset + self.index_range.size)), 
            wgpu::IndexFormat::Uint32
        );        
        
        // Вызываем отрисовку. Параметры команд и счетчика считываются прямо из срезов frame_ring_buffer.
        render_pass.multi_draw_indexed_indirect_count(
            &self.frame_ring_buffer,            
            current_ranges.indirect.offset,          // Смещение до начала массива команд текущего кадра
            &self.frame_ring_buffer,
            current_ranges.counter.offset,           // Смещение до u32 счетчика команд текущего кадра
            self.max_meshlets_count                  // Безопасный лимит команд
        );
    }    
}
