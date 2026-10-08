use std::borrow::Cow;
use gpu_api_dto::TextureType;
use gpu_api_relay::model_bindless_data::{CameraUniform, CullingTask, DrawIndexedIndirectCommand, MaterialFactors, NodeData, StaticVertex, VisibleInstanceData};
use log::info;
use wgpu::{TextureFormat, util::{DeviceExt, StagingBelt}};
use crate::{camera::CAMERA_UNIFORM_SIZE, pipeline::{model_pipeline::model::InitData, static_bindless_pipeline::NUM_FRAMES_IN_FLIGHT}};

pub fn align_to(size: u64, alignment: u64) -> u64 {
    (size + alignment - 1) & !(alignment - 1)
}

#[derive(Clone, Copy, Debug)]
pub struct BufferRange {
    pub offset: u64,
    pub size: u64,
}

/// Смещения динамических ресурсов внутри FrameRingBuffer для конкретного кадра
#[derive(Debug)]
pub struct FrameResourceRanges {
    pub counter: BufferRange,
    pub indirect: BufferRange,
    pub visible_instances: BufferRange,
    pub camera: BufferRange,
}

pub struct FrameResources {
    // Вся кадровые данные теперь находятся внутри единого монолитного frame_ring_buffer,
    // но для удобства мы сохраняем готовые BindGroup, нарезающие этот буфер на срезы.
    pub culling_compute_bind_group: wgpu::BindGroup,
    pub render_bind_group: wgpu::BindGroup,
}

impl FrameResources {
    pub fn create(
        device: &wgpu::Device,
        // Layout'ы пайплайнов для сборки BindGroup
        culling_compute_layout: &wgpu::BindGroupLayout,
        render_bind_group_layout: &wgpu::BindGroupLayout,
        // Ссылки на 2 основных монолитных буфера, откуда нарезаются срезы
        scene_data_buffer: &wgpu::Buffer,
        frame_ring_buffer: &wgpu::Buffer,
        // Глобальные статические диапазоны памяти сцены
        culling_tasks_range: &BufferRange,
        instances_range: &BufferRange,
        mesh_infos_range: &BufferRange,
        meshlets_range: &BufferRange,
        nodes_range: &BufferRange,
        meshlet_local_indices_range: &BufferRange,
        meshlet_vertex_redirect_range: &BufferRange,
        // Массив просчитанных кадровых смещений (по одному на каждый NUM_FRAMES_IN_FLIGHT)
        frame_ranges: &[FrameResourceRanges],
    ) -> Vec<Self> {
        let mut frame_resources = Vec::new();

        for i in 0..NUM_FRAMES_IN_FLIGHT {
            let ranges = &frame_ranges[i];

            // 1. Сборка монолитной Bind Group для Compute-пасса куллинга текущего кадра
            let culling_compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Culling Compute Bind Group Frame {}", i)),
                layout: culling_compute_layout,
                entries: &[                
                    // Камера текущего кадра (Uniform срез из FrameRingBuffer)
                    wgpu::BindGroupEntry { 
                        binding: 0, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: frame_ring_buffer, offset: ranges.camera.offset, size: wgpu::BufferSize::new(ranges.camera.size)
                        })
                    },
                    // Статические задачи куллинга (Storage срез из SceneDataBuffer)
                    wgpu::BindGroupEntry { 
                        binding: 1, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: culling_tasks_range.offset, size: wgpu::BufferSize::new(culling_tasks_range.size)
                        })
                    },                
                    // Данные всех инстансов сцены (Storage срез из SceneDataBuffer)
                    wgpu::BindGroupEntry { 
                        binding: 2, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: instances_range.offset, size: wgpu::BufferSize::new(instances_range.size)
                        })
                    },
                    // Инфо о мешах (Storage срез из SceneDataBuffer)
                    wgpu::BindGroupEntry { 
                        binding: 3, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: mesh_infos_range.offset, size: wgpu::BufferSize::new(mesh_infos_range.size)
                        })
                    },                
                    // Описания всех мешлетов (Storage срез из SceneDataBuffer)
                    wgpu::BindGroupEntry { 
                        binding: 4, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: meshlets_range.offset, size: wgpu::BufferSize::new(meshlets_range.size)
                        })
                    },
                    // ВЫХОД: Индексы видимых мешлетов текущего кадра (Storage срез из FrameRingBuffer)
                    wgpu::BindGroupEntry { 
                        binding: 5, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: frame_ring_buffer, offset: ranges.visible_instances.offset, size: wgpu::BufferSize::new(ranges.visible_instances.size)
                        })
                    },                
                    // ВЫХОД: Буфер indirect команд отрисовки (Storage срез из FrameRingBuffer)
                    wgpu::BindGroupEntry { 
                        binding: 6, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: frame_ring_buffer, offset: ranges.indirect.offset, size: wgpu::BufferSize::new(ranges.indirect.size)
                        })
                    },
                    // ВЫХОД: Атомарный счетчик команд (Storage срез из FrameRingBuffer)
                    wgpu::BindGroupEntry { 
                        binding: 7, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: frame_ring_buffer, offset: ranges.counter.offset, size: wgpu::BufferSize::new(ranges.counter.size)
                        })
                    },
                ],
            });

            // 2. Сборка монолитной Bind Group для графического Render-пайплайна текущего кадра
            let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Render Bind Group Frame {}", i)),
                layout: render_bind_group_layout,
                entries: &[
                    // Камера текущего кадра (Uniform срез из FrameRingBuffer)
                    wgpu::BindGroupEntry { 
                        binding: 0, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: frame_ring_buffer, offset: ranges.camera.offset, size: wgpu::BufferSize::new(ranges.camera.size)
                        })
                    },
                    // Иерархия нод / трансформаций (Storage срез из SceneDataBuffer)
                    wgpu::BindGroupEntry { 
                        binding: 1, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: nodes_range.offset, size: wgpu::BufferSize::new(nodes_range.size)
                        })
                    },
                    // Данные инстансов (Storage срез из SceneDataBuffer)
                    wgpu::BindGroupEntry { 
                        binding: 2, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: instances_range.offset, size: wgpu::BufferSize::new(instances_range.size)
                        })
                    },
                    // Инфо о мешах (Storage срез из SceneDataBuffer)
                    wgpu::BindGroupEntry { 
                        binding: 3, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: mesh_infos_range.offset, size: wgpu::BufferSize::new(mesh_infos_range.size)
                        })
                    },
                    // ВХОД ИЗ COMPUTE: Список отсеянных мешлетов (Storage срез из FrameRingBuffer)
                    wgpu::BindGroupEntry { 
                        binding: 4, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: frame_ring_buffer, offset: ranges.visible_instances.offset, size: wgpu::BufferSize::new(ranges.visible_instances.size)
                        })
                    },                
                    // Все мешлеты сцены (Storage срез из SceneDataBuffer)
                    wgpu::BindGroupEntry { 
                        binding: 5, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: meshlets_range.offset, size: wgpu::BufferSize::new(meshlets_range.size)
                        })
                    },
                    // Локальные индексы треугольников внутри мешлетов (Storage срез из SceneDataBuffer)
                    wgpu::BindGroupEntry { 
                        binding: 6, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: meshlet_local_indices_range.offset, size: wgpu::BufferSize::new(meshlet_local_indices_range.size)
                        })
                    },
                    // Таблица перенаправления вершин (Storage срез из SceneDataBuffer)
                    wgpu::BindGroupEntry { 
                        binding: 7, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: meshlet_vertex_redirect_range.offset, size: wgpu::BufferSize::new(meshlet_vertex_redirect_range.size)
                        })
                    },
                ],
            });

            frame_resources.push(FrameResources {
                culling_compute_bind_group,
                render_bind_group,
            });
        }

        frame_resources
    }
}
