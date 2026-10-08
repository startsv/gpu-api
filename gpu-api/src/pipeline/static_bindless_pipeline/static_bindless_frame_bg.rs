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

pub struct FrameBindgroups {
    // Вся кадровые данные теперь находятся внутри единого монолитного frame_ring_buffer,
    // но для удобства мы сохраняем готовые BindGroup, нарезающие этот буфер на срезы.
    pub culling_compute_bind_group: wgpu::BindGroup,
    pub render_bind_group: wgpu::BindGroup,
}

impl FrameBindgroups {
    pub fn create(
        device: &wgpu::Device,
        // Layout'ы пайплайнов для сборки BindGroup
        culling_compute_layout: &wgpu::BindGroupLayout,
        render_bind_group_layout: &wgpu::BindGroupLayout,
        geometry_buffer: &wgpu::Buffer, // <--- Добавлено!
        // Ссылки на 2 основных монолитных буфера, откуда нарезаются срезы
        scene_data_buffer: &wgpu::Buffer,
        frame_ring_buffer: &wgpu::Buffer,
        camera_ring_buffer: &wgpu::Buffer,
        vertex_range: &BufferRange,     // <--- Добавлено!
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

            // =====================================================================
            // 1. Сборка монолитной Bind Group для Compute-пасса куллинга текущего кадра
            // Всего 8 записей: binding(0) .. binding(7) -> Строго соответствует culling.wgsl
            // =====================================================================
            let culling_compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Culling Compute Bind Group Frame {}", i)),
                layout: culling_compute_layout,
                entries: &[                
                    wgpu::BindGroupEntry { 
                        binding: 0, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &camera_ring_buffer, // <--- ИСПРАВЛЕНО
                            offset: ranges.camera.offset, 
                            size: wgpu::BufferSize::new(std::mem::size_of::<CameraUniform>() as u64)
                        })
                    },
                    wgpu::BindGroupEntry { 
                        binding: 1, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: culling_tasks_range.offset, size: wgpu::BufferSize::new(culling_tasks_range.size)
                        })
                    },                
                    wgpu::BindGroupEntry { 
                        binding: 2, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: instances_range.offset, size: wgpu::BufferSize::new(instances_range.size)
                        })
                    },
                    wgpu::BindGroupEntry { 
                        binding: 3, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: mesh_infos_range.offset, size: wgpu::BufferSize::new(mesh_infos_range.size)
                        })
                    },                
                    wgpu::BindGroupEntry { 
                        binding: 4, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: meshlets_range.offset, size: wgpu::BufferSize::new(meshlets_range.size)
                        })
                    },
                    wgpu::BindGroupEntry { 
                        binding: 5, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: frame_ring_buffer, offset: ranges.visible_instances.offset, size: wgpu::BufferSize::new(ranges.visible_instances.size)
                        })
                    },                
                    wgpu::BindGroupEntry { 
                        binding: 6, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: frame_ring_buffer, offset: ranges.indirect.offset, size: wgpu::BufferSize::new(ranges.indirect.size)
                        })
                    },
                    wgpu::BindGroupEntry { 
                        binding: 7, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: frame_ring_buffer, offset: ranges.counter.offset, size: wgpu::BufferSize::new(ranges.counter.size)
                        })
                    },
                ],
            });

            // =====================================================================
            // 2. Сборка монолитной Bind Group для графического рендеринга текущего кадра
            // Всего 9 записей: binding(0) .. binding(8) -> Строго соответствует render.wgsl
            // =====================================================================
            let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Render Bind Group Frame {}", i)),
                layout: render_bind_group_layout,
                entries: &[
                    // binding(0): camera
                    wgpu::BindGroupEntry { 
                        binding: 0, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: camera_ring_buffer, offset: ranges.camera.offset, size: wgpu::BufferSize::new(ranges.camera.size)
                        })
                    },
                    // binding(1): static_vertices
                    wgpu::BindGroupEntry { 
                        binding: 1, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: geometry_buffer, offset: vertex_range.offset, size: wgpu::BufferSize::new(vertex_range.size)
                        })
                    },
                    // binding(2): global_nodes
                    wgpu::BindGroupEntry { 
                        binding: 2, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: nodes_range.offset, size: wgpu::BufferSize::new(nodes_range.size)
                        })
                    },
                    // binding(3): global_instances
                    wgpu::BindGroupEntry { 
                        binding: 3, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: instances_range.offset, size: wgpu::BufferSize::new(instances_range.size)
                        })
                    },
                    // binding(4): global_mesh_infos
                    wgpu::BindGroupEntry { 
                        binding: 4, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: mesh_infos_range.offset, size: wgpu::BufferSize::new(mesh_infos_range.size)
                        })
                    },
                    // binding(5): visible_instances (Вход из Compute)
                    wgpu::BindGroupEntry { 
                        binding: 5, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: frame_ring_buffer, offset: ranges.visible_instances.offset, size: wgpu::BufferSize::new(ranges.visible_instances.size)
                        })
                    },                
                    // binding(6): global_meshlets
                    wgpu::BindGroupEntry { 
                        binding: 6, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: meshlets_range.offset, size: wgpu::BufferSize::new(meshlets_range.size)
                        })
                    },
                    // binding(7): meshlet_local_indices
                    wgpu::BindGroupEntry { 
                        binding: 7, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: meshlet_local_indices_range.offset, size: wgpu::BufferSize::new(meshlet_local_indices_range.size)
                        })
                    },
                    // КРИТИЧЕСКИЙ ФИКС: binding(8) теперь на месте! 
                    // meshlet_vertex_redirect замыкает цепочку распаковки
                    wgpu::BindGroupEntry { 
                        binding: 8, 
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: scene_data_buffer, offset: meshlet_vertex_redirect_range.offset, size: wgpu::BufferSize::new(meshlet_vertex_redirect_range.size)
                        })
                    },
                ],
            });

            frame_resources.push(FrameBindgroups {
                culling_compute_bind_group,
                render_bind_group,
            });
        }

        frame_resources
    }
}
