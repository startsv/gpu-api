use std::borrow::Cow;
use gpu_api_dto::TextureType;
use gpu_api_relay::model_bindless_data::{CameraUniform, CullingTask, DrawIndexedIndirectCommand, MaterialFactors, NodeData, StaticVertex, VisibleInstanceData};
use log::info;
use wgpu::{TextureFormat, util::{DeviceExt, StagingBelt}};
use crate::{camera::CAMERA_UNIFORM_SIZE, pipeline::{model_pipeline::model::InitData, static_bindless_pipeline::static_bindless_frame_bg::{BufferRange, FrameResourceRanges, FrameBindgroups}}};

pub mod static_bindless_layout;
pub mod static_bindless_pipelines;
pub mod static_bindless_new;
pub mod static_bindless_frame_bg;
pub mod static_bindless_frame;

pub const MAX_VERTICES: u64 = 1_000_000;
pub const MAX_INDICES: u64 = 3_000_000;
pub const MAX_INSTANCES: u64 = 10_000;
pub const MAX_MESH_INFOS: u64 = 10_000;
pub const MAX_MESHLETS: u64 = 100_000;
pub const MAX_MATERIALS: u64 = 1_000;
pub const MAX_NODES: u64 = 1_000;
pub const MAX_TEXTURES: u32 = 256;
pub const NUM_FRAMES_IN_FLIGHT: usize = 2;
pub const ALIGN: u64 = 256; // wgpu::BIND_BUFFER_ALIGNMENT

/*
• Если у вас в игре есть 10 уникальных 3D-моделей (домов, персонажей, деревьев), и каждая модель разбита на 100 мешлетов, то вам нужен буфер, способный вместить все \(10 \times 100 = 1000\) мешлетов.
• Следовательно, MAX_MESHLETS = 1000.
• Этот лимит определяет физический размер неизменяемых (Read-Only) Storage-массивов (global_meshlets, meshlet_local_indices, meshlet_vertex_redirect), куда Rust один раз копирует геометрию при старте уровня.

2. self.max_meshlets_count — Максимум отрисовки за один кадр

Она используется для выделения памяти в FrameRingBuffer под результаты работы Compute-шейдера куллинга.
• Даже если на уровне загружено 10 000 мешлетов (MAX_MESHLETS), игрок в каждый конкретный момент времени смотрит только в одну сторону. Камера физически не может увидеть больше мешлетов, чем помещается на экране.
• self.max_meshlets_count определяет размер буфера indirect_buffer и visible_instances_buffer для текущего кадра в полете.
• Шейдер куллинга берет все мешлеты сцены, отсекает невидимые, и записывает в indirect_buffer только прошедшие тест. Этот буфер не обязан быть размером со всю сцену — он должен быть равен максимальному числу мешлетов, которое теоретически может быть отрисовано одновременно на экране.
*/

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshInfo {
    pub start_meshlet_index: u32,
    pub meshlet_count: u32,
    pub vertex_buffer_offset: u32,
    pub base_vertex: i32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StaticMeshletDescription {
    pub aabb_min: [f32; 3],
    pub vertex_offset: u32,
    pub aabb_max: [f32; 3],
    pub index_offset: u32,
    pub index_count: u32,
    pub material_index: u32,
    pub pad0: u32,
    pub pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    pub model_matrix: [[f32; 4]; 4],
    pub is_animated: u32,       // Всегда 0u для статики, но сохраняем структуру
    pub node_index: u32,
    pub joints_offset: u32,     // Не используется
    pub material_index: u32,
    pub primitive_index: u32,   // Базовый ID indirect-команды первого мешлета этого инстанса
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
    pub aabb_min: [f32; 3],
    pub pad_aabb1: u32,
    pub aabb_max: [f32; 3],
    pub pad_aabb2: u32,
}

pub struct StaticBindlessResources {
    // 1. ТРИ МОНОЛИТНЫХ БУФЕРА
    pub geometry_buffer: wgpu::Buffer,   // Вершины + Сквозной индексный шаблон
    pub scene_data_buffer: wgpu::Buffer, // Статические структуры данных сцены
    pub frame_ring_buffer: wgpu::Buffer, // Кольцевой буфер динамических данных (Anti-Flicker)
    pub camera_ring_buffer: wgpu::Buffer,

    pub materials_buffer: wgpu::Buffer,

    // 2. ДИАПАЗОНЫ СТАТИЧЕСКИХ ДАННЫХ (Для queue.write_buffer и set_index_buffer)
    pub vertex_range: BufferRange,
    pub index_range: BufferRange,
    pub mesh_infos_range: BufferRange,
    pub meshlets_range: BufferRange,
    pub meshlet_local_indices_range: BufferRange,
    pub meshlet_vertex_redirect_range: BufferRange,
    pub materials_range: BufferRange,
    pub instances_range: BufferRange,
    pub nodes_range: BufferRange,
    pub culling_tasks_range: BufferRange,

    // 3. ДИАПАЗОНЫ КАДРОВЫХ ДАННЫХ
    pub frame_ranges: Vec<FrameResourceRanges>,
    pub frame_resources: Vec<FrameBindgroups>,
    pub frame_index: usize,
    pub max_meshlets_count: u32,

    // 4. ПАЙПЛАЙНЫ
    pub culling_compute_pipeline: wgpu::ComputePipeline,
    pub render_pipeline: wgpu::RenderPipeline,
    pub materials_bind_group: wgpu::BindGroup,
}
