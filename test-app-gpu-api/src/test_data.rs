use glam::{Mat4, Vec3};
use gpu_api::pipeline::{aa_line_pipeline::AaLineInstance, static_bindless_pipeline::{InstanceData, MeshInfo, StaticMeshletDescription}};
use gpu_api_relay::model_bindless_data::{CullingTask, DrawIndexedIndirectCommand, StaticVertex, SurfaceData, SurfaceMeshletDescription, SurfaceVertex}; // Используем вашу математическую библиотеку (скорее всего glam)

pub struct TestSceneData {
    pub vertices: Vec<StaticVertex>,
    pub indices: Vec<u32>,
    pub meshlets: Vec<StaticMeshletDescription>,
    pub mesh_infos: Vec<MeshInfo>,
    pub meshlet_vertex_redirect: Vec<u32>,
    pub meshlet_local_indices: Vec<u32>,
    pub instances: Vec<InstanceData>,    
    pub culling_tasks: Vec<CullingTask>,
}

pub fn generate_grid(
    lines: &mut Vec<AaLineInstance>, 
    slices: i32, 
    step: f32, 
    width: f32
) {
    let half_size = slices as f32 * step;
        
    let default_color = [0.3, 0.3, 0.3, 1.0];
    let axis_x_color  = [0.8, 0.2, 0.2, 1.0];
    let axis_z_color  = [0.2, 0.2, 0.8, 1.0];

    for i in -slices..=slices {
        let current_coord = i as f32 * step;
        
        let color_z = if i == 0 { 
            axis_z_color
        } else { 
            default_color 
        };
        
        lines.push(AaLineInstance {
            color: color_z,
            width,
            start_pos: [current_coord, 0.0, -half_size],
            end_pos:   [current_coord, 0.0, half_size],
        });
        
        let color_x = if i == 0 { 
            axis_x_color
        } else { 
            default_color 
        };

        lines.push(AaLineInstance {
            color: color_x,
            width,
            start_pos: [-half_size, 0.0, current_coord],
            end_pos:   [half_size, 0.0, current_coord],
        });
    }
}

pub fn generate_static_test_data(num_instances: u32) -> TestSceneData {
    // 1. Геометрия стандартного куба (8 уникальных вершин)
    let vertices = vec![
        StaticVertex { position: [-0.5, -0.5,  0.5], uv: [0.0, 0.0], normal: [0.0, 0.0, 1.0], tangent: [1.0, 0.0, 0.0], bitangent: [0.0, 1.0, 0.0], ..Default::default() }, // 0
        StaticVertex { position: [ 0.5, -0.5,  0.5], uv: [1.0, 0.0], normal: [0.0, 0.0, 1.0], tangent: [1.0, 0.0, 0.0], bitangent: [0.0, 1.0, 0.0], ..Default::default() }, // 1
        StaticVertex { position: [ 0.5,  0.5,  0.5], uv: [1.0, 1.0], normal: [0.0, 0.0, 1.0], tangent: [1.0, 0.0, 0.0], bitangent: [0.0, 1.0, 0.0], ..Default::default() }, // 2
        StaticVertex { position: [-0.5,  0.5,  0.5], uv: [0.0, 1.0], normal: [0.0, 0.0, 1.0], tangent: [1.0, 0.0, 0.0], bitangent: [0.0, 1.0, 0.0], ..Default::default() }, // 3
        StaticVertex { position: [-0.5, -0.5, -0.5], uv: [1.0, 0.0], normal: [0.0, 0.0, -1.0], tangent: [-1.0, 0.0, 0.0], bitangent: [0.0, 1.0, 0.0], ..Default::default() }, // 4
        StaticVertex { position: [ 0.5, -0.5, -0.5], uv: [0.0, 0.0], normal: [0.0, 0.0, -1.0], tangent: [-1.0, 0.0, 0.0], bitangent: [0.0, 1.0, 0.0], ..Default::default() }, // 5
        StaticVertex { position: [ 0.5,  0.5, -0.5], uv: [0.0, 1.0], normal: [0.0, 0.0, -1.0], tangent: [-1.0, 0.0, 0.0], bitangent: [0.0, 1.0, 0.0], ..Default::default() }, // 6
        StaticVertex { position: [-0.5,  0.5, -0.5], uv: [1.0, 1.0], normal: [0.0, 0.0, -1.0], tangent: [-1.0, 0.0, 0.0], bitangent: [0.0, 1.0, 0.0], ..Default::default() }, // 7
    ];

    // =========================================================================
    // МЕШЛЕТ 1: Передняя, Правая и Нижняя грани (6 треугольников = 18 индексов)
    // Содержит уникальные вершины: 0, 1, 2, 3, 4, 5, 6
    let meshlet1_redirect = vec![0, 1, 2, 3, 4, 5, 6]; 
    
    // Маппинг глобальных вершин на индексы в redirect массиве:
    let meshlet1_local_indices = vec![
        0, 1, 2,  2, 3, 0, // Передняя грань (глобальные 0, 1, 2, 2, 3, 0)
        1, 5, 6,  6, 2, 1, // Правая грань   (глобальные 1, 5, 6, 6, 2, 1)
        4, 5, 1,  1, 0, 4, // Нижняя грань   (глобальные 4, 5, 1, 1, 0, 4)
    ];

    // =========================================================================
    // МЕШЛЕТ 2: Задняя, Левая и Верхняя грани (6 треугольников = 18 индексов)
    // Содержит уникальные вершины: 0, 1, 2, 3, 4, 5, 6, 7
    let meshlet2_redirect = vec![0, 1, 2, 3, 4, 5, 6, 7];
    
    // Локальные индексы строятся СТРОГО по позициям в meshlet2_redirect:
    let meshlet2_local_indices = vec![
        6, 5, 4,  4, 7, 6, // Задняя грань  (глобальные 6, 5, 4, 4, 7, 6)
        3, 0, 4,  4, 7, 3, // Левая грань   (глобальные 3, 0, 4, 4, 7, 3)
        2, 1, 5,  5, 6, 2, // Верхняя грань  (глобальные 2, 1, 5, 5, 6, 2)
    ];

    // Объединяем локальные данные мешлетов в глобальные массивы для GPU
    let mut meshlet_vertex_redirect = Vec::new();
    let mut meshlet_local_indices = Vec::new();

    let m1_vertex_offset = meshlet_vertex_redirect.len() as u32;
    meshlet_vertex_redirect.extend(&meshlet1_redirect);
    let m1_index_offset = meshlet_local_indices.len() as u32;
    meshlet_local_indices.extend(&meshlet1_local_indices);

    let m2_vertex_offset = meshlet_vertex_redirect.len() as u32;
    meshlet_vertex_redirect.extend(&meshlet2_redirect);
    let m2_index_offset = meshlet_local_indices.len() as u32;
    meshlet_local_indices.extend(&meshlet2_local_indices);

    // Описание мешлетов
    let meshlets = vec![
        StaticMeshletDescription {
            aabb_min: [-0.5, -0.5, -0.5],
            vertex_offset: m1_vertex_offset,
            aabb_max: [0.5, 0.5, 0.5],
            index_offset: m1_index_offset,
            index_count: 18,
            material_index: 0,
            pad0: 0, pad1: 0,
        },
        StaticMeshletDescription {
            aabb_min: [-0.5, -0.5, -0.5],
            vertex_offset: m2_vertex_offset,
            aabb_max: [0.5, 0.5, 0.5],
            index_offset: m2_index_offset,
            index_count: 18,
            material_index: 0,
            pad0: 0, pad1: 0,
        },
    ];

    // Описываем базовый меш (Куб)
    let mesh_infos = vec![MeshInfo {
        start_meshlet_index: 0,
        meshlet_count: 2, 
        vertex_buffer_offset: 0,
        base_vertex: 0,
    }];

    // КРИТИЧЕСКИЙ ОПТИМИЗАЦИОННЫЙ ФИКС:
    // Поскольку `index_count` в непрямой команде для мешлета куба равен 18,
    // видеокарта аппаратно считывает индексы от 0 до 17 для каждого мешлета.
    // Шаблона на 18 элементов [0, 1, 2, ..., 17] абсолютно достаточно для всей сцены!
    let dummy_indices_template: Vec<u32> = (0..18).collect();

    let mut instances = Vec::new();
    for i in 0..num_instances {
        // Расставляем кубы в ряд по оси X
        let position = glam::Vec3::new((i as f32) * 2.5, 0.0, -5.0);
        let model_matrix = glam::Mat4::from_translation(position);

        instances.push(InstanceData {
            model_matrix: model_matrix.to_cols_array_2d(),
            is_animated: 0,
            node_index: 0,
            joints_offset: 0,
            material_index: 0,
            primitive_index: 0, // Указывает на куб (mesh_infos[0])
            pad0: 0, 
            pad1: 0, 
            pad2: 0,
            aabb_min: [-0.5, -0.5, -0.5],
            pad_aabb1: 0,
            aabb_max: [0.5, 0.5, 0.5],
            pad_aabb2: 0,
        });
    }

    let culling_tasks = vec![CullingTask {
        start_object_index: 0,
        object_count: num_instances,
        lod_level: 0,
        _padding: 0,
    }];

    TestSceneData {
        vertices,
        indices: dummy_indices_template, 
        meshlets,
        mesh_infos,
        meshlet_local_indices,           
        meshlet_vertex_redirect,          
        instances,
        culling_tasks,
    }
}

const SURFACE_MESHLET_SIZE: u32 = 8;
const SURFACE_VERTICES_PER_MESHLET: u32 = SURFACE_MESHLET_SIZE * SURFACE_MESHLET_SIZE;

pub fn generate_test_surface(
    width_in_meshlets: u32,
    depth_in_meshlets: u32,
    vertex_spacing: f32,
) -> SurfaceData {
    let mut surface_data = SurfaceData::new();    
    
    let total_meshlets = width_in_meshlets * depth_in_meshlets;
        
    surface_data.vertices.reserve((total_meshlets * SURFACE_VERTICES_PER_MESHLET) as usize);
    
    let indices_per_meshlet = (SURFACE_MESHLET_SIZE - 1) * (SURFACE_MESHLET_SIZE - 1) * 6;
    surface_data.indices.reserve((total_meshlets * indices_per_meshlet) as usize);
    surface_data.meshlets.reserve(total_meshlets as usize);
    
    let meshlet_world_size = (SURFACE_MESHLET_SIZE - 1) as f32 * vertex_spacing;

    let mut current_vertex_offset = 0;
    let mut current_index_offset = 0;

    for mz in 0..depth_in_meshlets {
        for mx in 0..width_in_meshlets {
            let mut aabb_min = [f32::MAX, f32::MAX, f32::MAX];
            let mut aabb_max = [f32::MIN, f32::MIN, f32::MIN];

            let meshlet_x_origin = mx as f32 * meshlet_world_size;
            let meshlet_z_origin = mz as f32 * meshlet_world_size;
            
            for lz in 0..SURFACE_MESHLET_SIZE {
                for lx in 0..SURFACE_MESHLET_SIZE {
                    let world_x = meshlet_x_origin + (lx as f32 * vertex_spacing);
                    let world_z = meshlet_z_origin + (lz as f32 * vertex_spacing);
                    
                    let world_y = (world_x * 0.1).sin() * 5.0 + (world_z * 0.05).cos() * 8.0;
                    
                    let nx = -0.1 * (world_x * 0.1).cos() * 5.0;
                    let nz = -0.05 * -(world_z * 0.05).sin() * 8.0;
                    let mut normal = [nx, 1.0, nz];
                    
                    let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
                    normal[0] /= len;
                    normal[1] /= len;
                    normal[2] /= len;

                    let vertex = SurfaceVertex {
                        position: [world_x, world_y, world_z],
                        _pad0: 0.0,
                        normal,
                        _pad1: 0.0,
                    };

                    surface_data.vertices.push(vertex);
                    
                    aabb_min[0] = aabb_min[0].min(world_x);
                    aabb_min[1] = aabb_min[1].min(world_y);
                    aabb_min[2] = aabb_min[2].min(world_z);

                    aabb_max[0] = aabb_max[0].max(world_x);
                    aabb_max[1] = aabb_max[1].max(world_y);
                    aabb_max[2] = aabb_max[2].max(world_z);
                }
            }
            
            for lz in 0..(SURFACE_MESHLET_SIZE - 1) {
                for lx in 0..(SURFACE_MESHLET_SIZE - 1) {
                    
                    let i0 = lz * SURFACE_MESHLET_SIZE + lx;
                    let i1 = lz * SURFACE_MESHLET_SIZE + (lx + 1);
                    let i2 = (lz + 1) * SURFACE_MESHLET_SIZE + lx;
                    let i3 = (lz + 1) * SURFACE_MESHLET_SIZE + (lx + 1);
                    
                    surface_data.indices.push(current_vertex_offset + i0);
                    surface_data.indices.push(current_vertex_offset + i2);
                    surface_data.indices.push(current_vertex_offset + i1);

                    surface_data.indices.push(current_vertex_offset + i1);
                    surface_data.indices.push(current_vertex_offset + i2);
                    surface_data.indices.push(current_vertex_offset + i3);
                }
            }
            
            let meshlet_index_count = (SURFACE_MESHLET_SIZE - 1) * (SURFACE_MESHLET_SIZE - 1) * 6;
            
            surface_data.meshlets.push(SurfaceMeshletDescription {
                aabb_min,
                vertex_offset: current_vertex_offset,
                aabb_max,
                index_offset: current_index_offset,
                index_count: meshlet_index_count,
                material_index: 0,
                pad0: 0,
                pad1: 0,
            });

            current_vertex_offset += SURFACE_VERTICES_PER_MESHLET;
            current_index_offset += meshlet_index_count;
        }
    }
    
    surface_data.indirect_commands.push(DrawIndexedIndirectCommand {
        index_count: 0,
        instance_count: 0,
        first_index: 0,
        base_vertex: 0,
        first_instance: 0,
    });

    surface_data
}
