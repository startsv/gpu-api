use glam::{Mat4, Vec3};
use gpu_api::pipeline::{aa_line_pipeline::AaLineInstance, static_bindless_pipeline::{InstanceData, MeshInfo, StaticMeshletDescription}};
use gpu_api_relay::model_bindless_data::{CullingTask, DrawIndexedIndirectCommand, StaticVertex}; // Используем вашу математическую библиотеку (скорее всего glam)

pub struct TestSceneData {
    pub vertices: Vec<StaticVertex>,
    pub indices: Vec<u32>,
    pub meshlets: Vec<StaticMeshletDescription>,
    pub mesh_infos: Vec<MeshInfo>,
    pub meshlet_vertex_redirect: Vec<u32>,
    pub meshlet_local_indices: Vec<u32>,
    pub instances: Vec<InstanceData>,
    pub indirect_commands: Vec<DrawIndexedIndirectCommand>,
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

    // 2. Строим таблицы перенаправления и локальных индексов для мешлетов куба
    // Мешлет 1 (содержит вершины 0, 1, 2, 3, 5, 6 для первых 3 граней)
    let meshlet1_redirect = vec![0, 1, 2, 3, 5, 6]; 
    // Локальные индексы треугольников, ссылающиеся строго на индексы внутри meshlet1_redirect (0..5)
    let meshlet1_local_indices = vec![
        0, 1, 2,  2, 3, 0, // Передняя грань (вершины 0,1,2,3)
        1, 4, 5,  5, 2, 1, // Правая грань (вершины 1,5,6,2 -> локальные 1,4,5,2)
        3, 2, 5,  5, 2, 3, // Пример верхней грани куба (усеченно для тестов)
    ];

    // Мешлет 2 (оставшаяся геометрия куба)
    let meshlet2_redirect = vec![4, 5, 6, 7, 0, 3];
    let meshlet2_local_indices = vec![
        3, 2, 1,  1, 0, 3, // Задняя грань
        0, 4, 5,  5, 1, 0, // Левая грань
        0, 1, 4,  4, 5, 0, // Нижня грань
    ];

    // Объединяем локальные данные мешлетов в глобальные массивы для GPU
    let mut meshlet_vertex_redirect = Vec::new();
    let mut meshlet_local_indices = Vec::new();

    // Запоминаем стартовые смещения в мега-буферах мешлетов
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

    // Описываем наш единственный базовый меш (Куб)
    let mesh_infos = vec![MeshInfo {
        start_meshlet_index: 0,
        meshlet_count: 2, 
        vertex_buffer_offset: 0,
        base_vertex: 0,
    }];

    // 3. Генерируем инстансы кубов и их персональные indirect-команды    
    let dummy_indices_template: Vec<u32> = (0..18).collect();

    let mut instances = Vec::new();
    let mut indirect_commands = Vec::new();

    for i in 0..num_instances {
        let position = Vec3::new((i as f32) * 2.5, 0.0, -5.0);
        let model_matrix = Mat4::from_translation(position);

        let base_command_id = (i * 2) as u32;

        // Команда для Мешлета 1 этого куба
        indirect_commands.push(DrawIndexedIndirectCommand {
            index_count: 18,
            instance_count: 0, 
            first_index: 0, // СТРОГО 0
            base_vertex: (base_command_id << 16) as i32, // Запекаем ID в старшие 16 бит
            first_instance: 0, 
        });

        // Команда для Мешлета 2 этого куба
        indirect_commands.push(DrawIndexedIndirectCommand {
            index_count: 18,
            instance_count: 0, 
            first_index: 0, // СТРОГО 0
            base_vertex: ((base_command_id + 1) << 16) as i32, // Запекаем ID в старшие 16 бит

            first_instance: 0, 
        });

        instances.push(InstanceData {
            model_matrix: model_matrix.to_cols_array_2d(),
            is_animated: 0,
            node_index: 0,
            joints_offset: 0,
            material_index: 0,
            primitive_index: 0, 
            pad0: base_command_id, 
            pad1: 0, pad2: 0,
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
        indices: dummy_indices_template, // Возвращаем шаблон вместо старых индексов
        meshlets,
        mesh_infos,
        meshlet_local_indices,           // Не забудьте добавить эти поля в вашу TestSceneData
        meshlet_vertex_redirect,          // Не забудьте добавить эти поля в вашу TestSceneData
        instances,
        indirect_commands,
        culling_tasks,
    }
}

