use glam::{Mat4, Vec3};
use gpu_api::pipeline::{aa_line_pipeline::AaLineInstance, static_bindless_pipeline::{InstanceData, MeshInfo, StaticMeshletDescription}};
use gpu_api_relay::model_bindless_data::{CullingTask, DrawIndexedIndirectCommand, StaticVertex}; // Используем вашу математическую библиотеку (скорее всего glam)

pub struct TestSceneData {
    pub vertices: Vec<StaticVertex>,
    pub indices: Vec<u32>,
    pub meshlets: Vec<StaticMeshletDescription>,
    pub mesh_infos: Vec<MeshInfo>,
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
    // 1. Геометрия стандартного куба (8 вершин) с нормалями и UV
    let vertices = vec![
        // Передняя грань
        StaticVertex { position: [-0.5, -0.5,  0.5], uv: [0.0, 0.0], normal: [0.0, 0.0, 1.0], tangent: [1.0, 0.0, 0.0], bitangent: [0.0, 1.0, 0.0], ..Default::default() },
        StaticVertex { position: [ 0.5, -0.5,  0.5], uv: [1.0, 0.0], normal: [0.0, 0.0, 1.0], tangent: [1.0, 0.0, 0.0], bitangent: [0.0, 1.0, 0.0], ..Default::default() },
        StaticVertex { position: [ 0.5,  0.5,  0.5], uv: [1.0, 1.0], normal: [0.0, 0.0, 1.0], tangent: [1.0, 0.0, 0.0], bitangent: [0.0, 1.0, 0.0], ..Default::default() },
        StaticVertex { position: [-0.5,  0.5,  0.5], uv: [0.0, 1.0], normal: [0.0, 0.0, 1.0], tangent: [1.0, 0.0, 0.0], bitangent: [0.0, 1.0, 0.0], ..Default::default() },
        // Задняя грань
        StaticVertex { position: [-0.5, -0.5, -0.5], uv: [1.0, 0.0], normal: [0.0, 0.0, -1.0], tangent: [-1.0, 0.0, 0.0], bitangent: [0.0, 1.0, 0.0], ..Default::default() },
        StaticVertex { position: [ 0.5, -0.5, -0.5], uv: [0.0, 0.0], normal: [0.0, 0.0, -1.0], tangent: [-1.0, 0.0, 0.0], bitangent: [0.0, 1.0, 0.0], ..Default::default() },
        StaticVertex { position: [ 0.5,  0.5, -0.5], uv: [0.0, 1.0], normal: [0.0, 0.0, -1.0], tangent: [-1.0, 0.0, 0.0], bitangent: [0.0, 1.0, 0.0], ..Default::default() },
        StaticVertex { position: [-0.5,  0.5, -0.5], uv: [1.0, 1.0], normal: [0.0, 0.0, -1.0], tangent: [-1.0, 0.0, 0.0], bitangent: [0.0, 1.0, 0.0], ..Default::default() },
    ];

    // Индексы для всех 6 граней куба (12 треугольников, 36 индексов)
    let indices = vec![
        0, 1, 2,  2, 3, 0, // Передняя
        1, 5, 6,  6, 2, 1, // Правая
        7, 6, 5,  5, 4, 7, // Задняя
        4, 0, 3,  3, 7, 4, // Левая
        4, 5, 1,  1, 0, 4, // Нижняя
        3, 2, 6,  6, 7, 3, // Верхняя
    ];

    // 2. Разбиваем куб на 2 тестовых мешлета (по 18 индексов каждый)
    let meshlets = vec![
        // Мешлет 1: Первые 3 грани. Задаем локальный AABB.
        StaticMeshletDescription {
            aabb_min: [-0.5, -0.5, -0.5],
            vertex_offset: 0,
            aabb_max: [0.5, 0.5, 0.5],
            index_offset: 0,
            index_count: 18,
            material_index: 0,
            pad0: 0, pad1: 0,
        },
        // Мешлет 2: Оставшиеся 3 грани.
        StaticMeshletDescription {
            aabb_min: [-0.5, -0.5, -0.5],
            vertex_offset: 0,
            aabb_max: [0.5, 0.5, 0.5],
            index_offset: 18,
            index_count: 18,
            material_index: 0,
            pad0: 0, pad1: 0,
        },
    ];

    // Описываем наш единственный базовый меш (Куб)
    let mesh_infos = vec![MeshInfo {
        start_meshlet_index: 0,
        meshlet_count: 2, // У нашего куба 2 мешлета
        vertex_buffer_offset: 0,
        base_vertex: 0,
    }];

    // 3. Генерируем инстансы кубов на сцене
    let mut instances = Vec::new();
    let mut indirect_commands = Vec::new();

    // Создаем ВСЕГО 2 команды на всю сцену!
    // Команда 0 рисует первую половину ВСЕХ видимых кубов
    indirect_commands.push(DrawIndexedIndirectCommand {
        index_count: 18,
        instance_count: 0, // Заполнит шейдер
        first_index: 0,
        base_vertex: 0,
        first_instance: 0, // Пишет в visible_instances с 0 по 99 slot
    });

    // Команда 1 рисует вторую половину ВСЕХ видимых кубов
    indirect_commands.push(DrawIndexedIndirectCommand {
        index_count: 18,
        instance_count: 0, // Заполнит шейдер
        first_index: 18,
        base_vertex: 0,
        first_instance: num_instances, // Пишет в visible_instances со 100 по 199 slot
    });

    for i in 0..num_instances {
        let position = Vec3::new((i as f32) * 2.5, 0.0, -5.0);
        let model_matrix = Mat4::from_translation(position);

        instances.push(InstanceData {
            model_matrix: model_matrix.to_cols_array_2d(),
            is_animated: 0,
            node_index: 0,
            joints_offset: 0,
            material_index: 0,
            primitive_index: 0, // Ссылается на mesh_infos[0]
            pad0: 0, // Не используется в этой схеме
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
        indices,
        meshlets,
        mesh_infos,
        instances,
        indirect_commands,
        culling_tasks,
    }
}
