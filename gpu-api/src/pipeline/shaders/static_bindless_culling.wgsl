struct CameraUniform {
    camera_position: vec3<f32>,
    padding: u32,    
    view_proj: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
};

struct InstanceData {
    model_matrix: mat4x4<f32>,
    is_animated: u32,
    node_index: u32,
    joints_offset: u32,
    material_index: u32,
    primitive_index: u32,  // Используется как mesh_info_index
    base_command_id: u32,  // Смещение первой indirect-команды для данного типа меша
    pad1: u32,
    pad2: u32,
    aabb_min: vec3<f32>,
    pad_aabb1: u32,
    aabb_max: vec3<f32>,
    pad_aabb2: u32,
};

struct MeshInfo {
    start_meshlet_index: u32,
    meshlet_count: u32,
    vertex_buffer_offset: u32,
    base_vertex: i32,
};

struct StaticMeshletDescription {
    aabb_min: vec3<f32>,
    vertex_offset: u32,
    aabb_max: vec3<f32>,
    index_offset: u32,
    index_count: u32,
    material_index: u32,
    pad0: u32,
    pad1: u32,
};

struct CullingTask {
    start_object_index: u32,
    object_count: u32,
    lod_level: u32,
    _padding: u32,
};

struct DrawIndexedIndirectCommand {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

struct VisibleInstanceData {
    instance_id: u32,
    material_index: u32,
    meshlet_index: u32, // Добавлено поле, чтобы Vertex Shader знал, какой именно мешлет рисовать!
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var<storage, read> culling_tasks: array<CullingTask>;
@group(1) @binding(1) var<storage, read> global_instances: array<InstanceData>;
@group(1) @binding(2) var<storage, read> global_mesh_infos: array<MeshInfo>;
@group(1) @binding(3) var<storage, read> global_meshlets: array<StaticMeshletDescription>;

@group(1) @binding(4) var<storage, read_write> visible_instances: array<VisibleInstanceData>;
@group(1) @binding(5) var<storage, read_write> indirect_commands: array<DrawIndexedIndirectCommand>;

// Вспомогательная функция для трансформации локального AABB в Мировой AABB с учетом матрицы объекта
fn transform_aabb(aabb_min: vec3<f32>, aabb_max: vec3<f32>, m: mat4x4<f32>) -> array<vec3<f32>, 2> {
    let center = (aabb_min + aabb_max) * 0.5;
    let extents = (aabb_max - aabb_min) * 0.5;
    
    let world_center = (m * vec4<f32>(center, 1.0)).xyz;
    
    // Берем абсолютные значения осей матрицы трансформации для корректного выравнивания мирового AABB
    let row0 = vec3<f32>(abs(m[0].x), abs(m[1].x), abs(m[2].x));
    let row1 = vec3<f32>(abs(m[0].y), abs(m[1].y), abs(m[2].y));
    let row2 = vec3<f32>(abs(m[0].z), abs(m[1].z), abs(m[2].z));
    
    let world_extents = vec3<f32>(
        dot(row0, extents),
        dot(row1, extents),
        dot(row2, extents)
    );
    
    var result: array<vec3<f32>, 2>;
    result[0] = world_center - world_extents; // world_min
    result[1] = world_center + world_extents; // world_max
    return result;
}

// Функция проверки видимости AABB против 6 плоскостей фрустума камеры
fn is_aabb_visible(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = camera.frustum_planes[i];
        var p = aabb_min;
        if (plane.x >= 0.0) { p.x = aabb_max.x; }
        if (plane.y >= 0.0) { p.y = aabb_max.y; }
        if (plane.z >= 0.0) { p.z = aabb_max.z; }
                
        if (dot(plane.xyz, p) + plane.w < 0.0) {
            return false; // Полностью за пределами одной из плоскостей
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn culling_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let task = culling_tasks[0u]; 
    let global_instance_id = task.start_object_index + global_id.x;
    
    if (global_id.x >= task.object_count) { return; }
    
    let instance = global_instances[global_instance_id];
    let mesh_info = global_mesh_infos[instance.primitive_index];
    
    // Слот записи в буферы жестко привязан к ID базовой команды объекта
    let base_cmd_id = instance.base_command_id;

    // --- ЭТАП 1: Куллинг на уровне ОБЪЕКТА целиком ---
    let world_object_aabb = transform_aabb(instance.aabb_min, instance.aabb_max, instance.model_matrix);
    let object_visible = is_aabb_visible(world_object_aabb[0], world_object_aabb[1]);
    
    // Если весь объект скрыт за фрустумом, мы обязаны явно занулить instance_count 
    // для всех его мешлетов, чтобы GPU их не рисовал!
    if (!object_visible) {
        for (var m_idx = 0u; m_idx < mesh_info.meshlet_count; m_idx = m_idx + 1u) {
            let cmd_id = base_cmd_id + m_idx;
            indirect_commands[cmd_id].instance_count = 0u;
        }
        return;
    }
            
    // --- ЭТАП 2: Куллинг на уровне отдельных МЕШЛЕТОВ внутри видимого объекта ---
    for (var m_idx = 0u; m_idx < mesh_info.meshlet_count; m_idx = m_idx + 1u) {
        let global_meshlet_id = mesh_info.start_meshlet_index + m_idx;
        let meshlet = global_meshlets[global_meshlet_id];
        
        // Трансформируем AABB конкретного мешлета в мировые координаты с матрицей этого инстанса
        let world_meshlet_aabb = transform_aabb(meshlet.aabb_min, meshlet.aabb_max, instance.model_matrix);
        
        let cmd_id = base_cmd_id + m_idx; 
        let write_index = cmd_id; 
        
        if (is_aabb_visible(world_meshlet_aabb[0], world_meshlet_aabb[1])) {
            // Мешлет видим! Разрешаем отрисовку этой indirect-команды
            indirect_commands[cmd_id].instance_count = 1u;
                                        
            // Записываем метаданные для вершинного шейдера
            visible_instances[write_index].instance_id = global_instance_id;
            visible_instances[write_index].material_index = instance.material_index;
            visible_instances[write_index].meshlet_index = global_meshlet_id;
        } else {
            // Мешлет отсечен фрустумом! Зануляем команду, растеризатор ее пропустит
            indirect_commands[cmd_id].instance_count = 0u;
        }
    }
}
