struct CameraUniform {
    camera_position: vec3<f32>,
    padding: u32,    
    view_proj: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
}

struct InstanceData {
    model_matrix: mat4x4<f32>,
    is_animated: u32,
    node_index: u32,
    joints_offset: u32,
    material_index: u32,
    primitive_index: u32,  
    base_command_id: u32,  
    pad1: u32,
    pad2: u32,
    aabb_min: vec3<f32>,
    pad_aabb1: u32,
    aabb_max: vec3<f32>,
    pad_aabb2: u32,
}

struct MeshInfo {
    start_meshlet_index: u32,
    meshlet_count: u32,
    vertex_buffer_offset: u32,
    base_vertex: i32,
}

struct StaticMeshletDescription {
    aabb_min: vec3<f32>,
    vertex_offset: u32,
    aabb_max: vec3<f32>,
    index_offset: u32,
    index_count: u32,
    material_index: u32,
    pad0: u32,
    pad1: u32,
}

struct CullingTask {
    start_object_index: u32,
    object_count: u32,
    lod_level: u32,
    _padding: u32,
}

struct DrawIndexedIndirectCommand {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32, 
}

struct VisibleInstanceData {
    instance_id: u32,
    material_index: u32,
    meshlet_index: u32, 
}

struct IndirectCount {
    count: atomic<u32>,
}

// Временная структура для накопления данных внутри локальной памяти группы
struct LocalMeshletTask {
    instance_id: u32,
    global_meshlet_id: u32,
    index_count: u32,
    material_index: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var<storage, read> culling_tasks: array<CullingTask>;
@group(1) @binding(1) var<storage, read> global_instances: array<InstanceData>;
@group(1) @binding(2) var<storage, read> global_mesh_infos: array<MeshInfo>;
@group(1) @binding(3) var<storage, read> global_meshlets: array<StaticMeshletDescription>;

@group(1) @binding(4) var<storage, read_write> visible_instances: array<VisibleInstanceData>;
@group(1) @binding(5) var<storage, read_write> indirect_commands: array<DrawIndexedIndirectCommand>;
@group(1) @binding(6) var<storage, read_write> command_counter: IndirectCount;

// ОПТИМИЗАЦИЯ: Память рабочей группы (сверхбыстрый кэш на чипе GPU)
var<workgroup> wg_visible_count: atomic<u32>;
var<workgroup> wg_global_offset: u32;
var<workgroup> wg_local_tasks: array<LocalMeshletTask, 256>; // Буфер накопления группы

fn transform_aabb(aabb_min: vec3<f32>, aabb_max: vec3<f32>, m: mat4x4<f32>) -> array<vec3<f32>, 2> {
    let center = (aabb_min + aabb_max) * 0.5;
    let extents = (aabb_max - aabb_min) * 0.5;
    let world_center = (m * vec4<f32>(center, 1.0)).xyz;
    
    let row0 = vec3<f32>(abs(m[0].x), abs(m[1].x), abs(m[2].x));
    let row1 = vec3<f32>(abs(m[0].y), abs(m[1].y), abs(m[2].y));
    let row2 = vec3<f32>(abs(m[0].z), abs(m[1].z), abs(m[2].z));
    
    let world_extents = vec3<f32>(dot(row0, extents), dot(row1, extents), dot(row2, extents));
    
    var result: array<vec3<f32>, 2>;
    result[0] = world_center - world_extents;
    result[1] = world_center + world_extents;
    return result;
}

fn is_aabb_visible(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = camera.frustum_planes[i];
        var p = aabb_min;
        if (plane.x >= 0.0) { p.x = aabb_max.x; }
        if (plane.y >= 0.0) { p.y = aabb_max.y; }
        if (plane.z >= 0.0) { p.z = aabb_max.z; }
                
        if (dot(plane.xyz, p) + plane.w < 0.0) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn culling_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_id: u32
) {
    // Инициализируем локальный счетчик группы силами первого потока
    if (local_id == 0u) {
        atomicStore(&wg_visible_count, 0u);
    }
    // Синхронизация: гарантируем, что память инициализирована всеми 64 потоками перед работой
    workgroupBarrier();

    let task = culling_tasks[0u]; 
    let global_instance_id = task.start_object_index + global_id.x;
    
    // Проверяем, находится ли поток в границах массива объектов кадра
    if (global_id.x < task.object_count) {
        let instance = global_instances[global_instance_id];
        let mesh_info = global_mesh_infos[instance.primitive_index];

        // --- ЭТАП 1: Куллинг объекта целиком ---
        let world_object_aabb = transform_aabb(instance.aabb_min, instance.aabb_max, instance.model_matrix);
        
        if (is_aabb_visible(world_object_aabb[0], world_object_aabb[1])) {
            
            // --- ЭТАП 2: Куллинг мешлетов внутри видимого объекта ---
            for (var m_idx = 0u; m_idx < mesh_info.meshlet_count; m_idx = m_idx + 1u) {
                let global_meshlet_id = mesh_info.start_meshlet_index + m_idx;
                let meshlet = global_meshlets[global_meshlet_id];
                
                let world_meshlet_aabb = transform_aabb(meshlet.aabb_min, meshlet.aabb_max, instance.model_matrix);
                
                if (is_aabb_visible(world_meshlet_aabb[0], world_meshlet_aabb[1])) {
                    // Мешлет видим! Атомарно занимаем слот внутри ЛОКАЛЬНОГО кэша группы
                    let local_slot = atomicAdd(&wg_visible_count, 1u);
                    
                    // Защита от переполнения локального массива (256 элементов на группу из 64 потоков)
                    if (local_slot < 256u) {
                        wg_local_tasks[local_slot].instance_id = global_instance_id;
                        wg_local_tasks[local_slot].global_meshlet_id = global_meshlet_id;
                        wg_local_tasks[local_slot].index_count = meshlet.index_count;
                        wg_local_tasks[local_slot].material_index = instance.material_index;
                    }
                }
            }
        }
    }

    // Синхронизация: ждем, пока ВСЕ потоки группы завершат обход своих мешлетов и запишут данные в кэш
    workgroupBarrier();

    // --- ЭТАП 3: Выделение места в глобальной VRAM ОДНИМ запросом от группы ---
    let total_wg_visible = atomicLoad(&wg_visible_count);
    let safe_wg_visible = min(total_wg_visible, 256u); // Отрезаем по лимиту кэша

    if (local_id == 0u) {
        if (safe_wg_visible > 0u) {
            // Только ОДИН поток лезет в медленную глобальную память и сдвигает счетчик сразу на N элементов
            wg_global_offset = atomicAdd(&command_counter.count, safe_wg_visible);
        }
    }
    // Синхронизация: расшариваем полученный глобальный сдвиг на всю группу
    workgroupBarrier();

    // --- ЭТАП 4: Распараллеливание записи из кэша в глобальный буфер VRAM ---
    // Распределяем накопленные в группе задачи (safe_wg_visible) между 64 потоками группы.
    // Если задач больше 64, потоки сделают несколько шагов (цикл по шагу 64).
    for (var i = local_id; i < safe_wg_visible; i = i + 64u) {
        let local_task = wg_local_tasks[i];
        let cmd_id = wg_global_offset + i;

        // Записываем метаданные для вершинного шейдера
        visible_instances[cmd_id].instance_id = local_task.instance_id;
        visible_instances[cmd_id].material_index = local_task.material_index;
        visible_instances[cmd_id].meshlet_index = local_task.global_meshlet_id;

        // Формируем финальную плотную indirect-команду для графического чипа
        var cmd: DrawIndexedIndirectCommand;
        cmd.index_count = local_task.index_count;
        cmd.instance_count = 1u; // 1 экземпляр = 1 мешлет
        cmd.first_index = 0u;
        cmd.base_vertex = 0;
        cmd.first_instance = cmd_id; // Пробрасываем cmd_id в @builtin(instance_index)

        indirect_commands[cmd_id] = cmd;
    }
}
