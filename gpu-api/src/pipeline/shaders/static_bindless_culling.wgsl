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
    primitive_index: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    aabb_min: vec3<f32>,
    pad_aabb1: u32,
    aabb_max: vec3<f32>,
    pad_aabb2: u32,
};

// Задача на куллинг теперь описывает диапазон мешлетов
struct CullingTask {
    start_meshlet_index: u32,
    meshlet_count: u32,
    lod_level: u32,
    _padding: u32,
};

struct Meshlet {
    vertex_offset: u32,
    vertex_count: u32,
    index_offset: u32,
    triangle_count: u32,
    instance_id: u32,          // Указывает на соответствующий global_instances
    bounding_center: vec3<f32>, // Центр сферы мешлета в локальных координатах
    bounding_radius: f32,     // Радиус сферы мешлета
};

struct DrawIndexedIndirectCommand {
    index_count: u32,
    instance_count: u32, // Для мешлетов всегда равен 1
    first_index: u32,
    base_vertex: i32,
    first_instance: u32, // Сюда запишем локальный индекс в visible_meshlets
};

struct VisibleMeshletData {
    meshlet_id: u32,
    material_index: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var<storage, read> culling_tasks: array<CullingTask>;
@group(1) @binding(1) var<storage, read> global_instances: array<InstanceData>;
@group(1) @binding(2) var<storage, read> global_meshlets: array<Meshlet>;

// Выходные буферы
@group(1) @binding(3) var<storage, read_write> visible_meshlets: array<VisibleMeshletData>;
@group(1) @binding(4) var<storage, read_write> indirect_commands: array<DrawIndexedIndirectCommand>;
// Глобальный атомарный счетчик видимых мешлетов (перед куллингом на CPU сбрасывается в 0)
@group(1) @binding(5) var<storage, read_write> global_draw_counter: atomic<u32>;

// Для мешлетов куллинг по сфере (Bounding Sphere) значительно быстрее и эффективнее, чем по AABB
fn is_sphere_visible(center: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = camera.frustum_planes[i];
        if (dot(plane.xyz, center) + plane.w < -radius) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn culling_main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let task_index = workgroup_id.x;
    let total_tasks = arrayLength(&culling_tasks);
    if (task_index >= total_tasks) { return; }
    
    let task = culling_tasks[task_index];
        
    // Итерируемся по мешлетам внутри задачи
    for (var i = local_id.x; i < task.meshlet_count; i = i + 64u) {
        let global_meshlet_id = task.start_meshlet_index + i;
        let meshlet = global_meshlets[global_meshlet_id];
        let instance = global_instances[meshlet.instance_id];
                
        let m = instance.model_matrix;
        
        // 1. Трансформируем Bounding Sphere мешлета в World Space
        let world_center = (m * vec4<f32>(meshlet.bounding_center, 1.0)).xyz;
        
        // Вычисляем масштаб из матрицы для корректного радиуса в мировых координатах
        let scale_x = length(m[0].xyz);
        let scale_y = length(m[1].xyz);
        let scale_z = length(m[2].xyz);
        let max_scale = max(scale_x, max(scale_y, scale_z));
        let world_radius = meshlet.bounding_radius * max_scale;
        
        // 2. Тест видимости сферы мешлета во фрустуме камеры
        if (is_sphere_visible(world_center, world_radius)) {                                    
            
            // Аллоцируем глобальный слот под эту команду отрисовки мешлета
            let write_index = atomicAdd(&global_draw_counter, 1u);
            
            // Заполняем MDI команду для рендера конкретного мешлета
            indirect_commands[write_index].index_count = meshlet.triangle_count * 3u;
            indirect_commands[write_index].instance_count = 1u; // Ровно один инстанс мешлета
            indirect_commands[write_index].first_index = meshlet.index_offset;
            indirect_commands[write_index].base_vertex = i32(meshlet.vertex_offset);
            
            // Связываем Vertex Shader с текущим элементом visible_meshlets через first_instance
            indirect_commands[write_index].first_instance = write_index;
                        
            // Сохраняем метаданные для чтения в VS/FS
            visible_meshlets[write_index].meshlet_id = global_meshlet_id;
            visible_meshlets[write_index].material_index = instance.material_index;            
        }
    }
}
