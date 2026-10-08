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
    pad0: u32, pad1: u32, pad2: u32,
    aabb_min: vec3<f32>, pad_aabb1: u32,
    aabb_max: vec3<f32>, pad_aabb2: u32,
};

struct Meshlet {
    vertex_offset: u32,
    vertex_count: u32,
    index_offset: u32,
    triangle_count: u32,
    instance_id: u32,          
    bounding_center: vec3<f32>, 
    bounding_radius: f32,     
};

// Структура команды MDI (ровно 20 байт, стандартный плотный шаг)
struct DrawIndexedIndirectCommand {
    index_count: u32,
    instance_count: u32, // Сюда мы будем писать 1u (видим) или 0u (скрыт)
    first_index: u32,
    base_vertex: i32,
    first_instance: u32, 
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var<storage, read> global_instances: array<InstanceData>;
@group(1) @binding(1) var<storage, read> global_meshlets: array<Meshlet>;
// Шейдер пишет напрямую в instance_count существующей команды
@group(1) @binding(2) var<storage, read_write> indirect_commands: array<DrawIndexedIndirectCommand>;

fn is_sphere_visible(center: vec3<f32>, radius: f32) -> bool {
    return true;
}

@compute @workgroup_size(64)
fn culling_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let global_meshlet_id = global_id.x;
    let total_meshlets = arrayLength(&global_meshlets);
    
    if (global_meshlet_id >= total_meshlets) { return; }
    
    let meshlet = global_meshlets[global_meshlet_id];
    let instance = global_instances[meshlet.instance_id];
    let m = instance.model_matrix;
    
    // 1. Трансформируем сферу мешлета в мировые координаты
    let world_center = (m * vec4<f32>(meshlet.bounding_center, 1.0)).xyz;
    let max_scale = max(length(m[0].xyz), max(length(m[1].xyz), length(m[2].xyz)));
    let world_radius = meshlet.bounding_radius * max_scale;
    
    // 2. Тест фрустума
    if (is_sphere_visible(world_center, world_radius)) {
        // Мешлет видим: активируем команду отрисовки
        indirect_commands[global_meshlet_id].instance_count = 1u;
    } else {
        // Мешлет скрыт: отключаем команду отрисовки
        indirect_commands[global_meshlet_id].instance_count = 0u;
    }
}
