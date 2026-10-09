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

// Заглушки, если структуры CullingTask и VisibleInstanceData еще объявлены в шейдере
struct CullingTask {
    start_object_index: u32,
    object_count: u32,
    lod_level: u32,
    _padding: u32,
};

struct VisibleInstanceData {
    instance_id: u32,
    material_index: u32,
};

struct Meshlet {
    vertex_offset: u32,
    vertex_count: u32,
    index_offset: u32,
    triangle_count: u32,
    
    instance_id: u32,

    bounding_center_x: f32,
    bounding_center_y: f32,
    bounding_center_z: f32,

    bounding_radius: f32,

    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct DrawIndexedIndirectCommand {
    index_count: u32,
    instance_count: u32, 
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

// --- ВАШИ ОРИГИНАЛЬНЫЕ БИНДИНГИ БЕЗ ИЗМЕНЕНИЙ ---
@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var<storage, read> culling_tasks: array<CullingTask>; // Не используется в статической схеме, но Layout сохранен
@group(1) @binding(1) var<storage, read> global_instances: array<InstanceData>;
@group(1) @binding(2) var<storage, read_write> visible_instances: array<VisibleInstanceData>; // Не используется, но Layout сохранен
@group(1) @binding(3) var<storage, read_write> indirect_commands: array<DrawIndexedIndirectCommand>;
@group(1) @binding(4) var<storage, read> global_meshlets: array<Meshlet>;

// Тест видимости коробки AABB
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

// Тест видимости сферы
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
fn culling_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let global_meshlet_id = global_id.x;
    let total_meshlets = arrayLength(&global_meshlets);
    
    if (global_meshlet_id >= total_meshlets) { return; }
    
    let meshlet = global_meshlets[global_meshlet_id];
    let instance = global_instances[meshlet.instance_id];
    let m = instance.model_matrix;
    
    // =================================================================
    // ЭТАП 1: ГРУБЫЙ КУЛЛИНГ (Проверяем объект целиком по AABB)
    // =================================================================
    let center = (instance.aabb_min.xyz + instance.aabb_max.xyz) * 0.5;
    let extents = (instance.aabb_max.xyz - instance.aabb_min.xyz) * 0.5;
            
    let object_world_center = (m * vec4<f32>(center, 1.0)).xyz;
                                    
    // ИСПРАВЛЕНО: Правильный доступ к индексам матрицы mat4x4
    let row0 = vec3<f32>(abs(m[0].x), abs(m[1].x), abs(m[2].x));
    let row1 = vec3<f32>(abs(m[0].y), abs(m[1].y), abs(m[2].y));
    let row2 = vec3<f32>(abs(m[0].z), abs(m[1].z), abs(m[2].z));
            
    let object_world_extents = vec3<f32>(
        dot(row0, extents),
        dot(row1, extents),
        dot(row2, extents)
    );
    
    let world_min = object_world_center - object_world_extents;
    let world_max = object_world_center + object_world_extents;
    
    if (!is_aabb_visible(world_min, world_max)) {
        indirect_commands[global_meshlet_id].instance_count = 0u;
        return; 
    }

    // =================================================================
    // ЭТАП 2: ТОНКИЙ КУЛЛИНГ (Проверяем конкретную сферу мешлета)
    // =================================================================
    let local_sphere_center = vec3<f32>(meshlet.bounding_center_x, meshlet.bounding_center_y, meshlet.bounding_center_z);
    
    let sphere_world_center = (m * vec4<f32>(local_sphere_center, 1.0)).xyz;
    
    // Правильный расчет масштаба из колонок матрицы
    let scale_x = length(m[0].xyz);
    let scale_y = length(m[1].xyz);
    let scale_z = length(m[2].xyz);
    let max_scale = max(scale_x, max(scale_y, scale_z));
    
    let sphere_world_radius = meshlet.bounding_radius * max_scale;
    
    if (is_sphere_visible(sphere_world_center, sphere_world_radius)) {
        indirect_commands[global_meshlet_id].instance_count = 1u;
    } else {
        indirect_commands[global_meshlet_id].instance_count = 0u;
    }
}
