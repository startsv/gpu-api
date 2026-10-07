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
    primitive_index: u32, // Будем использовать как mesh_info_index (всегда 0 для куба)
    base_command_id: u32, // Переименовали pad0! Сюда запишем i * 2 для indirect-команд
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
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var<storage, read> culling_tasks: array<CullingTask>;
@group(1) @binding(1) var<storage, read> global_instances: array<InstanceData>;
@group(1) @binding(2) var<storage, read> global_mesh_infos: array<MeshInfo>;
@group(1) @binding(3) var<storage, read> global_meshlets: array<StaticMeshletDescription>;

@group(1) @binding(4) var<storage, read_write> visible_instances: array<VisibleInstanceData>;
@group(1) @binding(5) var<storage, read_write> indirect_commands: array<DrawIndexedIndirectCommand>;

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
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let global_instance_id = global_id.x;
    let task = culling_tasks[0u]; 
    
    if (global_instance_id >= task.object_count) { return; }
    
    let instance = global_instances[global_instance_id];
    let mesh_info = global_mesh_infos[instance.primitive_index];
            
    for (var m_idx = 0u; m_idx < mesh_info.meshlet_count; m_idx = m_idx + 1u) {
        
        // ВРЕМЕННО ХАРДКОДИМ ИСТИНУ ДЛЯ ОТЛАДКИ ВЫВОДА ВСЕХ КУБОВ
        if (true) {                                
            // cmd_id равен строго 0 или 1 (индекс мешлета)
            let cmd_id = m_idx; 
                            
            // Атомарно увеличиваем количество инстансов для этого мешлета
            let local_slot = atomicAdd(&indirect_commands[cmd_id].instance_count, 1u);
                            
            // Вычисляем плотный индекс записи
            // Для Мешлета 0: 0 + local_slot (диапазон 0..99)
            // Для Мешлета 1: 100 + local_slot (диапазон 100..199)
            let base_offset = indirect_commands[cmd_id].first_instance;
            let write_index = base_offset + local_slot;
                                        
            visible_instances[write_index].instance_id = global_instance_id;
            visible_instances[write_index].material_index = instance.material_index;
        }
    }
}
