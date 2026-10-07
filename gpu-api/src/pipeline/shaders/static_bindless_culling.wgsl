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
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let task_index = workgroup_id.x;
    if (task_index >= arrayLength(&culling_tasks)) { return; }
    
    let task = culling_tasks[task_index];
        
    for (var i = local_id.x; i < task.object_count; i = i + 64u) {
        let global_instance_id = task.start_object_index + i;
        let instance = global_instances[global_instance_id];
        let mesh_info = global_mesh_infos[instance.primitive_index];
                
        let m = instance.model_matrix;
        
        for (var m_idx = 0u; m_idx < mesh_info.meshlet_count; m_idx = m_idx + 1u) {
            let global_meshlet_id = mesh_info.start_meshlet_index + m_idx;
            let meshlet = global_meshlets[global_meshlet_id];
            
            // ВРЕМЕННО ХАРДКОДИМ ИСТИНУ, ЧТОБЫ УВИДЕТЬ ВСЕ КУБЫ
            if (true) {                                
                let cmd_id = instance.base_command_id + m_idx;
                                
                indirect_commands[cmd_id].index_count = meshlet.index_count;
                indirect_commands[cmd_id].first_index = meshlet.index_offset;
                indirect_commands[cmd_id].base_vertex = 0; // Строго 0 для Vertex Pulling!
                
                // Выставляем instance_count в 1, чтобы GPU нарисовал этот мешлет
                indirect_commands[cmd_id].instance_count = 1u;
                                
                // ПРЯМАЯ ЗАПИСЬ: Записываем ID инстанса прямо в слот, равный cmd_id
                let write_index = cmd_id;
                                            
                visible_instances[write_index].instance_id = global_instance_id;
                visible_instances[write_index].material_index = instance.material_index;
            } else {
                // Если мешлет не прошел куллинг (в будущем)
                let cmd_id = instance.base_command_id + m_idx;
                indirect_commands[cmd_id].instance_count = 0u;
            }
        }
    }
}


