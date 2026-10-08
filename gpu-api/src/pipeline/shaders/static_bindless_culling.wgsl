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

@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var<storage, read> culling_tasks: array<CullingTask>;
@group(1) @binding(1) var<storage, read> global_instances: array<InstanceData>;
@group(1) @binding(2) var<storage, read> global_mesh_infos: array<MeshInfo>;
@group(1) @binding(3) var<storage, read> global_meshlets: array<StaticMeshletDescription>;

@group(1) @binding(4) var<storage, read_write> visible_instances: array<VisibleInstanceData>;
@group(1) @binding(5) var<storage, read_write> indirect_commands: array<DrawIndexedIndirectCommand>;
@group(1) @binding(6) var<storage, read_write> command_counter: IndirectCount;

fn transform_aabb(aabb_min: vec3<f32>, aabb_max: vec3<f32>, m: mat4x4<f32>) -> array<vec3<f32>, 2> {
    let center = (aabb_min + aabb_max) * 0.5;
    let extents = (aabb_max - aabb_min) * 0.5;
    let world_center = (m * vec4<f32>(center, 1.0)).xyz;
    
    let row0 = vec3<f32>(abs(m[0].x), abs(m[1].x), abs(m[2].x));
    let row1 = vec3<f32>(abs(m[0].y), abs(m[1].y), abs(m[2].y));
    let row2 = vec3<f32>(abs(m[0].z), abs(m[1].z), abs(m[2].z));
    
    let world_extents = vec3<f32>(
        dot(row0, extents),
        dot(row1, extents),
        dot(row2, extents)
    );
    
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
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let task = culling_tasks[0u]; 
    let global_instance_id = task.start_object_index + global_id.x;
    
    // Безопасный выход за границы массива инстансов
    if (global_id.x >= task.object_count) { return; }
    
    let instance = global_instances[global_instance_id];
    let mesh_info = global_mesh_infos[instance.primitive_index];

    // ХАРДКОД ДЛЯ ТЕСТА: Полностью отключаем culling. 
    // Говорим GPU, что ВСЕ объекты всегда 100% видимы!
    let object_visible = true; 
    
    if (object_visible) {
            // ... внутри цикла по мешлетам ...
        for (var m_idx = 0u; m_idx < mesh_info.meshlet_count; m_idx = m_idx + 1u) {
            let global_meshlet_id = mesh_info.start_meshlet_index + m_idx;
            let meshlet = global_meshlets[global_meshlet_id];
            
            // Атомарно инкрементируем счетчик
            let cmd_id = atomicAdd(&command_counter.count, 1u);
            
            // КРИТИЧЕСКАЯ ЗАЩИТА: Заменяем hardcode-лимит на максимальный размер ваших буферов.
            // Для 100 инстансов по 2 мешлета максимальный cmd_id должен быть СТРОГО меньше 200!
            let max_allowed_commands = 200u; 
                                        
            if (cmd_id < max_allowed_commands) {
                visible_instances[cmd_id].instance_id = global_instance_id;
                visible_instances[cmd_id].material_index = instance.material_index;
                visible_instances[cmd_id].meshlet_index = global_meshlet_id;

                var cmd: DrawIndexedIndirectCommand;
                cmd.index_count = meshlet.index_count; 
                cmd.instance_count = 1u;               
                cmd.first_index = cmd_id * 18u; // Наш сквозной индекс
                cmd.base_vertex = 0;           
                cmd.first_instance = 0u;       

                indirect_commands[cmd_id] = cmd;
            }
        }
    }
}
