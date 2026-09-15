struct CameraUniform {
    camera_position: vec3<f32>,
    padding: u32,
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
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
@group(1) @binding(2) var<storage, read_write> visible_instances: array<VisibleInstanceData>;
@group(1) @binding(3) var<storage, read_write> indirect_commands: array<DrawIndexedIndirectCommand>;

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
    let total_tasks = arrayLength(&culling_tasks);
    if (task_index >= total_tasks) { return; }
    
    let task = culling_tasks[task_index];
    let chunk_lod = task.lod_level;
        
    for (var i = local_id.x; i < task.object_count; i = i + 64u) {
        let global_instance_id = task.start_object_index + i;
        let instance = global_instances[global_instance_id];
                
        let m = instance.model_matrix;
        
        let aabb_min_vec = instance.aabb_min.xyz;
        let aabb_max_vec = instance.aabb_max.xyz;
        
        let center = (aabb_min_vec + aabb_max_vec) * 0.5;
        let extents = (aabb_max_vec - aabb_min_vec) * 0.5;
                
        let world_center = (m * vec4<f32>(center, 1.0)).xyz;
                                        
        let row0 = vec3<f32>(abs(m[0].x), abs(m[1].x), abs(m[2].x));
        let row1 = vec3<f32>(abs(m[0].y), abs(m[1].y), abs(m[2].y));
        let row2 = vec3<f32>(abs(m[0].z), abs(m[1].z), abs(m[2].z));
                
        let world_extents = vec3<f32>(
            dot(row0, extents),
            dot(row1, extents),
            dot(row2, extents)
        );
        
        let world_min = world_center - world_extents;
        let world_max = world_center + world_extents;
        
        if (is_aabb_visible(world_min, world_max)) {                                    
            let cmd_id = (instance.primitive_index * 3u) + chunk_lod;                                                        
            let local_slot = atomicAdd(&indirect_commands[cmd_id].instance_count, 1u);                                                                    
            let base_offset = indirect_commands[cmd_id].first_instance;
            let write_index = base_offset + local_slot;
                        
            visible_instances[write_index].instance_id = global_instance_id;
            visible_instances[write_index].material_index = instance.material_index;            
        }
    }
}
