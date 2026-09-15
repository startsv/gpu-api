struct CameraUniform {
    camera_position: vec3<f32>,
    padding: u32,
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
};

struct TerrainMeshletDescription {
    aabb_min: vec3<f32>,
    vertex_offset: u32,
    aabb_max: vec3<f32>,
    index_offset: u32,
    index_count: u32,
    material_index: u32,
    pad0: u32,
    pad1: u32,
};

struct TerrainCullingTask {
    start_meshlet_index: u32,
    meshlet_count: u32,
    indirect_cmd_index: u32, 
    _padding: u32,
};

struct DrawIndexedIndirectCommand {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<storage, read> terrain_tasks: array<TerrainCullingTask>;
@group(1) @binding(1) var<storage, read> terrain_meshlets: array<TerrainMeshletDescription>;

@group(1) @binding(3) var<storage, read_write> draw_commands: array<DrawIndexedIndirectCommand>;

fn is_aabb_visible(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = camera.frustum_planes[i];
        var p = aabb_min;
        if (plane.x >= 0.0) { p.x = aabb_max.x; }
        if (plane.y >= 0.0) { p.y = aabb_max.y; }
        if (plane.z >= 0.0) { p.z = aabb_max.z; }
        if (dot(plane.xyz, p) + plane.w < 0.0) { return false; }
    }
    return true;
}

@compute @workgroup_size(64)
fn culling_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {    
    let task = terrain_tasks[0u];   
    let local_meshlet_id = global_id.x;
    if (local_meshlet_id >= task.meshlet_count) {
        return;
    }
    
    let global_meshlet_id = task.start_meshlet_index + local_meshlet_id;
    let meshlet = terrain_meshlets[global_meshlet_id];
    let cmd_index = task.indirect_cmd_index + local_meshlet_id;

    if (is_aabb_visible(meshlet.aabb_min, meshlet.aabb_max)) {
        draw_commands[cmd_index].index_count = meshlet.index_count;
        draw_commands[cmd_index].instance_count = 1u;
        draw_commands[cmd_index].first_index = meshlet.index_offset;
        draw_commands[cmd_index].base_vertex = 0u;
        draw_commands[cmd_index].first_instance = 0u;
    } else {
        draw_commands[cmd_index].instance_count = 0u;
        draw_commands[cmd_index].index_count = 0u;
    }
}
