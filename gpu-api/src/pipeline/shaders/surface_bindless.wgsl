enable wgpu_binding_array;

@group(0) @binding(0) var base_color_textures: binding_array<texture_2d<f32>>;
@group(0) @binding(1) var base_color_samplers: binding_array<sampler>;
@group(0) @binding(2) var metallic_roughness_textures: binding_array<texture_2d<f32>>;
@group(0) @binding(3) var metallic_roughness_samplers: binding_array<sampler>;
@group(0) @binding(4) var normal_textures: binding_array<texture_2d<f32>>;
@group(0) @binding(5) var normal_samplers: binding_array<sampler>;
@group(0) @binding(6) var emissive_textures: binding_array<texture_2d<f32>>;
@group(0) @binding(7) var emissive_samplers: binding_array<sampler>;

struct MaterialFactors {
    base_color_factor: vec4<f32>,
    emissive_factor: vec3<f32>,
    metallic_factor: f32,
    roughness_factor: f32,
    padding: vec3<f32>,
}
@group(0) @binding(8) var<storage, read> global_materials: array<MaterialFactors>;

struct CameraUniform {
    camera_position: vec3<f32>,
    padding: u32,
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
};
@group(1) @binding(0) var<uniform> camera: CameraUniform;

struct TerrainVertex {
    position: vec3<f32>,
    pad0: f32,
    normal: vec3<f32>,
    pad1: f32,
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

@group(2) @binding(0) var<storage, read> terrain_vertices: array<TerrainVertex>;
@group(2) @binding(1) var<storage, read> terrain_indices: array<u32>;
@group(2) @binding(2) var<storage, read> terrain_meshlets: array<TerrainMeshletDescription>;
@group(2) @binding(3) var<storage, read> active_meshlets: array<u32>;

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) material_index: u32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) meshlet_id: u32
) -> FragmentInput {            
    let meshlet = terrain_meshlets[meshlet_id];
    let vertex = terrain_vertices[vertex_id];

    var out: FragmentInput;
    let world_pos = vec4<f32>(vertex.position, 1.0);
            
    out.clip_position = camera.projection * world_pos;
    
    out.world_position = world_pos.xyz;
    out.normal = vertex.normal;
    out.uv = world_pos.xz * 0.05;
    out.material_index = meshlet.material_index;
    
    return out;
}

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    //return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    
    let mat_idx = in.material_index;
    let factors = global_materials[mat_idx];
    
    let base_color = textureSample(
        base_color_textures[mat_idx], 
        base_color_samplers[mat_idx], 
        in.uv
    ) * factors.base_color_factor;

    let normal_map = textureSample(
        normal_textures[mat_idx], 
        normal_samplers[mat_idx], 
        in.uv
    );

    let metallic_roughness = textureSample(
        metallic_roughness_textures[mat_idx], 
        metallic_roughness_samplers[mat_idx], 
        in.uv
    );    

    return base_color;   
}
