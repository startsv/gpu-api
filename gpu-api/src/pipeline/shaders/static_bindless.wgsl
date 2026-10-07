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
    view_proj: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
};
@group(1) @binding(0) var<uniform> camera: CameraUniform;

struct StaticVertex {    
    position: array<f32, 3>,
    pad0: f32,                // Дополняем до vec4 (16 байт)
    uv: array<f32, 2>,
    pad1: array<f32, 2>,      // Дополняем до vec4 (16 байт)
    normal: array<f32, 3>,
    pad2: f32,                // Дополняем до vec4 (16 байт)
    tangent: array<f32, 3>,
    pad3: f32,                // Дополняем до vec4 (16 байт)
    bitangent: array<f32, 3>,
    pad4: f32,                // Дополняем до vec4 (16 байт)
};



struct NodeData {
    info: vec4<u32>,
    transform: mat4x4<f32>,
};

struct MeshInfo {
    start_meshlet_index: u32,
    meshlet_count: u32,
    vertex_buffer_offset: u32,
    base_vertex: i32,
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

struct VisibleInstanceData {
    instance_id: u32,
    material_index: u32,
};

@group(2) @binding(0) var<storage, read> static_vertices: array<StaticVertex>;
@group(2) @binding(1) var<storage, read> global_nodes: array<NodeData>;
@group(2) @binding(2) var<storage, read> global_instances: array<InstanceData>;
@group(2) @binding(3) var<storage, read> global_mesh_infos: array<MeshInfo>;
@group(2) @binding(4) var<storage, read> visible_instances: array<VisibleInstanceData>;

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) bitangent: vec3<f32>,
    @location(4) world_position: vec3<f32>,
    @location(5) @interpolate(flat) material_index: u32, 
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) draw_instance_idx: u32
) -> FragmentInput {    
    let vertex = static_vertices[vertex_id];
    
    // Распаковываем массивы в нормальные вектора WGSL
    let raw_pos = vec3<f32>(vertex.position[0], vertex.position[1], vertex.position[2]);
    let raw_uv = vec2<f32>(vertex.uv[0], vertex.uv[1]);

    // Тестовая матрица (Куб перед камерой)
    let hardcoded_model_matrix = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, -5.0, 1.0) 
    );
    
    let model_position = hardcoded_model_matrix * vec4<f32>(raw_pos, 1.0);
    
    var out: FragmentInput;
    out.clip_position = camera.view_proj * model_position; 
    out.world_position = model_position.xyz;
    out.uv = raw_uv;
    out.material_index = 0u;
    
    out.normal = vec3<f32>(vertex.normal[0], vertex.normal[1], vertex.normal[2]);
    out.tangent = vec3<f32>(vertex.tangent[0], vertex.tangent[1], vertex.tangent[2]);
    out.bitangent = vec3<f32>(vertex.bitangent[0], vertex.bitangent[1], vertex.bitangent[2]);
    
    return out;    
}


@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {    
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
