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
    position: vec3<f32>,
    pad0: f32,                // Дополняем до vec4 (16 байт)
    uv: vec2<f32>,
    pad1: vec2<f32>,      // Дополняем до vec4 (16 байт)
    normal: vec3<f32>,
    pad2: f32,                // Дополняем до vec4 (16 байт)
    tangent: vec3<f32>,
    pad3: f32,                // Дополняем до vec4 (16 байт)
    bitangent: vec3<f32>,
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
    // draw_instance_idx теперь напрямую указывает на ID команды/мешлета на сцене!
    let render_data = visible_instances[draw_instance_idx];
    let instance = global_instances[render_data.instance_id];
    let mesh_info = global_mesh_infos[instance.primitive_index];
    
    // Ручной сдвиг вершины в мега-буфере статики
    let global_vertex_idx = vertex_id + mesh_info.vertex_buffer_offset;
    let vertex = static_vertices[global_vertex_idx];
    
    // Восстановление Std430 vec3 полей из массивов
    let raw_pos = vertex.position;
    let raw_uv = vertex.uv;
    let raw_normal = vertex.normal;

    let model_matrix = instance.model_matrix;
    let model_position = model_matrix * vec4<f32>(raw_pos, 1.0);
    
    var out: FragmentInput;
    out.clip_position = camera.view_proj * model_position; 
    out.world_position = model_position.xyz;
    out.uv = raw_uv;
    out.material_index = render_data.material_index; 
        
    let normal_matrix = mat3x3<f32>(model_matrix[0].xyz, model_matrix[1].xyz, model_matrix[2].xyz);
    out.normal = normalize(normal_matrix * raw_normal);
    out.tangent = normalize(normal_matrix * vertex.tangent);
    out.bitangent = normalize(normal_matrix * vertex.bitangent);
    
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
