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
}

struct DrawIndexedIndirectCommand {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: u32,
    first_instance: u32, // Сюда ваш Compute-шейдер пишет индекс мешлета в visible_instances
}

@group(1) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(1) var<storage, read> indirect_commands: array<DrawIndexedIndirectCommand>;

struct StaticVertex {    
    position: vec3<f32>,
    pad0: f32,
    uv: vec2<f32>,
    pad1: vec2<f32>,
    normal: vec3<f32>,
    pad2: f32,
    tangent: vec3<f32>,
    pad3: f32,
    bitangent: vec3<f32>,
    pad4: f32,
}

struct NodeData {
    info: vec4<u32>,
    transform: mat4x4<f32>,
}

struct MeshInfo {
    start_meshlet_index: u32,
    meshlet_count: u32,
    vertex_buffer_offset: u32,
    base_vertex: i32,
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

struct VisibleInstanceData {
    instance_id: u32,
    material_index: u32,
    meshlet_index: u32, 
}

@group(2) @binding(0) var<storage, read> static_vertices: array<StaticVertex>;
@group(2) @binding(1) var<storage, read> global_nodes: array<NodeData>;
@group(2) @binding(2) var<storage, read> global_instances: array<InstanceData>;
@group(2) @binding(3) var<storage, read> global_mesh_infos: array<MeshInfo>;
@group(2) @binding(4) var<storage, read> visible_instances: array<VisibleInstanceData>;

@group(2) @binding(5) var<storage, read> global_meshlets: array<StaticMeshletDescription>;
@group(2) @binding(6) var<storage, read> meshlet_local_indices: array<u32>;
@group(2) @binding(7) var<storage, read> meshlet_vertex_redirect: array<u32>;

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) bitangent: vec3<f32>,
    @location(4) world_position: vec3<f32>,
    @location(5) @interpolate(flat) material_index: u32, 
}

@vertex
fn vs_main(
    // Благодаря смещению first_index, этот ID гарантированно равен: (cmd_id * 18) + локальный_индекс
    @builtin(vertex_index) hardware_vertex_id: u32 
) -> FragmentInput {    
    
    let indices_per_meshlet = 18u; 
    
    // Математика теперь работает со 100% стабильностью, так как hardware_vertex_id растет непрерывно
    let cmd_id = hardware_vertex_id / indices_per_meshlet;
    let vertex_id = hardware_vertex_id % indices_per_meshlet;
    
    // Вся остальная выборка данных остается без изменений...
    let render_data = visible_instances[cmd_id];
    
    let instance = global_instances[render_data.instance_id];
    let meshlet = global_meshlets[render_data.meshlet_index];
    let mesh_info = global_mesh_infos[instance.primitive_index];
    
    let local_index_address = meshlet.index_offset + vertex_id;
    let local_vertex_id = meshlet_local_indices[local_index_address];
    
    let redirect_address = meshlet.vertex_offset + local_vertex_id;
    let actual_vertex_id = meshlet_vertex_redirect[redirect_address];
    
    let global_vertex_idx = actual_vertex_id + mesh_info.vertex_buffer_offset;
    let vertex = static_vertices[global_vertex_idx];
    
    let model_matrix = instance.model_matrix;
    let model_position = model_matrix * vec4<f32>(vertex.position, 1.0);
    
    var out: FragmentInput;
    out.clip_position = camera.view_proj * model_position; 
    out.world_position = model_position.xyz;
    out.uv = vertex.uv;
    out.material_index = instance.material_index; 
    
    let normal_matrix = mat3x3<f32>(model_matrix[0].xyz, model_matrix[1].xyz, model_matrix[2].xyz);
    out.normal = normalize(normal_matrix * vertex.normal);
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
