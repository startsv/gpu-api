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

struct NodeData {
    info: vec4<u32>,
    transform: mat4x4<f32>,
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

struct VisibleMeshletData {
    meshlet_id: u32,
    material_index: u32,
};

struct Meshlet {
    vertex_offset: u32,
    vertex_count: u32,
    index_offset: u32,
    triangle_count: u32,
    instance_id: u32, 
};

@group(2) @binding(0) var<storage, read> global_nodes: array<NodeData>;
@group(2) @binding(1) var global_joint_texture: texture_2d<f32>;
@group(2) @binding(2) var<storage, read> global_instances: array<InstanceData>;
@group(2) @binding(3) var<storage, read> global_meshlets: array<Meshlet>; // Переехал на binding 3

struct VertexInput {    
    @location(0) position: vec3<f32>,    
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
    @location(5) joints: vec4<u32>,
    @location(6) weights: vec4<f32>
};

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) bitangent: vec3<f32>,
    @location(4) world_position: vec3<f32>,
    @location(5) @interpolate(flat) material_index: u32, 
};

fn read_baked_matrix(matrix_index: u32) -> mat4x4<f32> {
    let texture_width: u32 = 2048u;
    let matrices_per_row: u32 = 512u;

    let row = matrix_index / matrices_per_row;
    let col_matrix_offset = (matrix_index % matrices_per_row) * 4u;

    let y = i32(row);
    let x = i32(col_matrix_offset);
    
    let row0 = textureLoad(global_joint_texture, vec2<i32>(x + 0, y), 0);
    let row1 = textureLoad(global_joint_texture, vec2<i32>(x + 1, y), 0);
    let row2 = textureLoad(global_joint_texture, vec2<i32>(x + 2, y), 0);
    let row3 = textureLoad(global_joint_texture, vec2<i32>(x + 3, y), 0);

    return mat4x4<f32>(row0, row1, row2, row3);
}

@vertex
fn vs_main(
    vertex_input: VertexInput, 
    @builtin(instance_index) global_meshlet_id: u32 // Напрямую равен ID мешлета!
) -> FragmentInput {    
    let meshlet = global_meshlets[global_meshlet_id];
    let instance = global_instances[meshlet.instance_id];
    var model_matrix = instance.model_matrix;
    let node = global_nodes[instance.node_index];
    
    // Вычисление скелетной анимации (остается вашей оригинальной логикой)
    if (instance.is_animated == 1u) {
        if (node.info[0] == 1u) {
            model_matrix = model_matrix * node.transform;
        } else {
            let j0 = instance.joints_offset + vertex_input.joints[0];
            let j1 = instance.joints_offset + vertex_input.joints[1];
            let j2 = instance.joints_offset + vertex_input.joints[2];
            let j3 = instance.joints_offset + vertex_input.joints[3];

            let m0 = read_baked_matrix(j0);
            let m1 = read_baked_matrix(j1);
            let m2 = read_baked_matrix(j2);
            let m3 = read_baked_matrix(j3);

            var skin_matrix: mat4x4<f32> = 
                vertex_input.weights[0] * m0 + 
                vertex_input.weights[1] * m1 + 
                vertex_input.weights[2] * m2 + 
                vertex_input.weights[3] * m3;

            model_matrix = model_matrix * skin_matrix * node.transform;            
        }        
    } else {
        model_matrix = model_matrix * node.transform;
    }

    let model_position = model_matrix * vec4<f32>(vertex_input.position, 1.0);
    
    var out: FragmentInput;
    out.clip_position = camera.view_proj * model_position; 
    out.world_position = model_position.xyz;
    out.uv = vertex_input.uv;
    out.material_index = instance.material_index; // Читаем материал напрямую из инстанса
        
    let normal_matrix = mat3x3<f32>(model_matrix[0].xyz, model_matrix[1].xyz, model_matrix[2].xyz);
    out.normal = normalize(normal_matrix * vertex_input.normal);
    out.tangent = normalize(normal_matrix * vertex_input.tangent);
    out.bitangent = normalize(normal_matrix * vertex_input.bitangent);
    
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
