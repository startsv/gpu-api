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

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) bitangent: vec3<f32>,
    @location(4) world_position: vec3<f32>,
    @location(5) @interpolate(flat) material_index: u32, 
}

// =====================================================================
// ГРУППА 0: СТАТИЧЕСКАЯ ГЛОБАЛЬНАЯ ГРУППА МАТЕРИАЛОВ (materials_bind_group)
// =====================================================================
// Здесь при необходимости могут быть массивы текстур или глобальные параметры материалов
// Для примера оставим пустое место или заглушку, соответствующую set_bind_group(0)

// =====================================================================
// ГРУППА 1: МОНОЛИТНАЯ КАДРОВАЯ ГРУППА (Соответствует render_bind_group на Rust)
// Нарезана через Buffer Slices. Драйвер GPU автоматически смещает массивы к 0-му индексу.
// =====================================================================
@group(1) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(1) var<storage, read> static_vertices: array<StaticVertex>;
@group(1) @binding(2) var<storage, read> global_nodes: array<NodeData>;
@group(1) @binding(3) var<storage, read> global_instances: array<InstanceData>;
@group(1) @binding(4) var<storage, read> global_mesh_infos: array<MeshInfo>;

// Выходные данные из Compute-пасса куллинга текущего кадра
@group(1) @binding(5) var<storage, read> visible_instances: array<VisibleInstanceData>;

// Тяжелые неизменяемые массивы мешлет-архитектуры сцены
@group(1) @binding(6) var<storage, read> global_meshlets: array<StaticMeshletDescription>;
@group(1) @binding(7) var<storage, read> meshlet_local_indices: array<u32>;
@group(1) @binding(8) var<storage, read> meshlet_vertex_redirect: array<u32>;

@vertex
fn vs_main(
    // Благодаря set_index_buffer(index_buffer.slice(..)) на Rust, 
    // этот ID аппаратно считывается как последовательность [0, 1, 2, 3...]
    @builtin(vertex_index) vertex_id: u32,
    // Наш уникальный виртуальный ID, проброшенный через cmd.first_instance из Compute-пасса
    @builtin(instance_index) cmd_id: u32
) -> FragmentInput {    
    
    // 1. Получаем метаданные отрисовки для текущего видимого мешлета
    let render_data = visible_instances[cmd_id];
    
    let instance = global_instances[render_data.instance_id];
    let meshlet = global_meshlets[render_data.meshlet_index];
    let mesh_info = global_mesh_infos[instance.primitive_index];
    
    // 2. Вычисляем локальный индекс вершины внутри мешлета (в пределах от 0 до 17)
    // Так как буфер индексов сквозной, мы просто берем оффсет текущего мешлета
    let local_index_address = meshlet.index_offset + vertex_id;
    let local_vertex_id = meshlet_local_indices[local_index_address];
    
    // 3. Перенаправляем локальный индекс на реальный индекс вершины в базовом кубе
    let redirect_address = meshlet.vertex_offset + local_vertex_id;
    let actual_vertex_id = meshlet_vertex_redirect[redirect_address];
    
    // 4. С учетом оффсета конкретного меша, получаем финальный индекс в глобальном буфере вершин
    // Для нашего куба mesh_info.vertex_buffer_offset равен 0.
    let global_vertex_idx = actual_vertex_id + mesh_info.vertex_buffer_offset;
    let vertex = static_vertices[global_vertex_idx];
    
    // 5. Трансформация геометрии
    let model_matrix = instance.model_matrix;
    let model_position = model_matrix * vec4<f32>(vertex.position, 1.0);
    
    // 6. Формирование выходных данных для фрагментного шейдера
    var out: FragmentInput;
    out.clip_position = camera.view_proj * model_position; 
    out.world_position = model_position.xyz;
    out.uv = vertex.uv;
    out.material_index = instance.material_index; 
    
    // Корректный перенос нормалей, тангенсов и битангенсов в мировое пространство
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
