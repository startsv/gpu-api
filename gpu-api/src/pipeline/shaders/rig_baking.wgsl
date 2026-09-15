//Универсальность объектов: Если вам нужно прикрепить к руке персонажа жесткий меч, вам больше не нужен node_transform. 
//Меч рендерится этим же шейдером. Его вершины будут иметь joints = vec4<u32>(HAND_BONE_ID, 0, 0, 0) и weights = vec4<f32>(1.0, 0.0, 0.0, 0.0).
//Меч будет идеально следовать за рукой, не деформируясь.

// 1. Окружение (Камера)
struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// 2. БАЗА ДАННЫХ АНИМАЦИЙ (Все кадры всех костей лежат здесь)
struct JointStorage {
    // Плоский массив матриц размера [Количество_Кадров * Количество_Костей]
    matrices: array<mat4x4<f32>>,
};
// Меняем uniform на read-only storage буфер!
@group(1) @binding(0)
var<storage, read> global_joint_storage: JointStorage;

// 3. Данные геометрии (Vertex Buffer Object)
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    // Индексы 4 костей, влияющих на вершину
    @location(3) joints: vec4<u32>,
    // Веса влияния этих костей (в сумме дают 1.0)
    @location(4) weights: vec4<f32>,
};

// 4. Данные экземпляра (Instance Buffer) - передаются для каждого зомби/моба отдельно
struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    // Самая важная переменная: какой кадр сейчас играет конкретно этот экземпляр
    @location(9) current_frame: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
};

// Константа: сколько костей в скерете ОДНОЙ модели (например, 60)
const TOTAL_BONES_PER_MESH: u32 = 60u;

@vertex
fn vs_main(vertex_input: VertexInput, instance: InstanceInput) -> VertexOutput {
    // Собираем матрицу трансформации инстанса в мире
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3
    );

    // ВЫЧИСЛЯЕМ СДВИГ ПО КАДРУ
    // Находим, где в глобальном Storage-буфере начинается текущий кадр этого инстанса
    let frame_offset = instance.current_frame * TOTAL_BONES_PER_MESH;

    // Извлекаем индексы 4 матриц для этой вершины с учетом кадра
    let idx0 = frame_offset + vertex_input.joints[0];
    let idx1 = frame_offset + vertex_input.joints[1];
    let idx2 = frame_offset + vertex_input.joints[2];
    let idx3 = frame_offset + vertex_input.joints[3];

    // КЛАССИЧЕСКИЙ СКИННИНГ (Но матрицы берутся из одной большой базы данных)
    let skin_matrix = 
        vertex_input.weights[0] * global_joint_storage.matrices[idx0] +
        vertex_input.weights[1] * global_joint_storage.matrices[idx1] +
        vertex_input.weights[2] * global_joint_storage.matrices[idx2] +
        vertex_input.weights[3] * global_joint_storage.matrices[idx3];

    // Объединяем трансформации: Скиннинг -> Позиция в мире
    let final_model_matrix = model_matrix * skin_matrix;
    let world_pos = final_model_matrix * vec4<f32>(vertex_input.position, 1.0);

    // Формируем вывод
    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_pos;
    out.world_position = world_pos.xyz;
    out.tex_coords = vertex_input.tex_coords;

    // Перевод нормалей в мировые координаты
    let normal_matrix = mat3x3<f32>(
        final_model_matrix[0].xyz,
        final_model_matrix[1].xyz,
        final_model_matrix[2].xyz
    );
    out.normal = normalize(normal_matrix * vertex_input.normal);

    return out;
}
