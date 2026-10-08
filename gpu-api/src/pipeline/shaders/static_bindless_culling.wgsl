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

struct LocalTask {
    instance_id: u32,
    global_meshlet_id: u32,
    index_count: u32,
    material_index: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var<storage, read> culling_tasks: array<CullingTask>;
@group(1) @binding(1) var<storage, read> global_instances: array<InstanceData>;
@group(1) @binding(2) var<storage, read> global_mesh_infos: array<MeshInfo>;
@group(1) @binding(3) var<storage, read> global_meshlets: array<StaticMeshletDescription>;

@group(1) @binding(4) var<storage, read_write> visible_instances: array<VisibleInstanceData>;
@group(1) @binding(5) var<storage, read_write> indirect_commands: array<DrawIndexedIndirectCommand>;
@group(1) @binding(6) var<storage, read_write> command_counter: IndirectCount;

// Локальная память рабочей группы (кэш на чипе)
var<workgroup> wg_visible_count: atomic<u32>;
var<workgroup> wg_global_offset: u32;
var<workgroup> wg_local_tasks: array<LocalTask, 256>; 

@compute @workgroup_size(64)
fn culling_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_id: u32
) {    
    // Инициализируем локальный счетчик группы
    if (local_id == 0u) {
        atomicStore(&wg_visible_count, 0u);
    }
    
    // Синхронизируем инициализацию локальной памяти группы
    workgroupBarrier();

    let task = culling_tasks[0u]; 
    let global_instance_id = task.start_object_index + global_id.x;
    
    // 1. Проверяем границы массива задач куллинга
    if (global_id.x < task.object_count) {
        let instance = global_instances[global_instance_id];
        let mesh_info = global_mesh_infos[instance.primitive_index];

        let object_visible = true; 
        
        if (object_visible) {
            for (var m_idx = 0u; m_idx < mesh_info.meshlet_count; m_idx = m_idx + 1u) {
                let global_meshlet_id = mesh_info.start_meshlet_index + m_idx;
                let meshlet = global_meshlets[global_meshlet_id];
                
                let meshlet_visible = true; 
                
                if (meshlet_visible) {
                    // Атомарно бронируем место в локальном кэше группы
                    let local_slot = atomicAdd(&wg_visible_count, 1u);
                    
                    // Предотвращаем разрушение памяти, если группа нашла суммарно > 256 мешлетов
                    if (local_slot < 256u) {
                        wg_local_tasks[local_slot].instance_id = global_instance_id;
                        wg_local_tasks[local_slot].global_meshlet_id = global_meshlet_id;
                        wg_local_tasks[local_slot].index_count = meshlet.index_count;
                        wg_local_tasks[local_slot].material_index = instance.material_index;
                    }
                }
            }
        }
    }

    // Ждем, пока ВСЕ 64 потока группы соберут данные в локальный кэш
    workgroupBarrier();

    // Обрезаем общее число строго до размера локального массива (256)
    let total_wg_visible = atomicLoad(&wg_visible_count);
    let safe_wg_visible = min(total_wg_visible, 256u);

    // Выделяем место в глобальном VRAM буфере одним общим забросом от всей группы
    if (local_id == 0u) {
        if (safe_wg_visible > 0u) {
            wg_global_offset = atomicAdd(&command_counter.count, safe_wg_visible);
        }
    }
    
    // Барьер, чтобы все потоки группы узнали полученный wg_global_offset
    workgroupBarrier();

    // Выгружаем упорядоченные команды в глобальный буфер
    for (var i = local_id; i < safe_wg_visible; i = i + 64u) {
        let local_task = wg_local_tasks[i];
        let cmd_id = wg_global_offset + i;
        
        // Жесткая проверка на максимальный лимит текущего среза глобального буфера (200)
        if (cmd_id < 200u) {
            visible_instances[cmd_id].instance_id = local_task.instance_id;
            visible_instances[cmd_id].material_index = local_task.material_index;
            visible_instances[cmd_id].meshlet_index = local_task.global_meshlet_id;

            var cmd: DrawIndexedIndirectCommand;
            cmd.index_count = local_task.index_count; 
            cmd.instance_count = 1u;               
            
            // СТРАТЕГИЧЕСКИЙ ФИКС: Для системы с Buffer Slices или байтовыми смещениями 
            // first_index всегда 0u. Смещение контролирует WebGPU на стороне Rust API!
            cmd.first_index = 0u; 
            cmd.base_vertex = 0;           
            
            // Пробрасываем текущий cmd_id в вершинный шейдер. 
            // Он будет перехвачен там через встроенную переменную @builtin(instance_index).
            cmd.first_instance = cmd_id;       

            indirect_commands[cmd_id] = cmd;
        }
    }
}
