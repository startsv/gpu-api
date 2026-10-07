struct CameraUniform {
    camera_position: vec3<f32>,
    padding: u32,    
    view_proj: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
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

struct MeshInfo {
    start_meshlet_index: u32,
    meshlet_count: u32,
    vertex_buffer_offset: u32,
    base_vertex: i32,
};

struct StaticMeshletDescription {
    aabb_min: vec3<f32>,
    vertex_offset: u32,
    aabb_max: vec3<f32>,
    index_offset: u32,
    index_count: u32,
    material_index: u32,
    pad0: u32,
    pad1: u32,
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
@group(1) @binding(2) var<storage, read> global_mesh_infos: array<MeshInfo>;
@group(1) @binding(3) var<storage, read> global_meshlets: array<StaticMeshletDescription>;

@group(1) @binding(4) var<storage, read_write> visible_instances: array<VisibleInstanceData>;
@group(1) @binding(5) var<storage, read_write> indirect_commands: array<DrawIndexedIndirectCommand>;

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
        let mesh_info = global_mesh_infos[instance.primitive_index];
                
        let m = instance.model_matrix;
                
        let obj_center = (instance.aabb_min.xyz + instance.aabb_max.xyz) * 0.5;
        let obj_extents = (instance.aabb_max.xyz - instance.aabb_min.xyz) * 0.5;
        let obj_world_center = (m * vec4<f32>(obj_center, 1.0)).xyz;
                     
        // Извлечение строк из Column-Major матрицы с сохранением знаков для OBB теста                   
        let row0 = vec3<f32>(abs(m[0].x), abs(m[1].x), abs(m[2].x));
        let row1 = vec3<f32>(abs(m[0].y), abs(m[1].y), abs(m[2].y));
        let row2 = vec3<f32>(abs(m[0].z), abs(m[1].z), abs(m[2].z));
                
        let obj_world_extents = vec3<f32>(
            dot(row0, obj_extents),
            dot(row1, obj_extents),
            dot(row2, obj_extents)
        );
        
        // --- Этап 1: Грубый куллинг всего объекта ---
        if (!is_aabb_visible(obj_world_center - obj_world_extents, obj_world_center + obj_world_extents)) {
            //continue;
        }
        
        // --- Этап 2: По-мешлетный ювелирный куллинг ---
        for (var m_idx = 0u; m_idx < mesh_info.meshlet_count; m_idx = m_idx + 1u) {
            let global_meshlet_id = mesh_info.start_meshlet_index + m_idx;
            let meshlet = global_meshlets[global_meshlet_id];
            
            let meshlet_center = (meshlet.aabb_min + meshlet.aabb_max) * 0.5;
            let meshlet_extents = (meshlet.aabb_max - meshlet.aabb_min) * 0.5;
            
            let world_meshlet_center = (m * vec4<f32>(meshlet_center, 1.0)).xyz;
            
            let world_meshlet_extents = vec3<f32>(
                dot(row0, meshlet_extents),
                dot(row1, meshlet_extents),
                dot(row2, meshlet_extents)
            );

            let meshlet_world_min = world_meshlet_center - world_meshlet_extents;
            let meshlet_world_max = world_meshlet_center + world_meshlet_extents;
            
            // ВНИМАНИЕ: Для отладки куллинга вы можете временно заменить условие на `if (true)`
            if (true) {
            //if (is_aabb_visible(meshlet_world_min, meshlet_world_max)) {
                let cmd_id = instance.base_command_id + m_idx; // Теперь берем из выделенного поля
                                
                // Перезаписываем параметры геометрии конкретного мешлета
                indirect_commands[cmd_id].index_count = meshlet.index_count;
                indirect_commands[cmd_id].first_index = meshlet.index_offset;
                
                // ВАЖНО ДЛЯ VERTEX PULLING: Обнуляем аппаратный base_vertex.
                // Смещение меша `mesh_info.vertex_buffer_offset` мы применим вручную во Vertex Shader.
                indirect_commands[cmd_id].base_vertex = 0;
                
                // Атомарно занимаем инстанс-слот для данного мешлета
                let local_slot = atomicAdd(&indirect_commands[cmd_id].instance_count, 1u);
                                
                // Читаем базовое смещение в глобальном буфере видимости visible_instances.
                // Оно настраивается на CPU в `generate_static_test_data` индивидуально для каждой команды.
                let base_offset = indirect_commands[cmd_id].first_instance;
                let write_index = base_offset + local_slot;
                                            
                // Записываем данные для рендеринга
                visible_instances[write_index].instance_id = global_instance_id;
                visible_instances[write_index].material_index = instance.material_index;
            }
        }
    }
}

