use glam::{Mat4, Vec3};
use gpu_api::pipeline::{aa_line_pipeline::AaLineInstance, static_bindless_pipeline::MeshletData};
use gpu_api_relay::model_bindless_data::{CullingTask, DrawIndexedIndirectCommand, InstanceData, StaticVertex, SurfaceData, SurfaceMeshletDescription, SurfaceVertex, Vertex}; // Используем вашу математическую библиотеку (скорее всего glam)

// Хелпер для хранения метаданных уникального меша до создания инстансов
pub struct GeneratedMeshAsset {
    pub start_meshlet_index: u32,
    pub meshlet_count: u32,
}

/// Спавнит заданное количество инстансов, случайно выбирая для них одну из N уникальных моделей,
/// и формирует финальные буферы для отправки на GPU.
pub fn build_test_scene(
    mesh_assets: &[GeneratedMeshAsset],
    base_meshlets: &[MeshletData],
    instances_count: u32,
) -> (Vec<InstanceData>, Vec<CullingTask>, Vec<MeshletData>) {
    let mut instances = Vec::new();
    let mut culling_tasks = Vec::new();
    let mut scene_global_meshlets = Vec::new();

    // Зададим шаг сетки для расстановки инстансов в пространстве
    let grid_size = (instances_count as f32).sqrt().ceil() as i32;
    let spacing = 4.0f32;

    for i in 0..instances_count {
        let instance_id = i;
        
        // Выбираем для этого инстанса один из 10 уникальных мешей по кругу (или псевдорандомно)
        let asset_idx = (i % mesh_assets.len() as u32) as usize;
        let asset = &mesh_assets[asset_idx];

        // Рассчитываем мировую позицию инстанса на сетке XZ
        let x_pos = (i as i32 % grid_size) as f32 * spacing;
        let z_pos = (i as i32 / grid_size) as f32 * spacing;
        
        let model_matrix = Mat4::from_translation(Vec3::new(x_pos, 0.0, z_pos));

        // 1. Создаем InstanceData объекта
        instances.push(InstanceData {
            model_matrix,
            is_animated: 0,
            node_index: 0,      // В тестах нода идентична инстансу (identity transform)
            joints_offset: 0,
            material_index: asset_idx as u32, // Зададим уникальный материал для каждой модели
            primitive_index: asset_idx as u32,
            _pad0: 0, _pad1: 0, _pad2: 0,
            aabb_min: [-1.0, -1.0, -1.0], // Грубый AABB инстанса
            _pad_aabb1: 0,
            aabb_max: [1.0, 1.0, 1.0],
            _pad_aabb2: 0,
        });

        // 2. Дублируем мешлеты этой модели для текущего инстанса
        // Так как куллинг идет поштучно по мешлетам, каждый инстанцированный мешлет
        // должен знать свой уникальный `instance_id`, чтобы применить правильную матрицу трансформации.
        let scene_start_meshlet_index = scene_global_meshlets.len() as u32;

        for m_idx in 0..asset.meshlet_count {
            let base_meshlet = base_meshlets[(asset.start_meshlet_index + m_idx) as usize];
            
            let mut instance_meshlet = base_meshlet;
            instance_meshlet.instance_id = instance_id; // Важнейшая связь: мешлет -> инстанс
            
            scene_global_meshlets.push(instance_meshlet);
        }

        // 3. Создаем CullingTask для этого инстанса
        // Поток GPU возьмет эту задачу и обработает пачку мешлетов конкретно этого инстанса
        culling_tasks.push(CullingTask {
            start_object_index: scene_start_meshlet_index,
            object_count: asset.meshlet_count,
            lod_level: 0,
            _padding: 0,
        });
    }

    (instances, culling_tasks, scene_global_meshlets)
}


/// Генерирует N уникальных моделей (сеток) в один набор буферов
pub fn generate_unique_mesh_assets(
    object_count: u32,
) -> (Vec<Vertex>, Vec<u32>, Vec<MeshletData>, Vec<GeneratedMeshAsset>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut global_meshlets = Vec::new();
    let mut mesh_assets = Vec::new();

    for obj_id in 0..object_count {
        let start_meshlet_index = global_meshlets.len() as u32;
        
        // Делаем каждый объект уникальным: меняем размер и плотность сетки в зависимости от obj_id
        let segments = 4 + (obj_id * 2); // От 4x4 до более плотных сеток
        let size = 1.0 + (obj_id as f32 * 0.5); // Разные физические размеры объектов
        let base_vertex_offset = vertices.len() as u32;

        let mut current_meshlet_indices = Vec::new();
        let triangles_per_meshlet = 32;

        // Генерация вершин уникального объекта
        for z in 0..=segments {
            for x in 0..=segments {
                let x_frac = x as f32 / segments as f32;
                let z_frac = z as f32 / segments as f32;
                
                let pos = Vec3::new((x_frac - 0.5) * size, 0.0, (z_frac - 0.5) * size);

                vertices.push(Vertex {
                    position: pos.to_array(),
                    uv: [x_frac, z_frac],
                    normal: [0.0, 1.0, 0.0],
                    tangent: [1.0, 0.0, 0.0],
                    bitangent: [0.0, 0.0, 1.0],
                    joints: [0; 4],
                    weights: [1.0, 0.0, 0.0, 0.0],
                });
            }
        }

        // Генерация треугольников объекта
        for z in 0..segments {
            for x in 0..segments {
                let row_length = segments + 1;
                let i0 = base_vertex_offset + (z * row_length + x);
                let i1 = base_vertex_offset + (z * row_length + (x + 1));
                let i2 = base_vertex_offset + ((z + 1) * row_length + x);
                let i3 = base_vertex_offset + ((z + 1) * row_length + (x + 1));

                current_meshlet_indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);

                if current_meshlet_indices.len() / 3 >= triangles_per_meshlet {
                    flush_test_meshlet(&current_meshlet_indices, &vertices, &mut indices, &mut global_meshlets);
                    current_meshlet_indices.clear();
                }
            }
        }

        if !current_meshlet_indices.is_empty() {
            flush_test_meshlet(&current_meshlet_indices, &vertices, &mut indices, &mut global_meshlets);
        }

        let meshlet_count = global_meshlets.len() as u32 - start_meshlet_index;
        
        mesh_assets.push(GeneratedMeshAsset {
            start_meshlet_index,
            meshlet_count,
        });
    }

    (vertices, indices, global_meshlets, mesh_assets)
}

fn flush_test_meshlet(
    meshlet_indices: &[u32],
    vertices: &[Vertex],
    mega_index_buffer: &mut Vec<u32>,
    global_meshlets: &mut Vec<MeshletData>,
) {
    let index_offset = mega_index_buffer.len() as u32;
    mega_index_buffer.extend_from_slice(meshlet_indices);

    // Считаем локальный AABB и сферу куллинга мешлета
    let mut min_bound = Vec3::splat(f32::INFINITY);
    let mut max_bound = Vec3::splat(f32::NEG_INFINITY);
    for &idx in meshlet_indices {
        let pos = Vec3::from_array(vertices[idx as usize].position);
        min_bound = min_bound.min(pos);
        max_bound = max_bound.max(pos);
    }
    let bounding_center = (min_bound + max_bound) * 0.5;

    let mut bounding_radius = 0.0f32;
    for &idx in meshlet_indices {
        let pos = Vec3::from_array(vertices[idx as usize].position);
        bounding_radius = bounding_radius.max(pos.distance(bounding_center));
    }

    global_meshlets.push(MeshletData {
        vertex_offset: 0, 
        vertex_count: 0,
        index_offset,
        triangle_count: (meshlet_indices.len() / 3) as u32,
        instance_id: 0, // Будет динамически переназначено при создании инстансов!
        bounding_center_x: bounding_center.x,
        bounding_center_y: bounding_center.y,
        bounding_center_z: bounding_center.z,
        bounding_radius,
        _pad0: 0, _pad1: 0, _pad2: 0,
    });
}


pub fn generate_grid(
    lines: &mut Vec<AaLineInstance>, 
    slices: i32, 
    step: f32, 
    width: f32
) {
    let half_size = slices as f32 * step;
        
    let default_color = [0.3, 0.3, 0.3, 1.0];
    let axis_x_color  = [0.8, 0.2, 0.2, 1.0];
    let axis_z_color  = [0.2, 0.2, 0.8, 1.0];

    for i in -slices..=slices {
        let current_coord = i as f32 * step;
        
        let color_z = if i == 0 { 
            axis_z_color
        } else { 
            default_color 
        };
        
        lines.push(AaLineInstance {
            color: color_z,
            width,
            start_pos: [current_coord, 0.0, -half_size],
            end_pos:   [current_coord, 0.0, half_size],
        });
        
        let color_x = if i == 0 { 
            axis_x_color
        } else { 
            default_color 
        };

        lines.push(AaLineInstance {
            color: color_x,
            width,
            start_pos: [-half_size, 0.0, current_coord],
            end_pos:   [half_size, 0.0, current_coord],
        });
    }
}

const SURFACE_MESHLET_SIZE: u32 = 8;
const SURFACE_VERTICES_PER_MESHLET: u32 = SURFACE_MESHLET_SIZE * SURFACE_MESHLET_SIZE;

pub fn generate_test_surface(
    width_in_meshlets: u32,
    depth_in_meshlets: u32,
    vertex_spacing: f32,
) -> SurfaceData {
    let mut surface_data = SurfaceData::new();    
    
    let total_meshlets = width_in_meshlets * depth_in_meshlets;
        
    surface_data.vertices.reserve((total_meshlets * SURFACE_VERTICES_PER_MESHLET) as usize);
    
    let indices_per_meshlet = (SURFACE_MESHLET_SIZE - 1) * (SURFACE_MESHLET_SIZE - 1) * 6;
    surface_data.indices.reserve((total_meshlets * indices_per_meshlet) as usize);
    surface_data.meshlets.reserve(total_meshlets as usize);
    
    let meshlet_world_size = (SURFACE_MESHLET_SIZE - 1) as f32 * vertex_spacing;

    let mut current_vertex_offset = 0;
    let mut current_index_offset = 0;

    for mz in 0..depth_in_meshlets {
        for mx in 0..width_in_meshlets {
            let mut aabb_min = [f32::MAX, f32::MAX, f32::MAX];
            let mut aabb_max = [f32::MIN, f32::MIN, f32::MIN];

            let meshlet_x_origin = mx as f32 * meshlet_world_size;
            let meshlet_z_origin = mz as f32 * meshlet_world_size;
            
            for lz in 0..SURFACE_MESHLET_SIZE {
                for lx in 0..SURFACE_MESHLET_SIZE {
                    let world_x = meshlet_x_origin + (lx as f32 * vertex_spacing);
                    let world_z = meshlet_z_origin + (lz as f32 * vertex_spacing);
                    
                    let world_y = (world_x * 0.1).sin() * 5.0 + (world_z * 0.05).cos() * 8.0;
                    
                    let nx = -0.1 * (world_x * 0.1).cos() * 5.0;
                    let nz = -0.05 * -(world_z * 0.05).sin() * 8.0;
                    let mut normal = [nx, 1.0, nz];
                    
                    let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
                    normal[0] /= len;
                    normal[1] /= len;
                    normal[2] /= len;

                    let vertex = SurfaceVertex {
                        position: [world_x, world_y, world_z],
                        _pad0: 0.0,
                        normal,
                        _pad1: 0.0,
                    };

                    surface_data.vertices.push(vertex);
                    
                    aabb_min[0] = aabb_min[0].min(world_x);
                    aabb_min[1] = aabb_min[1].min(world_y);
                    aabb_min[2] = aabb_min[2].min(world_z);

                    aabb_max[0] = aabb_max[0].max(world_x);
                    aabb_max[1] = aabb_max[1].max(world_y);
                    aabb_max[2] = aabb_max[2].max(world_z);
                }
            }
            
            for lz in 0..(SURFACE_MESHLET_SIZE - 1) {
                for lx in 0..(SURFACE_MESHLET_SIZE - 1) {
                    
                    let i0 = lz * SURFACE_MESHLET_SIZE + lx;
                    let i1 = lz * SURFACE_MESHLET_SIZE + (lx + 1);
                    let i2 = (lz + 1) * SURFACE_MESHLET_SIZE + lx;
                    let i3 = (lz + 1) * SURFACE_MESHLET_SIZE + (lx + 1);
                    
                    surface_data.indices.push(current_vertex_offset + i0);
                    surface_data.indices.push(current_vertex_offset + i2);
                    surface_data.indices.push(current_vertex_offset + i1);

                    surface_data.indices.push(current_vertex_offset + i1);
                    surface_data.indices.push(current_vertex_offset + i2);
                    surface_data.indices.push(current_vertex_offset + i3);
                }
            }
            
            let meshlet_index_count = (SURFACE_MESHLET_SIZE - 1) * (SURFACE_MESHLET_SIZE - 1) * 6;
            
            surface_data.meshlets.push(SurfaceMeshletDescription {
                aabb_min,
                vertex_offset: current_vertex_offset,
                aabb_max,
                index_offset: current_index_offset,
                index_count: meshlet_index_count,
                material_index: 0,
                pad0: 0,
                pad1: 0,
            });

            current_vertex_offset += SURFACE_VERTICES_PER_MESHLET;
            current_index_offset += meshlet_index_count;
        }
    }
    
    surface_data.indirect_commands.push(DrawIndexedIndirectCommand {
        index_count: 0,
        instance_count: 0,
        first_index: 0,
        base_vertex: 0,
        first_instance: 0,
    });

    surface_data
}
