use glam::{Mat4, Vec3};
use gpu_api::pipeline::{aa_line_pipeline::AaLineInstance, static_bindless_pipeline::MeshletData};
use gpu_api_relay::model_bindless_data::{CullingTask, DrawIndexedIndirectCommand, InstanceData, MaterialFactors, StaticVertex, SurfaceData, SurfaceMeshletDescription, SurfaceVertex, Vertex}; // Используем вашу математическую библиотеку (скорее всего glam)

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

/// Генерирует N кубов в пространстве. Каждый куб разбивается на 6 мешлетов (по одному на грань).
pub fn generate_n_cubes_scene(
    cube_count: u32,
) -> (Vec<Vertex>, Vec<u32>, Vec<MaterialFactors>, Vec<DrawIndexedIndirectCommand>, Vec<MeshletData>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut meshlets = Vec::new();
    let mut indirect_commands = Vec::new();
    let mut material_factors = Vec::new();

    // Размеры одного куба
    let half_size = 0.5f32;

    // Шаг сетки для расстановки кубов в пространстве
    let grid_size = (cube_count as f32).sqrt().ceil() as i32;
    let spacing = 3.0f32;

    // Определим вершины и индексы для 6 граней куба в локальных координатах.
    // Каждая грань — это 4 вершины и 2 треугольника (6 индексов).
    let face_normals = [
        Vec3::new(0.0, 0.0, 1.0),  // Front
        Vec3::new(0.0, 0.0, -1.0), // Back
        Vec3::new(-1.0, 0.0, 0.0), // Left
        Vec3::new(1.0, 0.0, 0.0),  // Right
        Vec3::new(0.0, 1.0, 0.0),  // Top
        Vec3::new(0.0, -1.0, 0.0), // Bottom
    ];

    // Локальные направления для построения плоскостей граней
    let face_tangents = [
        Vec3::new(1.0, 0.0, 0.0),  // Front
        Vec3::new(-1.0, 0.0, 0.0), // Back
        Vec3::new(0.0, 0.0, 1.0),  // Left
        Vec3::new(0.0, 0.0, -1.0), // Right
        Vec3::new(1.0, 0.0, 0.0),  // Top
        Vec3::new(1.0, 0.0, 0.0),  // Bottom
    ];

    for cube_id in 0..cube_count {
        // Добавляем один тестовый материал на куб
        material_factors.push(MaterialFactors {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],            
            emissive_factor: [0.0, 0.0, 0.0],
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            padding: [0, 0, 0],
        });

        // Считаем позицию куба на XZ сетке в мире
        let x_pos = (cube_id as i32 % grid_size) as f32 * spacing;
        let z_pos = (cube_id as i32 / grid_size) as f32 * spacing;
        let world_offset = Vec3::new(x_pos, 0.0, z_pos);

        // Генерируем 6 граней куба. Каждая грань станет отдельным мешлетом.
        for face_id in 0..6 {
            let normal = face_normals[face_id];
            let tangent = face_tangents[face_id];
            let bitangent = normal.cross(tangent);

            let base_vertex_idx = vertices.len() as u32;
            let start_index_offset = indices.len() as u32;

            // Центр конкретной грани куба
            let face_center = normal * half_size + world_offset;

            // Генерируем 4 вершины для текущей грани куба
            let local_quad_verts = [
                face_center - tangent * half_size - bitangent * half_size,
                face_center + tangent * half_size - bitangent * half_size,
                face_center + tangent * half_size + bitangent * half_size,
                face_center - tangent * half_size + bitangent * half_size,
            ];

            let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

            for i in 0..4 {
                vertices.push(Vertex {
                    position: local_quad_verts[i].to_array(),
                    uv: uvs[i],
                    normal: normal.to_array(),
                    tangent: tangent.to_array(),
                    bitangent: bitangent.to_array(),
                    joints: [0; 4],
                    weights: [1.0, 0.0, 0.0, 0.0],
                });
            }

            // Индексы для двух треугольников грани (Quad)
            indices.push(base_vertex_idx + 0);
            indices.push(base_vertex_idx + 1);
            indices.push(base_vertex_idx + 2);

            indices.push(base_vertex_idx + 0);
            indices.push(base_vertex_idx + 2);
            indices.push(base_vertex_idx + 3);

            let triangle_count = 2; // 2 треугольника на мешлет-грань

            // Вычисляем Bounding Sphere для этого мешлета (грани)
            // Радиус сферы, охватывающей квадратную грань куба
            let bounding_radius = (half_size * half_size + half_size * half_size).sqrt();

            // В нашей статической схеме instance_id мешлета жестко указывает на ID куба,
            // чтобы шейдер куллинга мог извлечь нужную model_matrix инстанса.
            let meshlet = MeshletData {
                vertex_offset: 0, // Работаем через единый глобальный буфер
                vertex_count: 4,
                index_offset: start_index_offset,
                triangle_count,
                
                instance_id: cube_id, // Связь мешлета с объектом
                bounding_center_x: face_center.x - world_offset.x, // В локальных координатах инстанса
                bounding_center_y: face_center.y - world_offset.y,
                bounding_center_z: face_center.z - world_offset.z,
                bounding_radius,
                
                _pad0: 0, _pad1: 0, _pad2: 0,
            };

            meshlets.push(meshlet);

            // Сразу же генерируем парную Draw-команду для этого мешлета
            indirect_commands.push(DrawIndexedIndirectCommand {
                index_count: triangle_count * 3,
                instance_count: 0, // Изначально выключен, шейдер куллинга включит
                first_index: start_index_offset,
                base_vertex: 0,    // Индексы уже глобальные внутри mega_index_buffer
                first_instance: 0,
            });
        }
    }

    (vertices, indices, material_factors, indirect_commands, meshlets)
}
