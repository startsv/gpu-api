use std::io::{Cursor, Read, Write};
use glam::{Mat4, Vec2, Vec3};
use image::{DynamicImage, ImageReader, Rgb, Rgba};
use log::*;
use gltf::{image::Format, iter, mesh::util::{ReadIndices, ReadJoints, ReadTexCoords, ReadWeights}};
use gpu_api_dto::{AlphaMode, Animation, AnimationChannel, AnimationProperty, ImageFormat, Interpolation, Joint, MaterialData, MeshData, ModelData, Node, PrimitiveData, Skin, TextureCompressionFormat, TextureData, TextureType};
use crate::material::create_material_data;

pub mod material;

pub fn load(model_name: &str, model_path: &str, add_pixes: bool, add_images: bool, attached_nodes_indices: Vec<usize>, is_animated: bool) -> (ModelData, Option<Vec<DynamicImage>>) {
    info!("Loading model {} from path {}", model_name, model_path);    
    let (document, buffers, images) = gltf::import(model_path).expect("Model import failed");
    
    info!("Found {} scenes", document.scenes().count());

    for scene in document.scenes() {
        info!("Found scene {:?}, index {}", scene.name(), scene.index());        
    }

    info!("Found {} nodes", document.nodes().count());    

    let mut nodes = vec![];

    /// Computes topological ordering and children->parent map.
    fn node_indices_topological_sort(nodes: iter::Nodes) -> (Vec<usize>, std::collections::BTreeMap<usize, usize>) {
        // NOTE: The algorithm uses BTreeMaps to guarantee consistent ordering.

        // Maps parent to list of children
        let mut children = std::collections::BTreeMap::<usize, Vec<usize>>::new();
        for node in nodes {
            children.insert(node.index(), node.children().map(|n| n.index()).collect());
        }

        // Maps child to parent
        let parents: std::collections::BTreeMap<usize, usize> =
            children.iter().flat_map(|(parent, children)| children.iter().map(|ch| (*ch, *parent))).collect();

        // Initialize the BFS queue with nodes that don't have any parent (i.e. roots)
        let mut queue: std::collections::VecDeque<usize> = children.keys().filter(|n| parents.get(n).is_none()).cloned().collect();

        let mut topological_sort = Vec::<usize>::new();

        while let Some(n) = queue.pop_front() {
            topological_sort.push(n);
            for ch in &children[&n] {
                queue.push_back(*ch);
            }
        }

        (topological_sort, parents)
    }

    let (node_topological_sorting, node_map) = node_indices_topological_sort(document.nodes());
    info!("{:?}", node_topological_sorting);
    info!("{:?}", node_map);

    for node in document.nodes() {        
        info!("Found node {:?}, index {} skin {:?}, mesh {:?}",
            node.name(), 
            node.index(),             
            node.skin().map(|v| v.index()),
            node.mesh().map(|v| v.name())
        );

        let mut child_list = node.index().to_string();        
        child_list.push_str(" node parent to :");

        for q in node.children() {            
            //info!("Found child node, index {}", q.index());
            child_list.push_str(" ");
            child_list.push_str(&q.index().to_string());
        }
        
        info!("{}", child_list);

        let local_transform_matrix = node.transform().matrix();

        let (translation, rotation, scale) = node.transform().decomposed();        

        nodes.push(Node {
            index: node.index(),
            mesh_index: node.mesh().map(|v| v.index()),
            children_nodes: node.children().map(|v| v.index()).collect(),
            name: node.name().map(|v| v.to_owned()),
            skin_index: node.skin().map(|v| v.index()),
            translation, 
            rotation, 
            scale,
            local_transform_matrix
        });
    }    

    let mut skins = vec![];

    for skin in document.skins() {
        info!("Found skin {:?}, index {}", skin.name(), skin.index());

        let num_joints = skin.joints().count();
        let reader = skin.reader(|b| Some(&buffers[b.index()][..b.length()]));   
        
        let inv_b_mats: Vec<[[f32; 4]; 4]> = match reader.read_inverse_bind_matrices() {            
            Some(inv_b_mats) => {
                inv_b_mats.collect()
            }
            None => {
                let mut inv_b_mats = vec![];

                let q = [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ];
                
                for _ in 0..num_joints {
                    inv_b_mats.push(q);
                }

                inv_b_mats
            }            
        };        

        let mut joints = vec![];

        let mut index = 0;        

        for joint in skin.joints() {
            info!("Found joint, index {}, name {:?}", joint.index(), joint.name().map(|v| v.to_owned()));

            joints.push(Joint {
                node_index: joint.index(),
                node_name: joint.name().map(|v| v.to_owned())
            });            

            index = index + 1;
        }

        info!("Joints total: {}", joints.len());

        skins.push(Skin {
                name: skin.name().map(|v| v.to_owned()),
                inverse_bind_matrices: inv_b_mats, 
                joints
            }
        );
    }

    info!("Skins total: {}", skins.len());

    let mut node_local_transforms: Vec<Mat4> = nodes.iter().map(|v| Mat4::from_cols_array_2d(&v.local_transform_matrix)).collect();

    for node_index in node_topological_sorting.iter() {
        match node_map.get(node_index) {
            Some(parent_index) => {
                let parent_transform = node_local_transforms[*parent_index];
                let current_transform = &mut node_local_transforms[*node_index];
                
                *current_transform = parent_transform * *current_transform;
            }
            None => {}
        }
    }

    let mut meshes = vec![];    

    for mesh in document.meshes() {
        info!("Found Mesh {:?}, index {}, has weights: {}", mesh.name(), mesh.index(), mesh.weights().is_some());

        let node_transform = match nodes.iter().find(|v| v.mesh_index == Some(mesh.index())) {
            Some(node) if attached_nodes_indices.contains(&node.index) => Some(node_local_transforms[node.index]),
            _ => None
        };

        let mut primitives = vec![];        

        for primitive in mesh.primitives() {            
            info!("Found primitive {}, mode {:?}, material {:?} {:?}", primitive.index(), primitive.mode(), primitive.material().index(), primitive.material().name());
            //info!("{:#?}", primitive.attributes());            

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));            

            let mut positions = vec![];
            
            match reader.read_positions() {
                Some(iter) => {
                    for vertex_position in iter {
                        //info!("{:?}", vertex_position);
                        positions.push(vertex_position);
                    }
                }
                None => {}
            }

            let mut indices = vec![];

            match reader.read_indices() {
                Some(iter) => {
                    // Different type indexes should not intersect.
                    match iter {
                        ReadIndices::U8(dog) => {
                            for q in dog {
                                indices.push(q as u32);
                            }
                        }
                        ReadIndices::U16(dog) => {
                            for q in dog {
                                indices.push(q as u32);
                            }
                        }
                        ReadIndices::U32(dog) => {
                            for q in dog {
                                indices.push(q as u32);
                            }
                        }
                    }
                }
                None => {}
            }

            let mut normals = vec![];

            match reader.read_normals() {
                Some(iter) => {
                    for q in iter {                        
                        normals.push(q);
                    }
                }
                None => {}
            }

            let mut tangents = vec![];            

            //bitangent = cross(normal.xyz, tangent.xyz) * tangent.w.

            match reader.read_tangents() {
                Some(iter) => {}
                None => {}
            }

            match reader.read_colors(0) {
                Some(iter) => {}
                None => {}
            }

            let mut joints = vec![];

            match reader.read_joints(0) {
                Some(read_joints) => {
                    match read_joints {
                        ReadJoints::U8(iter) => {                            
                            for joint in iter {
                                joints.push([joint[0] as u32, joint[1] as u32, joint[2] as u32, joint[3] as u32]);
                            }
                        }
                        ReadJoints::U16(iter) => {
                            for joint in iter {
                                joints.push([joint[0] as u32, joint[1] as u32, joint[2] as u32, joint[3] as u32]);
                            }
                        }
                    }
                }
                None => {}
            }

            let mut texture_coordinates = vec![];
            
            match reader.read_tex_coords(0) {
                Some(coords) => {
                    match coords {
                        ReadTexCoords::F32(iter) => {
                            for q in iter {                                
                                texture_coordinates.push(q);
                            }
                        }
                        ReadTexCoords::U8(iter) => {
                            for q in iter {
                                warn!("set 0 u8 {:?}", q);
                            }
                        }
                        ReadTexCoords::U16(iter) => {
                            for q in iter {
                                warn!("set 0 u16 {:?}", q);
                            }
                        }
                    }
                    
                }
                None => {}
            }            

            match reader.read_tex_coords(1) {
                Some(coords) => {
                    match coords {
                        ReadTexCoords::F32(iter) => {
                            for q in iter {
                                warn!("set 1 f32 {:?}", q);
                            }
                        }
                        ReadTexCoords::U8(iter) => {
                            for q in iter {
                                warn!("set 1 u8 {:?}", q);
                            }
                        }
                        ReadTexCoords::U16(iter) => {
                            for q in iter {
                                warn!("set 1 u16 {:?}", q);
                            }
                        }
                    }
                    
                }
                None => {}
            }

            let mut weights = vec![];
            
            match reader.read_weights(0) {
                Some(read_weights) => {
                    match read_weights {
                        ReadWeights::U8(iter) => {
                            for q in iter {
                                info!("Found u8 weight");
                            }
                        }
                        ReadWeights::U16(iter) => {
                            for q in iter {
                                info!("Found u16 weight");
                            }
                        }
                        ReadWeights::F32(iter) => {                            
                            for weight in iter {
                                weights.push(weight);
                            }
                        }
                    }
                }
                None => {}
                None => {}
            }

            reader.read_morph_targets();            

            info!("{}, mesh {:?} positions total: {}", model_name, mesh.name(), positions.len());
            info!("{}, mesh {:?} indices total: {}", model_name, mesh.name(), indices.len());
            info!("{}, mesh {:?} normals total: {}", model_name, mesh.name(), normals.len());
            info!("{}, mesh {:?} tangents total: {}", model_name, mesh.name(), tangents.len());
            info!("{}, mesh {:?} joints total: {}", model_name, mesh.name(), joints.len());
            info!("{}, mesh {:?} weights total: {}", model_name, mesh.name(), weights.len());
            info!("{}, mesh {:?} texture coordinates total: {}", model_name, mesh.name(), texture_coordinates.len());

            let mut bitangents = vec![];

            if true {
                /*
                for _ in 0..positions.len() {
                    tangents.push([0.0, 0.0, 0.0, 0.0]);
                }
                */

                struct ModelVertex{
                    position: [f32; 3],
                    tex_coords: [f32; 2],                    
                    tangent: [f32; 3],
                    bitangent: [f32; 3]
                }

                let mut vertices = (0..positions.len())
                .map(|i| ModelVertex {
                    position: positions[i],
                    tex_coords: texture_coordinates[i],                    
                    // We'll calculate these later
                    tangent: [0.0; 3],
                    bitangent: [0.0; 3]
                })
                .collect::<Vec<_>>();
            
                let mut triangles_included = vec![0; vertices.len()];

                // Calculate tangents and bitangets. We're going to
                // use the triangles, so we need to loop through the
                // indices in chunks of 3
                for c in indices.chunks(3) {
                    let v0 = &vertices[c[0] as usize];
                    let v1 = &vertices[c[1] as usize];
                    let v2 = &vertices[c[2] as usize];

                    let pos0 = Vec3::from_array(v0.position);
                    let pos1 = Vec3::from_array(v1.position);
                    let pos2 = Vec3::from_array(v2.position);

                    let uv0 = Vec2::from_array(v0.tex_coords);
                    let uv1 = Vec2::from_array(v1.tex_coords);
                    let uv2 = Vec2::from_array(v2.tex_coords);

                    // Calculate the edges of the triangle
                    let delta_pos1 = pos1 - pos0;
                    let delta_pos2 = pos2 - pos0;

                    // This will give us a direction to calculate the
                    // tangent and bitangent
                    let delta_uv1 = uv1 - uv0;
                    let delta_uv2 = uv2 - uv0;

                    // Solving the following system of equations will
                    // give us the tangent and bitangent.
                    //     delta_pos1 = delta_uv1.x * T + delta_u.y * B
                    //     delta_pos2 = delta_uv2.x * T + delta_uv2.y * B
                    // Luckily, the place I found this equation provided
                    // the solution!
                    let r = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
                    let tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;
                    // We flip the bitangent to enable right-handed normal
                    // maps with wgpu texture coordinate system
                    let bitangent = (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * -r;

                    // We'll use the same tangent/bitangent for each vertex in the triangle
                    vertices[c[0] as usize].tangent =
                        (tangent + Vec3::from_array(vertices[c[0] as usize].tangent)).to_array();
                    vertices[c[1] as usize].tangent =
                        (tangent + Vec3::from_array(vertices[c[1] as usize].tangent)).to_array();
                    vertices[c[2] as usize].tangent =
                        (tangent + Vec3::from_array(vertices[c[2] as usize].tangent)).to_array();
                    vertices[c[0] as usize].bitangent =
                        (bitangent + Vec3::from_array(vertices[c[0] as usize].bitangent)).to_array();
                    vertices[c[1] as usize].bitangent =
                        (bitangent + Vec3::from_array(vertices[c[1] as usize].bitangent)).to_array();
                    vertices[c[2] as usize].bitangent =
                        (bitangent + Vec3::from_array(vertices[c[2] as usize].bitangent)).to_array();

                    // Used to average the tangents/bitangents
                    triangles_included[c[0] as usize] += 1;
                    triangles_included[c[1] as usize] += 1;
                    triangles_included[c[2] as usize] += 1;
                }

                // Average the tangents/bitangents
                for (i, n) in triangles_included.into_iter().enumerate() {
                    let denom = 1.0 / n as f32;
                    let v = &mut vertices[i];
                    v.tangent = (Vec3::from_array(v.tangent) * denom).to_array();
                    v.bitangent = (Vec3::from_array(v.bitangent) * denom).to_array();
                }

                tangents = vertices.iter().map(|r| r.tangent).collect();
                bitangents = vertices.iter().map(|r| r.bitangent).collect();

                info!("{}, mesh {:?} calculated tangents total: {}", model_name, mesh.name(), tangents.len());
                info!("{}, mesh {:?} calculated bitangents total: {}", model_name, mesh.name(), bitangents.len());
            }

            primitives.push(PrimitiveData {
                positions,
                indices,
                normals,
                tangents,
                bitangents,
                joints,
                weights,
                material_index: primitive.material().index(),
                texture_coordinates                
            });
        }

        meshes.push(MeshData {
            index: mesh.index(),
            node_transform: node_transform.map(|nt| nt.to_cols_array_2d()),
            primitives
        });        
    }

    info!("{} meshes total: {}", model_name, meshes.len());    
    info!("{} images total: {}", model_name, images.len());    

    

    let mut materials = vec![];
    let mut loaded_images = vec![];
    let mut material_index = 0;

    for material in document.materials() {
        let material_data = create_material_data(model_name, material_index, &buffers, &images, add_images, add_pixes, &mut loaded_images, &material);
        
        materials.push(material_data);

        material_index = material_index + 1;
    }

    info!("Materials total: {}", materials.len());

    let mut animations = vec![];

    for animation in document.animations() {
        info!("Found animation, channels {}, samplers {}", animation.channels().count(), animation.samplers().count());

        let mut channels = vec![];
        
        for channel in animation.channels() {
            //info!("Found animation channel for node {:?} with {:?}, index {}", channel.target().node().name(), channel.target().property(), channel.target().node().index());                        

            let animation_property = match channel.target().property() {
                gltf::animation::Property::Translation => AnimationProperty::Translation,
                gltf::animation::Property::Rotation => AnimationProperty::Rotation,
                gltf::animation::Property::Scale => AnimationProperty::Scale,
                gltf::animation::Property::MorphTargetWeights => AnimationProperty::MorphTargetWeights
            };


            let interpolation = match channel.sampler().interpolation() {
                gltf::animation::Interpolation::Linear => Interpolation::Linear,
                gltf::animation::Interpolation::Step => Interpolation::Step,
                gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline
            };

            let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));

            let timestamps = match reader.read_inputs() {
                Some(inputs) => {
                    match inputs {
                        gltf::accessor::Iter::Standard(times) => {
                            let times: Vec<f32> = times.collect();                            
                            times
                        }
                        gltf::accessor::Iter::Sparse(_) => {
                            info!("Sparse keyframes not supported");
                            vec![]
                        }
                    }
                }
                None => {
                    info!("Empty animations timestamps");
                    vec![]
                }
            };

            let mut translations = vec![];
            let mut rotations = vec![];
            let mut scales = vec![];
            let mut weight_morphs = vec![];

            match reader.read_outputs() {
                Some(outputs) => {
                    match outputs {
                        gltf::animation::util::ReadOutputs::Translations(translation_iterator) => {
                            for value in translation_iterator {
                                translations.push(value);                                
                            }
                        }
                        gltf::animation::util::ReadOutputs::Rotations(rotation) => {
                            match rotation {
                                gltf::animation::util::Rotations::I8(_rotation_iterator) => {
                                    info!("I8 rotations not supported");
                                }
                                gltf::animation::util::Rotations::U8(_rotation_iterator) => {
                                    info!("U8 rotations not supported");
                                }
                                gltf::animation::util::Rotations::I16(_rotation_iterator) => {
                                    info!("I16 rotations not supported");
                                }
                                gltf::animation::util::Rotations::U16(_rotation_iterator) => {
                                    info!("U16 rotations not supported");
                                }
                                gltf::animation::util::Rotations::F32(rotation_iterator) => {
                                    for value in rotation_iterator {
                                        rotations.push(value);                                
                                    }
                                }
                            }                            
                        }
                        gltf::animation::util::ReadOutputs::Scales(scale_iterator) => {                            
                            for value in scale_iterator {
                                scales.push(value);                                
                            }
                        }
                        gltf::animation::util::ReadOutputs::MorphTargetWeights(morph_target_weights) => {
                            match morph_target_weights {
                                gltf::animation::util::MorphTargetWeights::I8(_morph_iterator) => {
                                    info!("I8 rotations not supported");
                                }
                                gltf::animation::util::MorphTargetWeights::U8(_morph_iterator) => {
                                    info!("U8 rotations not supported");
                                }
                                gltf::animation::util::MorphTargetWeights::I16(_morph_iterator) => {
                                    info!("I16 rotations not supported");
                                }
                                gltf::animation::util::MorphTargetWeights::U16(_morph_iterator) => {
                                    info!("U16 rotations not supported");
                                }
                                gltf::animation::util::MorphTargetWeights::F32(morph_iterator) => {
                                    for value in morph_iterator {
                                        weight_morphs.push(value);
                                    }
                                }
                            }
                        }
                    }
                }
                None => {
                    panic!("Empty animation outputs");
                }
            };                                                    

            channels.push(AnimationChannel {
                target_index: channel.target().node().index(),
                property: animation_property,    
                interpolation,        
                timestamps,
                translations,
                rotations,
                scales,
                weight_morphs
            })
        }        

        animations.push(Animation {
            name: animation.name().unwrap_or("Default").to_owned(),
            channels
        });
    }

    info!("Animations total: {}", animations.len());

    (ModelData {        
        name: model_name.to_owned(),
        nodes,
        node_topological_sorting,
        node_map,
        skins,
        meshes,
        materials,
        is_animated,
        animations
    }, match add_images {
        true => Some(loaded_images),
        false => None
    })
}
