use log::*;
use glam::{Mat4, Quat};
use image::DynamicImage;
use lz4_flex::decompress_size_prepended;
use wgpu::{Device, Buffer, util::DeviceExt, BindGroup, Queue};
use gpu_api_relay::model_bindless_data::{InstanceData, MaterialFactors, NodeData};
use gpu_api_dto::{AlphaMode, AnimationComputationMode, AnimationProperty, Interpolation, ModelData, PrimitiveData, TextureType, ViewSource};
use crate::pipeline::model_pipeline::model_instance::ModelInstance;
use crate::pipeline::model_pipeline::Vertex;
use crate::texture::Texture;
use crate::{pipeline::model_pipeline::{self, MaterialFactorsUniform, INSTANCE_SIZE, MAX_MODEL_INSTANCES_COUNT}};

pub struct Object {
    pub name: String,
    pub nodes: Vec<ModelNode>,
    pub node_topological_sorting: Vec<usize>,
    pub node_map: std::collections::BTreeMap<usize, usize>,
    pub skins: Vec<ModelSkin>,
    pub meshes: Vec<Mesh>,
    pub animation_computation_mode: AnimationComputationMode,
    pub animations: Vec<ModelAnimation>,
    pub instance_buffer: Buffer,
    pub instances: Vec<ObjectInstance>,    
    pub materials: Vec<ObjectMaterial>,
    pub joint_matrices_buffer: Buffer,
    pub joint_matrices_bind_group: BindGroup,
    pub model_instances: Vec<ModelInstance>,
    pub model_instance_size: u64,
    pub instances_count: u32
}

pub struct ModelNode {
    pub index: usize,
    pub name: Option<String>,
    pub skin_index: Option<usize>,
    pub mesh_index: Option<usize>,
    pub translation: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
    pub local_transform_matrix: Mat4,
    pub global_transform_matrix: Mat4
}

pub struct ModelSkin {
    pub name: Option<String>,    
    pub joints: Vec<ModelJoint>
}

pub struct ModelJoint {
    pub node_index: usize,
    pub node_name: Option<String>,
    pub inverse_bind_matrix: Mat4
}

pub struct Mesh {    
    pub name: String,
    pub index: usize,
    pub node_transform: Option<NodeData>,    
    pub primitives: Vec<Primitive>,
    pub node_transform_buffer: Buffer,
    pub node_transform_bind_group: BindGroup
}

pub struct Primitive {
    pub name: String,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32,
    pub material_index: usize
}

impl Primitive {
    pub fn get_pipeline_vertices(primitive_data: &PrimitiveData) -> Vec<Vertex> {
        let mut index = 0;
        let mut vertices = vec![];

        for position in &primitive_data.positions {
            vertices.push(model_pipeline::Vertex {
                position: *position,
                uv: if primitive_data.texture_coordinates.is_empty() {[0.0, 1.0]} else {primitive_data.texture_coordinates[index]},
                normal: if primitive_data.normals.is_empty() {[1.0, 1.0, 1.0]} else {primitive_data.normals[index]},
                tangent: if primitive_data.tangents.is_empty() {[1.0, 1.0, 1.0]} else {primitive_data.tangents[index]},
                bitangent: if primitive_data.bitangents.is_empty() {[1.0, 1.0, 1.0]} else {primitive_data.tangents[index]},
                joints: if primitive_data.joints.is_empty() {[0, 0, 0, 0]} else {primitive_data.joints[index]},
                weights: if primitive_data.weights.is_empty() {[1.0, 1.0, 1.0, 1.0]} else {primitive_data.weights[index]}
            });

            index = index + 1;
        }
        
        vertices        
    }

    pub fn update_buffers(&mut self, device: &Device, vertices: &Vec<Vertex>, indices: &Vec<u32>) {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Vertex Buffer", "dog")),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX
        });
    
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Index Buffer", "dog")),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX
        });

        self.vertex_buffer = vertex_buffer;
        self.index_buffer = index_buffer;
        self.num_elements = indices.len() as u32;
    }
}

pub struct ObjectMaterial {
    pub name: String,
    pub alpha_mode: AlphaMode,
    pub material_bind_group: wgpu::BindGroup,
    pub base_color_texture: crate::texture::Texture,    
    pub normal_texture: crate::texture::Texture,    
    pub metallic_roughness_texture: crate::texture::Texture,
    pub emissive_texture: Option<crate::texture::Texture>,
    pub occlusion_texture: Option<crate::texture::Texture>,
    pub factors_buffer: wgpu::Buffer
}

pub struct ObjectGroup {
    pub active: bool,
    pub objects: Vec<Object>
}

pub struct ObjectInstance {
    pub view_source: ViewSource,    
    pub is_moving: bool,
    pub move_target: glam::Vec3,
    pub move_direction_normalized: glam::Vec3,
    pub x_move_done: bool,
    pub y_move_done: bool,
    pub z_move_done: bool,
    pub bounding_box: BoundingBox
}

pub struct BoundingBox {
    pub box0: glam::Vec3,
    pub box1: glam::Vec3
}

pub struct ModelAnimation {
    pub name: String,
    pub channels: Vec<ModelAnimationChannel>,
    pub frame_index: usize,
    pub frame_cycle_count: usize,
    pub joint_matrices: Vec<[Mat4; model_pipeline::JOINT_MATRICES_COUNT]>,
    pub mesh_node_transforms: Vec<MeshNodeTransform>
}

pub struct MeshNodeTransform {
    pub mesh_index: usize,
    pub node_transforms: Vec<NodeData>
}

pub struct ModelAnimationChannel {
    pub target_index: usize,
    pub property: AnimationProperty,
    pub interpolation: Interpolation,
    pub timestamps: Vec<f32>,
    pub translations: Vec<glam::Vec3>,
    pub rotations: Vec<glam::Quat>,
    pub scales: Vec<glam::Vec3>,
    pub weight_morphs: Vec<f32>,
    pub frame_index: usize,
    pub channel_time: f32,
    #[cfg(not(target_arch = "wasm32"))]
    pub start_instant: std::time::Instant,
    #[cfg(target_arch = "wasm32")]
    pub start_instant: web_time::Instant
}

pub struct InitData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub factors: Vec<MaterialFactors>,
    pub materials: Vec<MaterialData>,
    pub instances: Vec<InstanceData>,
    pub joints: Vec<Mat4>,
    pub nodes: Vec<NodeData>,
}

impl InitData {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            factors: Vec::new(),
            materials: Vec::new(),
            instances: Vec::new(),
            joints: Vec::new(),
            nodes: Vec::new(),
        }
    }
}

pub struct MaterialData {    
    pub textures: Vec<(TextureType, Texture)>,
}

impl Object {
    pub fn new(device: &Device, queue: &Queue, pipeline: &model_pipeline::Pipeline, model_data: ModelData, view_sources: Vec<ViewSource>, loaded_images: Option<Vec<DynamicImage>>, frame_cycle_length: usize, init_data: &mut InitData) -> Object {
        let mut meshes = vec![];    

        for mesh in model_data.meshes {
            let mut primitives = vec![];            

            for primitive_data in mesh.primitives {                
                let vertices = Primitive::get_pipeline_vertices(&primitive_data);

                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{} Vertex Buffer", "dog")),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX
                });
                        
                let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{} Index Buffer", "dog")),
                    contents: bytemuck::cast_slice(&primitive_data.indices),
                    usage: wgpu::BufferUsages::INDEX
                });
                
                let base_vertex = init_data.vertices.len() as u32;

                let globalized_indices: Vec<u32> = primitive_data.indices
                    .iter()
                    .map(|&idx| idx as u32 + base_vertex)
                    .collect();
                
                init_data.vertices.extend_from_slice(&vertices);
                init_data.indices.extend_from_slice(&globalized_indices);
                //init_data.indices.extend_from_slice(&primitive_data.indices);
        
                primitives.push(Primitive {
                    name: "".to_owned(),
                    vertex_buffer,
                    index_buffer,
                    num_elements: primitive_data.indices.len() as u32,
                    material_index: primitive_data.material_index.expect("Primitive material index is empty")
                });
            }

            let node_transform = NodeData {
                info: [0; 4],
                transform: Mat4::IDENTITY                
            };         

            let node_transform_buffer = device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Node transform Buffer"),
                    contents: bytemuck::bytes_of(&node_transform),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                }
            );

            let node_transform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &pipeline.node_transform_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: node_transform_buffer.as_entire_binding(),
                    }
                ],
                label: Some("node_transform_bind_group")
            });

            meshes.push(Mesh {
                name: "".to_owned(),
                index: mesh.index,
                node_transform: mesh.node_transform.map(|r| NodeData { 
                    info: [1, 0, 0, 0],                     
                    transform: Mat4::from_cols_array_2d(&r)
                }),            
                primitives,
                node_transform_buffer,
                node_transform_bind_group
            });
        }        

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instance Buffer"),
            size: INSTANCE_SIZE * MAX_MODEL_INSTANCES_COUNT,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let mut materials = vec![];

        match loaded_images {
            Some(loaded_images) => {
                for material in model_data.materials {
                    let mut base_color = None;
                    let mut metallic_roughness = None;
                    let mut normal = None;
                    let mut emissive = None;
                    let mut occlusion = None;

                    let mut textures = Vec::new();

                    for texture_item in material.textures {                    
                        let index_str = texture_item.index.to_string();
        
                        log::warn!("Creating texture {:?} from image: {}, index {}", texture_item.texture_type, model_data.name, texture_item.index);                                 
                
                        let texture = crate::texture::Texture::from_image(&device, &queue, &loaded_images[texture_item.loaded_image_index], &texture_item.image_format, texture_item.texture_type.is_srgb(), Some(&("texture_".to_owned() + &index_str))).expect("Failed to create texture");
                        //let texture = crate::texture::Texture::from_image_to_rgba8(&device, &queue, &loaded_images[texture_item.loaded_image_index], texture_item.is_srgb(), Some(&("texture_".to_owned() + &index_str))).expect("Failed to create texture");

                        let texture2 = crate::texture::Texture::from_image(&device, &queue, &loaded_images[texture_item.loaded_image_index], &texture_item.image_format, texture_item.texture_type.is_srgb(), Some(&("texture_".to_owned() + &index_str))).expect("Failed to create texture");

                        textures.push((texture_item.texture_type.clone(), texture2));
                
                        match texture_item.texture_type {
                            TextureType::BaseColor => {
                                base_color = Some(texture);
                            }
                            TextureType::MetallicRoughness => {
                                metallic_roughness = Some(texture);
                            }
                            TextureType::Normal => {
                                normal = Some(texture);
                            }
                            TextureType::Emissive => {
                                emissive = Some(texture);
                            }
                            TextureType::Occlusion => {
                                occlusion = Some(texture);
                            }
                            _ => panic!("Not supported texture type found")
                        };
                    }

                    let base_color_texture = base_color.expect("Base color is empty");                
                    let metallic_roughness_texture = metallic_roughness.expect("Metallic roughness is empty");
                    let normal_texture = normal.expect("Normal is empty");
                    //let emissive_texture = emissive.as_ref().expect("Emissive is empty");

                    let factors = MaterialFactorsUniform {
                        base_color_factor: material.base_color_factor,
                        metallic_factor: material.metallic_factor,
                        roughness_factor: material.roughness_factor,
                        emissive_factor: material.emissive_factor,
                        padding: [0, 0, 0]
                    };
            
                    let factors_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Material Factors Buffer"),
                        contents: bytemuck::bytes_of(&factors),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

                    let material_bind_group = device.create_bind_group(
                        &wgpu::BindGroupDescriptor {
                            layout: &pipeline.material_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(&base_color_texture.view),
                                    //resource: wgpu::BindingResource::TextureViewArray(&textures.iter().map(|v| &v.view).collect::<Vec<&TextureView>>())
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(&pipeline.base_color_sampler)
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::TextureView(&metallic_roughness_texture.view),
                                    //resource: wgpu::BindingResource::TextureViewArray(&textures.iter().map(|v| &v.view).collect::<Vec<&TextureView>>())
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::Sampler(&pipeline.metallic_roughness_sampler)
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: wgpu::BindingResource::TextureView(&normal_texture.view),
                                    //resource: wgpu::BindingResource::TextureViewArray(&textures.iter().map(|v| &v.view).collect::<Vec<&TextureView>>())
                                },
                                wgpu::BindGroupEntry {
                                    binding: 5,
                                    resource: wgpu::BindingResource::Sampler(&pipeline.normal_sampler)
                                },
                                wgpu::BindGroupEntry {
                                    binding: 6,
                                    resource: wgpu::BindingResource::Buffer(
                                        factors_buffer.as_entire_buffer_binding()
                                    )
                                },
                                /*
                                wgpu::BindGroupEntry {
                                    binding: 7,
                                    resource: wgpu::BindingResource::TextureView(&emissive_texture.view),
                                    //resource: wgpu::BindingResource::TextureViewArray(&textures.iter().map(|v| &v.view).collect::<Vec<&TextureView>>())
                                },
                                wgpu::BindGroupEntry {
                                    binding: 8,
                                    resource: wgpu::BindingResource::Sampler(&pipeline.emissive_sampler)
                                },
                                */                            
                            ],
                            label: Some(&("Material bind group ".to_owned() + &material.index.to_string())),
                        }
                    );

                    init_data.factors.push(MaterialFactors {
                        base_color_factor: material.base_color_factor,
                        emissive_factor: material.emissive_factor,
                        metallic_factor: material.metallic_factor,
                        roughness_factor: material.roughness_factor,
                        padding: [0, 0, 0],
                    });
                    init_data.materials.push(MaterialData {
                        textures,
                    });

                    materials.push(ObjectMaterial {
                        name: material.name.unwrap_or("default".to_owned()),
                        alpha_mode: material.alpa_mode,
                        material_bind_group,
                        base_color_texture,                    
                        normal_texture,                    
                        metallic_roughness_texture,
                        emissive_texture: emissive,
                        occlusion_texture: occlusion,
                        factors_buffer
                    });
                }
            }
            None => {
                for material in model_data.materials {
                    let mut base_color = None;
                    let mut metallic_roughness = None;
                    let mut normal = None;
                    let mut emissive = None;
                    let mut occlusion = None;

                    let mut textures = Vec::new();

                    for texture_item in material.textures {
                        let index_str = texture_item.index.to_string();

                        log::warn!("Creating texture from vec: {}, index {}, size {}", model_data.name, texture_item.index, texture_item.payload.as_ref().expect("Empty encoded image for texture").len());

                        let texture_payload = texture_item.payload.as_ref().expect("Empty encoded image for texture");                
                        
                        let texture_image = match texture_item.image_format {
                            gpu_api_dto::ImageFormat::R8G8B8 => {                    
                                let image_buffer = image::ImageBuffer::from_raw(texture_item.width, texture_item.height, decompress_size_prepended(&texture_payload).expect("Failed to decompress texture payload")).expect("Failed to create image buffer");
                                DynamicImage::ImageRgb8(image_buffer)
                            }
                            gpu_api_dto::ImageFormat::R8G8B8A8 => {
                                let image_buffer = image::ImageBuffer::from_raw(texture_item.width, texture_item.height, decompress_size_prepended(&texture_payload).expect("Failed to decompress texture payload")).expect("Failed to create image buffer");
                                DynamicImage::ImageRgba8(image_buffer)
                            }
                            gpu_api_dto::ImageFormat::R16G16B16 => {
                                let pixels_u16: Vec<u16> = bytemuck::cast_slice(&decompress_size_prepended(&texture_payload).expect("Failed to decompress texture payload")).to_vec();
                                let image_buffer = image::ImageBuffer::from_raw(texture_item.width, texture_item.height, pixels_u16).expect("Failed to create image buffer");
                                DynamicImage::ImageRgb16(image_buffer)
                            }
                            gpu_api_dto::ImageFormat::R16G16B16A16 => {
                                let pixels_u16: Vec<u16> = bytemuck::cast_slice(&decompress_size_prepended(&texture_payload).expect("Failed to decompress texture payload")).to_vec();
                                let image_buffer = image::ImageBuffer::from_raw(texture_item.width, texture_item.height, pixels_u16).expect("Failed to create image buffer");
                                DynamicImage::ImageRgba16(image_buffer)
                            }
                            _ => {
                                let image_buffer = image::ImageBuffer::from_raw(texture_item.width, texture_item.height, decompress_size_prepended(&texture_payload).expect("Failed to decompress texture payload")).expect("Failed to create image buffer");
                                DynamicImage::ImageRgb8(image_buffer)
                            }                        
                        };

                        //let texture = crate::texture::Texture::from_image(&device, &queue, &texture_image, &texture_item.image_format, texture_item.is_srgb(), Some(&("texture_".to_owned() + &index_str))).expect("Failed to create texture");
                        let texture = crate::texture::Texture::from_image_to_rgba8(&device, &queue, &texture_image, texture_item.texture_type.is_srgb(), Some(&("texture_".to_owned() + &index_str))).expect("Failed to create texture");
                        let texture2 = crate::texture::Texture::from_image_to_rgba8(&device, &queue, &texture_image, texture_item.texture_type.is_srgb(), Some(&("texture_".to_owned() + &index_str))).expect("Failed to create texture");

                        textures.push((texture_item.texture_type.clone(), texture2));

                                        
                        /*
                        let texture_image = image::load_from_memory_with_format(texture_item.payload.as_ref().expect("Image encoded is empty"), image::ImageFormat::Jpeg).expect("Failed to load texture");
                        let texture = crate::texture::Texture::from_image(&device, &queue, &texture_image, Some(&("texture_".to_owned() + &index_str))).expect("Failed to create texture");
                        */

                        //let texture = crate::texture::Texture::from_ktx2(&device, &queue, texture_item.image_encoded.as_ref().expect("Empty encoded image for texture"), 512, 512, Some(&("texture_".to_owned() + &index_str))).expect("Failed to create texture");                                                                    
                
                        match texture_item.texture_type {
                            TextureType::BaseColor => {
                                base_color = Some(texture);
                            }
                            TextureType::MetallicRoughness => {
                                metallic_roughness = Some(texture);
                            }
                            TextureType::Normal => {
                                normal = Some(texture);
                            }
                            TextureType::Emissive => {
                                emissive = Some(texture);
                            }
                            TextureType::Occlusion => {
                                occlusion = Some(texture);
                            }
                            _ => panic!("Not supported texture type found")
                        };
                    }

                    let base_color_texture = base_color.expect("Base color is empty");
                    let metallic_roughness_texture = metallic_roughness.expect("Metallic roughness is empty");
                    let normal_texture = normal.expect("Normal is empty");

                    let factors = MaterialFactorsUniform {
                        base_color_factor: material.base_color_factor,
                        metallic_factor: material.metallic_factor,
                        roughness_factor: material.roughness_factor,
                        emissive_factor: material.emissive_factor,
                        padding: [0, 0, 0]
                    };
            
                    let factors_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Material Factors Buffer"),
                        contents: bytemuck::bytes_of(&factors),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

                    let material_bind_group = device.create_bind_group(
                        &wgpu::BindGroupDescriptor {
                            layout: &pipeline.material_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(&base_color_texture.view),
                                    //resource: wgpu::BindingResource::TextureViewArray(&textures.iter().map(|v| &v.view).collect::<Vec<&TextureView>>())
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(&pipeline.base_color_sampler)
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::TextureView(&metallic_roughness_texture.view),
                                    //resource: wgpu::BindingResource::TextureViewArray(&textures.iter().map(|v| &v.view).collect::<Vec<&TextureView>>())
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::Sampler(&pipeline.metallic_roughness_sampler)
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: wgpu::BindingResource::TextureView(&normal_texture.view),
                                    //resource: wgpu::BindingResource::TextureViewArray(&textures.iter().map(|v| &v.view).collect::<Vec<&TextureView>>())
                                },
                                wgpu::BindGroupEntry {
                                    binding: 5,
                                    resource: wgpu::BindingResource::Sampler(&pipeline.normal_sampler)
                                },
                                wgpu::BindGroupEntry {
                                    binding: 6,
                                    resource: wgpu::BindingResource::Buffer(
                                        factors_buffer.as_entire_buffer_binding()
                                    )
                                }
                            ],
                            label: Some(&("Material bind group ".to_owned() + &material.index.to_string())),
                        }
                    );

                    init_data.factors.push(MaterialFactors {
                        base_color_factor: material.base_color_factor,
                        emissive_factor: material.emissive_factor,
                        metallic_factor: material.metallic_factor,
                        roughness_factor: material.roughness_factor,
                        padding: [0, 0, 0],
                    });
                    init_data.materials.push(MaterialData {
                        textures,
                    });

                    materials.push(ObjectMaterial {
                        name: material.name.unwrap_or("default".to_owned()),
                        alpha_mode: material.alpa_mode,
                        material_bind_group,
                        base_color_texture,
                        normal_texture,
                        metallic_roughness_texture,
                        emissive_texture: emissive,
                        occlusion_texture: occlusion,
                        factors_buffer
                    });                    
                }
            }
        }    

        let mut views = vec![];
        let mut instances = vec![];

        for view_source in view_sources {
            let model_matrix = generate_model_matrix(&view_source);        

            views.push(ModelInstance {
                model_matrix: model_matrix,
                is_animated: match model_data.is_animated {
                    true => 1,
                    false => 0                
                },
                padding: [0, 0, 0]
            });

            instances.push(ObjectInstance {
                view_source,            
                is_moving: false,
                move_target: glam::vec3(0.0, 0.0, 0.0),
                move_direction_normalized: glam::vec3(0.0, 0.0, 0.0),
                x_move_done: false,
                y_move_done: false,
                z_move_done: false,
                bounding_box: BoundingBox { 
                    box0: glam::vec3(0.0, 0.0, 0.0),
                    box1: glam::vec3(0.0, 0.0, 0.0)
                }
            });
        }

        let views_size = views.len() as u64 * INSTANCE_SIZE;

        let instances_count = instances.len() as u32;

        let mut animations = vec![];

        for animation in model_data.animations {
            let mut channels = vec![];

            for channel in animation.channels {
                let translations = channel.translations.into_iter().map(|v| glam::Vec3::from_array(v)).collect();
                let rotations = channel.rotations.into_iter().map(|v| glam::Quat::from_array(v)).collect();
                let scales = channel.scales.into_iter().map(|v| glam::Vec3::from_array(v)).collect();

                channels.push(ModelAnimationChannel {
                    target_index: channel.target_index,
                    property: channel.property,
                    interpolation: channel.interpolation,
                    timestamps: channel.timestamps,
                    translations,
                    rotations,
                    scales,
                    weight_morphs: channel.weight_morphs,
                    frame_index: 0,
                    channel_time: 0.0,
                    #[cfg(not(target_arch = "wasm32"))]
                    start_instant: std::time::Instant::now(),
                    #[cfg(target_arch = "wasm32")]
                    start_instant: web_time::Instant::now()
                });
            }

            animations.push(ModelAnimation {
                name: animation.name,
                channels,
                frame_index: 0,
                frame_cycle_count: 0,
                joint_matrices: vec![],
                mesh_node_transforms: meshes.iter().map(|r| MeshNodeTransform {
                    mesh_index: r.index,
                    node_transforms: vec![]
                }).collect()
            });
        }

        let joint_matrices: [Mat4; model_pipeline::JOINT_MATRICES_COUNT] = [Mat4::IDENTITY; model_pipeline::JOINT_MATRICES_COUNT];

        //let joint_matrices_ref: &[[f32; 16]] = joint_matrices.as_ref();

        let joint_matrices_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Joint matrices Buffer"),
                contents: bytemuck::cast_slice(&joint_matrices),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let joint_matrices_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &pipeline.joint_matrices_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: joint_matrices_buffer.as_entire_binding(),
                }
            ],
            label: Some("joint_matrices_bind_group")
        });

        let mut nodes = vec![];

        for node in model_data.nodes {
            let local_transform_matrix = Mat4::from_cols_array_2d(&node.local_transform_matrix);
            
            nodes.push(ModelNode {
                index: node.index,
                name: node.name,            
                skin_index: node.skin_index,
                mesh_index: node.mesh_index,
                translation: glam::Vec3::from_array(node.translation),
                rotation: glam::Quat::from_array(node.rotation),
                scale: glam::Vec3::from_array(node.scale),
                local_transform_matrix,
                global_transform_matrix: local_transform_matrix
            });
        }

        let mut skins = vec![];

        for skin in model_data.skins {
            let mut joints = vec![];
            let mut index = 0;

            for joint in skin.joints {
                joints.push(ModelJoint {
                    node_index: joint.node_index,
                    node_name: joint.node_name,
                    inverse_bind_matrix: Mat4::from_cols_array_2d(&skin.inverse_bind_matrices[index])
                });

                index = index + 1;
            }

            skins.push(ModelSkin { 
                name: skin.name, 
                joints
            });        
        }

        let time_per_frame = 1.0 / frame_cycle_length as f32;

        for animation in &mut animations {        
            warn!("Animation {} started", animation.name);

            let mut animation_time = 0.0;

            let mut max_animation_time = 0.0;         

            for channel in &animation.channels {
                let max_channel_time = *channel.timestamps.last().expect("Empty animation timestamps");
                //info!("Channel max time value: {}", max_channel_time);            

                if max_channel_time > max_animation_time {
                    max_animation_time = max_channel_time;
                }
            }
                
            while animation_time < max_animation_time {
                for channel in &mut animation.channels {
                    let mut frame_index = channel.frame_index;            
        
                    for timestamp in channel.timestamps.iter().skip(frame_index) {
                        if timestamp > &channel.channel_time {
                            break;
                        }
                        
                        frame_index = frame_index + 1;
                    }
                    
                    if frame_index == channel.timestamps.len() {
                        frame_index = 0;
                        channel.frame_index = 0;
                        channel.channel_time = 0.0;
                    }                        
        
                    let previous_frame_index = match frame_index {
                        0 => 0,
                        _ => frame_index - 1
                    };                
        
                    let factor = (animation_time - channel.timestamps[previous_frame_index]) / (channel.timestamps[frame_index] - channel.timestamps[previous_frame_index]);
        
                    match &channel.property {
                        AnimationProperty::Translation => {
                            let translation = channel.translations[previous_frame_index].lerp(channel.translations[frame_index], factor);
                            nodes[channel.target_index].translation = translation;
                        }
                        AnimationProperty::Rotation => {                                                        
                            let rotation = channel.rotations[previous_frame_index].lerp(channel.rotations[frame_index], factor).normalize();
                            nodes[channel.target_index].rotation = rotation;
                        }
                        AnimationProperty::Scale => {
                            let scale = channel.scales[previous_frame_index].lerp(channel.scales[frame_index], factor);
                            nodes[channel.target_index].scale = scale;
                        }
                        AnimationProperty::MorphTargetWeights => {
                            let weight_morph = channel.weight_morphs[frame_index];
                        }
                    }

                    channel.channel_time = channel.channel_time + time_per_frame;                
                }

                for node_index in model_data.node_topological_sorting.iter() {
                    match model_data.node_map.get(node_index) {
                        Some(parent_index) => {
                            let parent_transform = nodes[*parent_index].global_transform_matrix;
                            let node = &mut nodes[*node_index];
            
                            let local_transform = glam::Mat4::from_scale_rotation_translation(node.scale, node.rotation, node.translation);

                            node.global_transform_matrix = parent_transform * local_transform;                        
                        }
                        None => {}
                    }
                }           

                let mut joint_matrices: [Mat4; model_pipeline::JOINT_MATRICES_COUNT] = [Mat4::IDENTITY; model_pipeline::JOINT_MATRICES_COUNT];
                
                let mut joint_matrix_index = 0;
                let skin_index = 0;            
                
                for joint in &skins[skin_index].joints {
                    let joint_matrix = nodes[joint.node_index].global_transform_matrix * joint.inverse_bind_matrix;
                    joint_matrices[joint_matrix_index] = joint_matrix;
                    joint_matrix_index = joint_matrix_index + 1;
                }

                init_data.joints.extend_from_slice(&joint_matrices);

                for mesh in &mut meshes {
                    if mesh.node_transform.is_some() {                    
                        let mesh_node_transform = animation.mesh_node_transforms.iter_mut().find(|r| r.mesh_index == mesh.index).expect("Mesh node tansform not found");

                        match nodes.iter().find(|r| r.mesh_index == Some(mesh.index)) {
                            Some(node) => {                            
                                mesh_node_transform.node_transforms.push(NodeData {
                                    info: [1, 0, 0, 0],                                     
                                    transform: node.global_transform_matrix
                                });
                            }
                            None => {}
                        }                    
                    }                
                }                                   

                animation.joint_matrices.push(joint_matrices);                                    

                animation_time = animation_time + time_per_frame;
            }

            if animation.joint_matrices.len() < 8 {
                panic!("Not enough joint matrices");
            }
            
            animation.frame_cycle_count = animation.joint_matrices.len();

            warn!("Animation {} done, joint matrices total: {}", animation.name, animation.joint_matrices.len());
        }

        Object {
            name: model_data.name,
            nodes,
            node_topological_sorting: model_data.node_topological_sorting,
            node_map: model_data.node_map,
            skins,
            meshes,
            animation_computation_mode: match model_data.is_animated {
                true => AnimationComputationMode::PreComputed,
                false => AnimationComputationMode::NotAnimated
            },
            animations,
            instance_buffer,        
            joint_matrices_buffer,
            joint_matrices_bind_group,
            instances,        
            materials,
            model_instances: views,
            model_instance_size: views_size,
            instances_count        
        }
    }

    pub fn update_all_views(&mut self) {
        self.model_instances.clear();        

        for instance in &self.instances {
            let model_matrix = generate_model_matrix(&instance.view_source);
            self.model_instances.push(ModelInstance {
                model_matrix,
                is_animated: match self.animation_computation_mode {
                    AnimationComputationMode::PreComputed |
                    AnimationComputationMode::ComputeInRealTime => 1,
                    AnimationComputationMode::NotAnimated => 0
                },
                padding: [0, 0, 0]
            });
        }

        self.model_instance_size = self.model_instances.len() as u64 * INSTANCE_SIZE;
        self.instances_count = self.instances.len() as u32;
    }

    pub fn set_instance_data_size(&mut self) {
        self.model_instance_size = self.model_instances.len() as u64 * INSTANCE_SIZE;
        self.instances_count = self.instances.len() as u32;
    }

    pub fn add_instance_view(&mut self, view_source: ViewSource) {        
        let model_matrix = generate_model_matrix(&view_source);
        self.instances.push(ObjectInstance {
            view_source,            
            is_moving: false,
            move_target: glam::vec3(0.0, 0.0, 0.0),
            move_direction_normalized: glam::vec3(0.0, 0.0, 0.0),
            x_move_done: false,
            y_move_done: false,
            z_move_done: false,
            bounding_box: BoundingBox { 
                box0: glam::vec3(0.0, 0.0, 0.0),
                box1: glam::vec3(0.0, 0.0, 0.0)
            }
        });
        self.model_instances.push(ModelInstance {
            model_matrix,
            is_animated: match self.animation_computation_mode {
                AnimationComputationMode::PreComputed |
                AnimationComputationMode::ComputeInRealTime => 1,
                AnimationComputationMode::NotAnimated => 0
            },
            padding: [0, 0, 0]
        });
    }

    pub fn update_instance_view(&mut self, instance_index: usize) {        
        let instance = &self.instances[instance_index];
        let model_matrix = generate_model_matrix(&instance.view_source);
        self.model_instances[instance_index].model_matrix = model_matrix;
    }

    pub fn update_instance_view_with_rotation(&mut self, instance_index: usize) {
        let instance = &mut self.instances[instance_index];            
        let quat = Quat::from_rotation_y(instance.view_source.rotation_y);
        let rotation_matrix = glam::Mat4::from_quat(quat);
        let model_matrix = generate_model_matrix(&instance.view_source) * rotation_matrix;
        self.model_instances[instance_index].model_matrix = model_matrix;        
    }    

    pub fn update_view_with_translation(&mut self, translation: &[f32; 3]) {
        self.model_instances.clear();        

        let translation_matrix = glam::Mat4::from_translation(glam::Vec3::new(translation[0], translation[1], translation[2]));        

        for instance in &self.instances {
            let model_matrix = translation_matrix * generate_model_matrix(&instance.view_source);
            self.model_instances.push(ModelInstance {
                model_matrix: model_matrix,
                is_animated: match self.animation_computation_mode {
                    AnimationComputationMode::PreComputed |
                    AnimationComputationMode::ComputeInRealTime => 1,
                    AnimationComputationMode::NotAnimated => 0
                },
                padding: [0, 0, 0]
            });
        }

        self.model_instance_size = self.model_instances.len() as u64 * INSTANCE_SIZE;
        self.instances_count = self.instances.len() as u32;
    }

    pub fn update_view_with_rotation(&mut self, rotation: Quat) {
        self.model_instances.clear();        

        let rotation_matrix = glam::Mat4::from_quat(rotation);

        for instance in &self.instances {
            let model_matrix = rotation_matrix * generate_model_matrix(&instance.view_source);
            self.model_instances.push(ModelInstance {
                model_matrix: model_matrix,
                is_animated: match self.animation_computation_mode {
                    AnimationComputationMode::PreComputed |
                    AnimationComputationMode::ComputeInRealTime => 1,
                    AnimationComputationMode::NotAnimated => 0
                },
                padding: [0, 0, 0]
            });            
        }

        self.model_instance_size = self.model_instances.len() as u64 * INSTANCE_SIZE;
        self.instances_count = self.instances.len() as u32;
    }

    pub fn update_view_with_scale(&mut self, scale: &[f32; 3]) {
        self.model_instances.clear();        

        let scale_matrix = glam::Mat4::from_scale(glam::Vec3::new(scale[0], scale[1], scale[2]));

        for instance in &self.instances {
            let model_matrix = scale_matrix * generate_model_matrix(&instance.view_source);
            self.model_instances.push(ModelInstance {
                model_matrix: model_matrix,                
                is_animated: match self.animation_computation_mode {
                    AnimationComputationMode::PreComputed |
                    AnimationComputationMode::ComputeInRealTime => 1,
                    AnimationComputationMode::NotAnimated => 0
                },
                padding: [0, 0, 0]
            });            
        }

        self.model_instance_size = self.model_instances.len() as u64 * INSTANCE_SIZE;
        self.instances_count = self.instances.len() as u32;
    }
}

pub fn generate_model_matrix(source: &ViewSource) -> glam::Mat4 {
    glam::Mat4::from_scale_rotation_translation(
        glam::Vec3::new(source.scale_x, source.scale_y, source.scale_z),
        glam::Quat::IDENTITY,
        glam::Vec3::new(source.x, source.y, source.z),
    )
}
