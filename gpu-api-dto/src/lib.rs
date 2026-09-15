use wincode::{SchemaWrite, SchemaRead, WriteError, ReadError, serialize, deserialize};
use serde_derive::{Serialize, Deserialize};
use image::DynamicImage;
use lz4_flex::compress_prepend_size;

#[derive(Debug, Clone, SchemaWrite, SchemaRead, Serialize, Deserialize)]
pub struct ModelData {
    pub name: String,
    pub nodes: Vec<Node>,
    pub node_topological_sorting: Vec<usize>,
    pub node_map: std::collections::BTreeMap<usize, usize>,
    pub skins: Vec<Skin>,
	pub meshes: Vec<MeshData>,
    pub materials: Vec<MaterialData>,
    pub is_animated: bool,
    pub animations: Vec<Animation>    
}

#[derive(Debug, Clone, SchemaWrite, SchemaRead, Serialize, Deserialize)]
pub struct Node {
    pub index: usize,    
    pub children_nodes: Vec<usize>,
    pub name: Option<String>,
    pub skin_index: Option<usize>,
    pub mesh_index: Option<usize>,
    pub translation: [f32; 3], 
    pub rotation: [f32; 4], 
    pub scale: [f32; 3],
    pub local_transform_matrix: [[f32; 4]; 4]
}

#[derive(Debug, Clone, SchemaWrite, SchemaRead, Serialize, Deserialize)]
pub struct Skin {
    pub name: Option<String>,
    pub inverse_bind_matrices: Vec<[[f32; 4]; 4]>,
    pub joints: Vec<Joint>
}

#[derive(Debug, Clone, SchemaWrite, SchemaRead, Serialize, Deserialize)]
pub struct Joint {
    pub node_index: usize,
    pub node_name: Option<String>
}

#[derive(Debug, Clone, SchemaWrite, SchemaRead, Serialize, Deserialize)]
pub struct MeshData {
    pub index: usize,
    pub node_transform: Option<[[f32; 4]; 4]>,
	pub primitives: Vec<PrimitiveData>
}

#[derive(Debug, Clone, SchemaWrite, SchemaRead, Serialize, Deserialize)]
pub struct PrimitiveData {
	pub positions: Vec<[f32; 3]>,
	pub indices: Vec<u32>,
	pub normals: Vec<[f32; 3]>,
	pub tangents: Vec<[f32; 3]>,
    pub bitangents: Vec<[f32; 3]>,
    pub joints: Vec<[u32; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub material_index: Option<usize>,
    pub texture_coordinates: Vec<[f32; 2]>    
}

impl PrimitiveData {
    pub fn new() -> PrimitiveData {
        PrimitiveData {
            positions: vec![],
            indices: vec![],
            normals: vec![],
            tangents: vec![],
            bitangents: vec![],
            joints: vec![],
            weights: vec![],
            texture_coordinates: vec![],
            material_index: Some(0)
        }
    }
}

#[derive(Debug, Clone, SchemaWrite, SchemaRead, Serialize, Deserialize)]
pub enum AnimationComputationMode {
    NotAnimated,
    ComputeInRealTime,
    PreComputed
}

#[derive(Debug, Clone, SchemaWrite, SchemaRead, Serialize, Deserialize)]
pub struct Animation {
    pub name: String,
    pub channels: Vec<AnimationChannel>
}

#[derive(Debug, Clone, SchemaWrite, SchemaRead, Serialize, Deserialize)]
pub struct AnimationChannel {
    pub target_index: usize,
    pub property: AnimationProperty,
    pub interpolation: Interpolation,
    pub timestamps: Vec<f32>,
    pub translations: Vec<[f32; 3]>,
    pub rotations: Vec<[f32; 4]>,
    pub scales: Vec<[f32; 3]>,
    pub weight_morphs: Vec<f32>
}

#[derive(Debug, Clone, SchemaWrite, SchemaRead, Serialize, Deserialize)]
pub enum Interpolation {
    Linear,
    Step,
    CubicSpline
}

#[derive(Debug, Clone, SchemaWrite, SchemaRead, Serialize, Deserialize)]
pub enum AnimationProperty {    
    Translation,    
    Rotation,    
    Scale,    
    MorphTargetWeights
}

#[derive(Debug, Clone, SchemaWrite, SchemaRead, Serialize, Deserialize)]
pub enum ImageFormat {
    R8G8B8,
    R8G8B8A8,
    R16G16B16,
    R16G16B16A16,
    Other
}

#[derive(Debug, Clone, SchemaWrite, SchemaRead, Serialize, Deserialize)]
pub enum TextureCompressionFormat {
    NotCompressed,
    ASTC,
    BC7
}

#[derive(Debug, Clone, SchemaWrite, SchemaRead, Serialize, Deserialize)]
pub enum TextureType {
    Diffuse,
    SpecularGlossinessDiffuse,
    SpecularGlossiness,
    BaseColor,
    MetallicRoughness,
    Normal,
    Occlusion,
    Emissive
}

impl TextureType {
    pub fn is_srgb(&self) -> bool {
        match self {
            TextureType::Normal |
            TextureType::Occlusion |
            TextureType::MetallicRoughness => true,
            _ => false
        }
    }
}

#[derive(Debug, Clone, SchemaWrite, SchemaRead, Serialize, Deserialize)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend
}

#[derive(Debug, Clone, SchemaWrite, SchemaRead, Serialize, Deserialize)]
pub struct MaterialData {
    pub index: usize,
    pub name: Option<String>,
    pub alpa_mode: AlphaMode,
    pub textures: Vec<TextureData>,
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_factor: [f32; 3]
}

impl MaterialData {
    pub fn new(texture_data: Vec<(TextureType, &[u8])>) -> MaterialData {
        let mut textures = vec![];
        let mut texture_index = 0;

        for (texture_type, data) in texture_data {
            let loaded_image = image::load_from_memory(data).expect("Failed to load texture image");
        
            let image_width = loaded_image.width();
            let image_height = loaded_image.height();
        
            let texture_data = compress_prepend_size(loaded_image.to_rgba8().as_raw());
            
            textures.push(TextureData {
                index: texture_index,
                name: None,
                image_index: texture_index,
                loaded_image_index: texture_index,
                texture_type,
                image_format: ImageFormat::R8G8B8A8,
                compression_format: TextureCompressionFormat::NotCompressed,
                width: image_width,
                height: image_height,
                payload: Some(texture_data)
            });            

            texture_index = texture_index + 1;
        };

        MaterialData {
            index: 0,
            name: None,
            alpa_mode: AlphaMode::Blend,
            textures,
            base_color_factor: [1.0; 4],
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            emissive_factor: [0.0; 3]
        }
    }

    pub fn from_images(texture_data: Vec<(TextureType, &DynamicImage)>) -> MaterialData {
        let mut textures = vec![];
        let mut texture_index = 0;

        for (texture_type, loaded_image) in texture_data {                    
            let image_width = loaded_image.width();
            let image_height = loaded_image.height();
        
            let texture_data = compress_prepend_size(loaded_image.to_rgba8().as_raw());
            
            textures.push(TextureData {
                index: texture_index,
                name: None,
                image_index: texture_index,
                loaded_image_index: texture_index,
                texture_type,
                image_format: ImageFormat::R8G8B8A8,
                compression_format: TextureCompressionFormat::NotCompressed,
                width: image_width,
                height: image_height,
                payload: Some(texture_data)
            });            

            texture_index = texture_index + 1;
        };

        MaterialData {
            index: 0,
            name: None,
            alpa_mode: AlphaMode::Blend,
            textures,
            base_color_factor: [1.0; 4],
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            emissive_factor: [0.0; 3]
        }
    }
}

#[derive(Debug, Clone, SchemaWrite, SchemaRead, Serialize, Deserialize)]
pub struct TextureData {
    pub index: usize,
    pub name: Option<String>,
    pub image_index: usize,
    pub loaded_image_index: usize,
    pub image_format: ImageFormat,
    pub compression_format: TextureCompressionFormat,
    pub texture_type: TextureType,
    pub width: u32,
    pub height: u32,
    pub payload: Option<Vec<u8>>
}

#[derive(Debug, Clone, SchemaWrite, SchemaRead, Serialize, Deserialize)]
pub struct ViewSource {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub scale_z: f32,
    pub rotation_y: f32
}

pub fn serialize_model_data(model_data: &ModelData) -> Result<Vec<u8>, WriteError> {
    serialize(model_data)
}

pub fn deserialize_model_data(buf: &[u8]) -> Result<ModelData, ReadError> {
    deserialize(&buf)
}
