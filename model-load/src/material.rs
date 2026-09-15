use lz4_flex::compress_prepend_size;
use image::{DynamicImage, ImageReader};
use log::*;
use gltf::{image::Format, Material};
use gpu_api_dto::{AlphaMode, ImageFormat, MaterialData, TextureCompressionFormat, TextureData, TextureType};

pub fn create_material_data(model_name: &str, material_index: usize, buffers: &Vec<gltf::buffer::Data>, images: &Vec<gltf::image::Data>, add_images: bool, add_pixes: bool, loaded_images: &mut Vec<DynamicImage>, material: &Material) -> MaterialData {
    info!("Found material {:?}, {:?}, {}", material.index(), material.name(), material.double_sided());

    let mut textures = vec![];        
    
    match material.pbr_specular_glossiness() {
        Some(pbr_specular_glossiness) => {
            match pbr_specular_glossiness.diffuse_texture() {
                Some(diffuse_texture) => {
                    info!("Found pbr specular glossiness diffuse texture, index {}, {:?}", diffuse_texture.texture().index(), diffuse_texture.texture().name());
                    let texture_data = create_texture_data(model_name, &buffers, &diffuse_texture.texture(), TextureType::SpecularGlossinessDiffuse, &images, add_images, add_pixes, loaded_images);
                    textures.push(texture_data);
                }
                None => {}
            }
            match pbr_specular_glossiness.specular_glossiness_texture() {
                Some(specular_glossiness_texture) => {
                    info!("Found pbr specular glossiness specular glossiness texture, index {}, {:?}", specular_glossiness_texture.texture().index(), specular_glossiness_texture.texture().name());
                    let texture_data = create_texture_data(model_name, &buffers, &specular_glossiness_texture.texture(), TextureType::SpecularGlossiness, &images, add_images, add_pixes, loaded_images);
                    textures.push(texture_data);
                }
                None => {}
            }
        }
        None => {}
    }

    let pbr_metallic_roughness = material.pbr_metallic_roughness();        

    match pbr_metallic_roughness.base_color_texture() {
        Some(base_color_texture) => {                
            info!("Found base color texture, index {}, {:?}", base_color_texture.texture().index(), base_color_texture.texture().name());
            let texture_data = create_texture_data(model_name, &buffers, &base_color_texture.texture(), TextureType::BaseColor, &images, add_images, add_pixes, loaded_images);
            textures.push(texture_data);
        }
        None => {
            warn!("No base_color_texture");
            let texture_data = create_texture_data_from_file(model_name, "DarkKnight_Eyes_Albedo", TextureType::BaseColor, add_images, add_pixes, loaded_images);
            textures.push(texture_data);
        }
    }
    match pbr_metallic_roughness.metallic_roughness_texture() {
        Some(metallic_roughness_texture) => {
            info!("Found metallic roughness texture, index {}, {:?}", metallic_roughness_texture.texture().index(), metallic_roughness_texture.texture().name());
            let texture_data = create_texture_data(model_name, &buffers, &metallic_roughness_texture.texture(), TextureType::MetallicRoughness, &images, add_images, add_pixes, loaded_images);
            textures.push(texture_data);
        }
        None => {
            match pbr_metallic_roughness.base_color_texture() {
                Some(base_color_texture) => {                
                    info!("Found base color texture, index {}, {:?}", base_color_texture.texture().index(), base_color_texture.texture().name());
                    let texture_data = create_texture_data(model_name, &buffers, &base_color_texture.texture(), TextureType::MetallicRoughness, &images, add_images, add_pixes, loaded_images);
                    textures.push(texture_data);
                }
                None => {
                    warn!("No metallic_roughness_texture");
                    let texture_data = create_texture_data_from_file(model_name, "DarkKnight_Eyes_Metalness", TextureType::MetallicRoughness, add_images, add_pixes, loaded_images);
                    textures.push(texture_data);
                }
            }
        }
    }
    match material.normal_texture() {
        Some(normal_texture) => {
            info!("Found normal texture, index {}, {:?}", normal_texture.texture().index(), normal_texture.texture().name());
            let texture_data = create_texture_data(model_name, &buffers, &normal_texture.texture(), TextureType::Normal, &images, add_images, add_pixes, loaded_images);
            textures.push(texture_data);
        }
        None => {
            match pbr_metallic_roughness.base_color_texture() {
                Some(base_color_texture) => {                
                    info!("Found base color texture, index {}, {:?}", base_color_texture.texture().index(), base_color_texture.texture().name());
                    let texture_data = create_texture_data(model_name, &buffers, &base_color_texture.texture(), TextureType::Normal, &images, add_images, add_pixes, loaded_images);
                    textures.push(texture_data);
                }
                None => {
                    warn!("No normal_texture");
                    let texture_data = create_texture_data_from_file(model_name, "DarkKnight_Eyes_normal_map", TextureType::Normal, add_images, add_pixes, loaded_images);
                    textures.push(texture_data);
                }
            }
        }
    }
    match material.occlusion_texture() {
        Some(occlusion_texture) => {
            info!("Found occlusion texture, index {}, {:?}", occlusion_texture.texture().index(), occlusion_texture.texture().name());
            let texture_data = create_texture_data(model_name, &buffers, &occlusion_texture.texture(), TextureType::Occlusion, &images, add_images, add_pixes, loaded_images);
            textures.push(texture_data);
        }
        None => {}
    }
    match material.emissive_texture() {
        Some(emmisive_texture) => {
            info!("Found emmisive texture, index {}", emmisive_texture.texture().index());
            let texture_data = create_texture_data(model_name, &buffers, &emmisive_texture.texture(), TextureType::Emissive, &images, add_images, add_pixes, loaded_images);
            textures.push(texture_data);
        }
        None => {}
    }        

    info!("Processed material: index {}, {:?}", material_index, material.name());

    //info!("Base color factor {:?}", pbr_metallic_roughness.base_color_factor());
    //info!("Metallic factor {}", pbr_metallic_roughness.metallic_factor());
    //info!("Roughness factor {}", pbr_metallic_roughness.roughness_factor());
    //info!("Emissive factor {:?}",  material.emissive_factor());

    MaterialData {
        index: material_index,
        name: material.name().map(|r| r.to_owned()),
        alpa_mode: match material.alpha_mode() {
            gltf::material::AlphaMode::Blend => AlphaMode::Blend,
            gltf::material::AlphaMode::Mask => AlphaMode::Mask,
            gltf::material::AlphaMode::Opaque => AlphaMode::Opaque
        },
        textures,
        base_color_factor: pbr_metallic_roughness.base_color_factor(),
        metallic_factor: pbr_metallic_roughness.metallic_factor(),
        roughness_factor: pbr_metallic_roughness.roughness_factor(),
        emissive_factor: material.emissive_factor()
    }
}

pub fn create_texture_data(model_dir: &str, buffers: &Vec<gltf::buffer::Data>, texture: &gltf::Texture<'_>, texture_type: TextureType, images: &Vec<gltf::image::Data>, add_images: bool, add_pixes: bool, loaded_images: &mut Vec<DynamicImage>) -> TextureData {        
    let image = texture.source();
    info!("Creating texture, image {:?}, image index {}, {}", image.name(), image.index(), model_dir);
    let image_gltf_data = &images[image.index()];
    info!("format {:?}", image_gltf_data.format);

    let sampler = texture.sampler();
    info!("{:?} {:?} {:?} {:?}", sampler.mag_filter(), sampler.min_filter(), sampler.wrap_s(), sampler.wrap_t());

    let image_format = match image_gltf_data.format {
        Format::R8G8B8 => ImageFormat::R8G8B8,
        Format::R8G8B8A8 => ImageFormat::R8G8B8A8,
        Format::R16G16B16 => ImageFormat::R16G16B16,
        Format::R16G16B16A16 => ImageFormat::R16G16B16A16,
        _ => ImageFormat::Other
    };

    let loaded_image_index = loaded_images.len();

    /*
    let mut name = image.name().expect("Empty image name");

    if name.contains("-") {
        name = name.split("-").next().expect("Empty split");
    }

    loaded_images.push(ImageReader::open("../models/".to_owned() + model_dir + "/textures-3/" + name + ".png").expect("Failed to open image").decode().expect("Failed to decode image"));
    */
        
    if add_images {
        match image.source() {
            gltf::image::Source::View { view, mime_type } => {
                info!("Mime type {}, buffer index {}", mime_type, view.buffer().index());

                let start = view.offset();
                let end = start + view.length();
                
                let image_slice = &buffers[view.buffer().index()].0[start..end];

                loaded_images.push(image::load_from_memory(image_slice).expect("Failed to load image from buffer"));                    
            }
            gltf::image::Source::Uri { uri, mime_type } => {
                info!("Mime type {:?}, uri {}", mime_type, uri);

                loaded_images.push(ImageReader::open("../models/".to_owned() + model_dir + "/" + uri).expect("Failed to open image").decode().expect("Failed to decode image"));
            }
        }
    }   

    TextureData {
        index: texture.index(),
        name: texture.name().map(|r| r.to_owned()),
        image_index: image.index(),
        loaded_image_index,
        image_format,
        compression_format: TextureCompressionFormat::NotCompressed,
        texture_type,
        width: image_gltf_data.width,
        height: image_gltf_data.height,
        payload: match add_pixes {
            true => Some(compress_prepend_size(&image_gltf_data.pixels)),
            false => None
        }
    }
}

pub fn create_texture_data_from_file(model_dir: &str, image_name: &str, texture_type: TextureType, add_images: bool, add_pixes: bool, loaded_images: &mut Vec<DynamicImage>) -> TextureData {
    info!("Creating texture from image name {}, {}", image_name, model_dir);
    
    let image_format = match texture_type {        
        TextureType::MetallicRoughness => ImageFormat::R8G8B8A8,
        _ => ImageFormat::R8G8B8
    };

    let loaded_image_index = loaded_images.len();

    let image = ImageReader::open("../models/".to_owned() + model_dir + "/textures-3/" + image_name + ".png").expect("Failed to open image").decode().expect("Failed to decode image");

    let width = image.width();
    let height = image.height();

    let payload = match add_pixes {
        true => Some(match image_format {
            ImageFormat::R16G16B16 |
            ImageFormat::R16G16B16A16 => {
                let raw_u16_data = image.to_rgba16().into_raw();
                let texture_data = bytemuck::cast_slice(&raw_u16_data);
                compress_prepend_size(texture_data)
            }
            _ => {
                let texture_data = image.to_rgba8();
                compress_prepend_size(&texture_data)                
            }
        }),
        false => None
    };

    loaded_images.push(image);

    TextureData {
        index: loaded_image_index,
        name: Some(image_name.to_owned()),
        image_index: loaded_image_index,
        loaded_image_index,
        image_format,
        compression_format: TextureCompressionFormat::NotCompressed,
        texture_type,
        width,
        height,
        payload
    }
}
