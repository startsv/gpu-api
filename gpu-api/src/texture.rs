use gpu_api_dto::{ImageFormat, TextureType};
use image::{GenericImageView, ImageError};
use wgpu::{util::DeviceExt, Device, Sampler};

pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView    
}

impl Texture {
    pub fn from_ktx2(device: &wgpu::Device, queue: &wgpu::Queue, texture_data: &[u8], width: u32, height: u32, label: Option<&str>) -> Result<Self, ImageError> {
        let reader = ktx2::Reader::new(texture_data).expect("Failed to create ktx2 reader");    

        let mut data = Vec::with_capacity(reader.data().len());
        for level in reader.levels() {
            data.extend_from_slice(level.data);
        }

        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {                
                label,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B4x4, channel: wgpu::AstcChannel::UnormSrgb },
                //format: wgpu::TextureFormat::Bc7RgbaUnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[]                
            },
            wgpu::util::TextureDataOrder::MipMajor,
            &data
        );
      
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());        
        
        Ok(Texture { texture, view })
    }    

    pub fn from_image(device: &wgpu::Device, queue: &wgpu::Queue, img: &image::DynamicImage, image_format: &ImageFormat, is_srgb: bool, label: Option<&str>) -> Result<Self, ImageError> {
        let dimensions = img.dimensions();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let format = match image_format {
            ImageFormat::R8G8B8 |                
            ImageFormat::R8G8B8A8 |
            ImageFormat::Other => {
                match is_srgb {
                    true => wgpu::TextureFormat::Rgba8Unorm,
                    false => wgpu::TextureFormat::Rgba8UnormSrgb
                }
            }
            ImageFormat::R16G16B16 => wgpu::TextureFormat::Rgba16Unorm,
            ImageFormat::R16G16B16A16 => wgpu::TextureFormat::Rgba16Unorm
        };
        
        let texture = device.create_texture(
            &wgpu::TextureDescriptor {                
                label,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[]                
            }
        );        

        match image_format {
            ImageFormat::R16G16B16 |
            ImageFormat::R16G16B16A16 => {
                let raw_u16_data = img.to_rgba16().into_raw();
                let texture_data = bytemuck::cast_slice(&raw_u16_data);

                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        aspect: wgpu::TextureAspect::All,
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                    },            
                    &texture_data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(8 * dimensions.0),
                        rows_per_image: Some(dimensions.1)
                    },
                    size
                );
            }
            _ => {
                let texture_data = img.to_rgba8();

                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        aspect: wgpu::TextureAspect::All,
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                    },            
                    &texture_data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * dimensions.0),
                        rows_per_image: Some(dimensions.1)
                    },
                    size,
                );
            }
        };        
        
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        Ok(Texture { texture, view })
    }

    pub fn from_image_to_rgba8(device: &wgpu::Device, queue: &wgpu::Queue, img: &image::DynamicImage, is_srgb: bool, label: Option<&str>) -> Result<Self, ImageError> {
        let dimensions = img.dimensions();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let format = match is_srgb {
            true => wgpu::TextureFormat::Rgba8Unorm,
            false => wgpu::TextureFormat::Rgba8UnormSrgb
        };
        
        let texture = device.create_texture(
            &wgpu::TextureDescriptor {                
                label,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[]                
            }
        );        

        let texture_data = img.to_rgba8();

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },            
            &texture_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1)
            },
            size,
        );     
        
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());        
        
        Ok(Texture { texture, view })
    }
        
    pub fn create_depth_texture(device: &Device, config: &wgpu::SurfaceConfiguration, label: &str) -> Self {
        log::warn!("Creating depth texture");
        
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1
        };
        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[]
        };
        let texture = device.create_texture(&desc);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        log::warn!("Depth texture created");
        
        Texture {
            texture, 
            view
        }        
    }    
}
