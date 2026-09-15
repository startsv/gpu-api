use std::io::Write;
use log::*;
use image::ImageReader;
use lz4_flex::block::compress_prepend_size;
use model_load::load;
use gpu_api_dto::serialize_model_data;

fn main() {
    env_logger::init();
    process_model("box", "box", "box.glb", false);
}

fn process_model(model_name: &str, folder_name: &str, file_name: &str, is_animated: bool) {    
    let (model_data, _loaded_images) = load(model_name, &format!("../models/{}/{}", folder_name, file_name), true, false, vec![74], is_animated);
    
    //let mut texture_index = 0;        

    //for loaded_image in loaded_images.unwrap() {
        //let image_file_name = format!("{}_{}.jpg", model_name, &texture_index.to_string());
        //loaded_image.save_with_format(&image_file_name, image::ImageFormat::Jpeg).expect("Failed to save texture image");
                
        // PVRTexToolCLI -i box_0.png -ics srgb -m -f ASTC_4X4,UBN,SRGB -q astcexhaustive -o box_astc_pvr.ktx2
        // compressonatorcli -fd BC7 box_0.png box_bc7_comp.ktx2

        /*
        let _ = Command::new("PVRTexToolCLI")
            .arg("-i")            
            .arg(&image_file_name)
            .arg("-ics")
            .arg("srgb")
            .arg("-m")
            .arg("-f")
            .arg("ASTC_4X4,UBN,SRGB")
            .arg("-q")
            .arg("astcexhaustive")
            .arg("-o")
            .arg(&format!("{}_{}_astc.ktx2", model_name, &texture_index.to_string()))
            .output()
            .expect("Failed to execute ASTC command");
        
        let _ = Command::new("compressonatorcli")
            .arg("-fd")
            .arg("BC7")
            .arg(&image_file_name)
            .arg(&format!("{}_{}_bc7.ktx2", model_name, &texture_index.to_string()))
            .output()
            .expect("Failed to execute BC7 command");
        */

        //texture_index = texture_index + 1;
    //}
    
    let mut file = std::fs::File::create(&format!("{}.model", model_name)).expect("Failed to create model file");
    
    match serialize_model_data(&model_data) {
        Ok(bytes) => {
            match file.write_all(&bytes) {
                Ok(()) => {
                    info!("Done");
                }
                Err(e) => {
                    error!("{}", e);
                }
            }
            
        }
        Err(e) => {
            error!("{}", e);
        }
    }    
}

fn process_image(file_path: &str, output_name: &str) {    
    let pixels_file_name = format!("{}.pixels", output_name);

    let img = ImageReader::open(file_path).expect("Failed to open image").decode().expect("Failed to decode image");        

    let compressed = compress_prepend_size(img.into_rgba8().as_raw());

    let mut pixels_file = std::fs::File::create(&pixels_file_name).expect("Failed to create model file");
    let pixels_res = pixels_file.write_all(&compressed).expect("Failed to write pixels data to file");
    info!("{:?}", pixels_res);    
}
