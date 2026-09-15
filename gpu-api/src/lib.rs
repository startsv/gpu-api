pub mod frame_counter;
pub mod texture;
pub mod camera;
pub mod pipeline {    
    pub mod solid_quad_pipeline;
    pub mod gradient_quad_pipeline;
    pub mod line_pipeline;
    pub mod image_pipeline;
    pub mod model_pipeline;
    pub mod model_bindless_pipeline;
    pub mod clear_commands_pipeline;
    pub mod surface_bindless_pipeline;
}
