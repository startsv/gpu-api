use glam::{Mat4, Vec2, Vec3, Vec4, Vec4Swizzles};
use gpu_api_relay::model_bindless_data::CameraUniform;

pub struct Camera {    
    pub angle_y: f32,
    pub angle_xz: f32,
    pub dist: f32,    
    pub position: Vec3,
    pub focus_point: Vec3,
    pub projection_source: Mat4,
    pub view: Mat4,    
    pub view_proj: Mat4, 
}

impl Camera {
    pub fn get_uniform(&self) -> CameraUniform {
        CameraUniform {
            camera_position: self.position.to_array(),
            padding: 0,
            view: self.view,
            projection: self.view_proj,
            frustum: gpu_api_relay::frustum::Frustum::to_uniform(self.view_proj),
        }
    }

    pub fn update(&mut self, width: f32, height: f32) {
        let (camera_position, focus_point, projection_source, view, projection) = generate_projection(width, height, self.focus_point.x, self.focus_point.y, self.focus_point.z, self.angle_xz, self.angle_y, self.dist);

        self.position = camera_position;
        self.focus_point = focus_point;
        self.projection_source = projection_source;
        self.view = view;        
        self.view_proj = projection;
    }

    pub fn screen_to_ray(&self, screen_pos: Vec2) -> (Vec3, Vec3) {        
        let ndc_x = screen_pos.x * 2.0 - 1.0;
        let ndc_y = (1.0 - screen_pos.y) * 2.0 - 1.0;
        
        let ndc_near = Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let ndc_far  = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        
        let inv_pv = self.view_proj.inverse(); 
        
        let world_near = inv_pv * ndc_near;
        let world_near = world_near.xyz() / world_near.w;
        
        let world_far = inv_pv * ndc_far;
        let world_far = world_far.xyz() / world_far.w;
        
        let ray_origin = world_near; 
                
        let ray_dir = (world_far - world_near).normalize();

        (ray_origin, ray_dir)
    }
}

pub fn create_camera(width: f32, height: f32, angle_xz: f32, angle_y: f32, dist: f32, focus_point_x: f32, focus_point_y: f32, focus_point_z: f32) -> Camera {    
    let (camera_position, focus_point, projection_source, view, projection) = generate_projection(width, height, focus_point_x, focus_point_y, focus_point_z, angle_xz, angle_y, dist);

    Camera {        
        angle_xz,
        angle_y,
        dist,
        position: camera_position,
        focus_point,
        projection_source,
        view,        
        view_proj: projection        
    }
}

pub fn generate_projection(
    width: f32,
    height: f32,
    focus_point_x: f32,
    focus_point_y: f32,
    focus_point_z: f32,
    angle_xz: f32, // Pitch
    angle_y: f32,  // Yaw
    dist: f32,
) -> (Vec3, Vec3, Mat4, Mat4, Mat4) {
    let aspect_ratio = if height > 0.0 { width / height } else { 1.0 };
        
    let projection = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect_ratio, 0.1, 1000.0);
    
    let max_pitch = std::f32::consts::FRAC_PI_2 - 0.001;
    let clamped_pitch = angle_xz.clamp(-max_pitch, max_pitch);

    let camera_position = Vec3::new(
        clamped_pitch.cos() * angle_y.sin() * dist + focus_point_x,
        clamped_pitch.sin() * dist + focus_point_y,
        clamped_pitch.cos() * angle_y.cos() * dist + focus_point_z,
    );

    let focus_point = Vec3::new(focus_point_x, focus_point_y, focus_point_z);
    
    let view = Mat4::look_at_rh(
        camera_position,
        focus_point,
        Vec3::Y,
    );    
    
    let view_proj = projection * view;
    
    (camera_position, focus_point, projection, view, view_proj)
}
