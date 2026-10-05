use glam::{Mat4, Vec3, Vec4};
use bounds::aabb::Aabb;

#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub normal: Vec3,
    pub d: f32,
}

impl Plane {    
    pub fn intersects_aabb(&self, aabb: &Aabb) -> bool {        
        let mut p = aabb.min;
        if self.normal.x >= 0.0 { p.x = aabb.max.x; }
        if self.normal.y >= 0.0 { p.y = aabb.max.y; }
        if self.normal.z >= 0.0 { p.z = aabb.max.z; }
        
        (self.normal.dot(p) + self.d) >= 0.0
    }
}

pub struct Frustum {
    pub planes: [Plane; 6],
}

impl Frustum {    
    pub fn from_view_projection(vp: Mat4) -> Self {
        let row = vp.to_cols_array_2d();
        
        let mut planes = [Plane { normal: Vec3::ZERO, d: 0.0 }; 6];
        
        planes[0] = Plane { normal: Vec3::new(row[0][3] + row[0][0], row[1][3] + row[1][0], row[2][3] + row[2][0]), d: row[3][3] + row[3][0] };
        // Right
        planes[1] = Plane { normal: Vec3::new(row[0][3] - row[0][0], row[1][3] - row[1][0], row[2][3] - row[2][0]), d: row[3][3] - row[3][0] };
        // Bottom
        planes[2] = Plane { normal: Vec3::new(row[0][3] + row[0][1], row[1][3] + row[1][1], row[2][3] + row[2][1]), d: row[3][3] + row[3][1] };
        // Top
        planes[3] = Plane { normal: Vec3::new(row[0][3] - row[0][1], row[1][3] - row[1][1], row[2][3] - row[2][1]), d: row[3][3] - row[3][1] };
        // Near
        planes[4] = Plane { normal: Vec3::new(row[0][2], row[1][2], row[2][2]), d: row[3][2] };
        // Far
        planes[5] = Plane { normal: Vec3::new(row[0][3] - row[0][2], row[1][3] - row[1][2], row[2][3] - row[2][2]), d: row[3][3] - row[3][2] };
        
        for plane in &mut planes {
            let length = plane.normal.length();
            plane.normal /= length;
            plane.d /= length;
        }

        Self { planes }
    }
    
    pub fn to_uniform(vp: Mat4) -> [Vec4; 6] {
        let row = vp.to_cols_array_2d();
        
        let mut planes = [Vec4::ZERO; 6];
        
        planes[0] = Vec4::new(row[0][3] + row[0][0], row[1][3] + row[1][0], row[2][3] + row[2][0], row[3][3] + row[3][0]);        
        planes[1] = Vec4::new(row[0][3] - row[0][0], row[1][3] - row[1][0], row[2][3] - row[2][0], row[3][3] - row[3][0]);        
        planes[2] = Vec4::new(row[0][3] + row[0][1], row[1][3] + row[1][1], row[2][3] + row[2][1], row[3][3] + row[3][1]);        
        planes[3] = Vec4::new(row[0][3] - row[0][1], row[1][3] - row[1][1], row[2][3] - row[2][1], row[3][3] - row[3][1]);        
        planes[4] = Vec4::new(row[0][2], row[1][2], row[2][2], row[3][2]);        
        planes[5] = Vec4::new(row[0][3] - row[0][2], row[1][3] - row[1][2], row[2][3] - row[2][2], row[3][3] - row[3][2]);
        
        for plane in &mut planes {            
            let normal_xyz = Vec3::new(plane.x, plane.y, plane.z);
            let length = normal_xyz.length();
                        
            *plane /= length;
        }

        planes
    }
    
    pub fn contains_aabb(&self, aabb: &Aabb) -> bool {
        for plane in &self.planes {
            if !plane.intersects_aabb(aabb) {
                return false;
            }
        }
        true
    }
}
