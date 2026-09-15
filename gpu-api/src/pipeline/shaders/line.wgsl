// Camera
struct CameraUniform {
    camera_position: vec3<f32>,
    padding: u32,
    view: mat4x4<f32>,
    projection: mat4x4<f32>
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) color: vec4<f32>,
    @location(1) pos: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
        
    let local_pos = vec4<f32>(in.pos, 1.0);    
    // WGSL matrix order
    out.clip_pos = camera.projection * local_pos;

    out.color = in.color;    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
