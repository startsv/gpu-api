struct CameraUniform {
    camera_position: vec3<f32>,
    padding: u32,    
    view_proj: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
};

struct LineInstance {
    @location(0) color: vec4<f32>,
    @location(1) width: f32,
    @location(2) start_pos: vec3<f32>,
    @location(3) end_pos: vec3<f32>,    
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VertexInput {
    @builtin(vertex_index) vertex_idx: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(linear) line_coord: f32,
};

@vertex
fn vs_main(
    in: VertexInput,
    instance: LineInstance
) -> VertexOutput {
    var out: VertexOutput;
    out.color = instance.color;
    
    var clip_start = camera.view_proj * vec4<f32>(instance.start_pos, 1.0);
    var clip_end   = camera.view_proj * vec4<f32>(instance.end_pos, 1.0);
    
    let ndc_start = clip_start.xy / clip_start.w;
    let ndc_end   = clip_end.xy / clip_end.w;

    let screen_size = vec2<f32>(1920.0, 1080.0);
    
    let screen_start = (ndc_start + 1.0) * 0.5 * screen_size;
    let screen_end   = (ndc_end + 1.0) * 0.5 * screen_size;
    
    let line_dir = normalize(screen_end - screen_start);
    let line_normal = vec2<f32>(-line_dir.y, line_dir.x);
    
    var u: f32 = 0.0;
    var v: f32 = 0.0;

    switch (in.vertex_idx) {
        case 0u: { u = 0.0; v = -1.0; }
        case 1u: { u = 1.0; v = -1.0; }
        case 2u: { u = 0.0; v =  1.0; }
        case 3u: { u = 0.0; v =  1.0; }
        case 4u: { u = 1.0; v = -1.0; }
        case 5u: { u = 1.0; v =  1.0; }
        default: { u = 0.0; v =  0.0; }
    }

    out.line_coord = v;
    
    let base_screen = mix(screen_start, screen_end, u);
    let base_clip   = mix(clip_start, clip_end, u);
    
    let total_width = instance.width + 2.0; 
    let offset_screen = line_normal * v * (total_width * 0.5);
    
    let offset_ndc = (offset_screen / screen_size) * 2.0;
        
    out.position = vec4<f32>(base_clip.xy + offset_ndc * base_clip.w, base_clip.z, base_clip.w);
    
    out.line_coord = v * (total_width / instance.width);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {    
    let distance = abs(in.line_coord);    
    let edge_width = fwidth(distance);    
    let alpha = 1.0 - smoothstep(1.0 - edge_width, 1.0, distance);
    
    if (alpha <= 0.0) {
        discard;
    }

    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
