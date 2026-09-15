@group(1) @binding(0)
var image_texture: texture_2d<f32>;
@group(1) @binding(1)
var image_sampler: sampler;

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,    
    @location(0) pos: vec2<f32>,
    @location(1) scale: vec2<f32>,
    @location(2) border_color: vec4<f32>,
    @location(3) border_radius: vec4<f32>,
    @location(4) border_width: f32,
    @location(5) shadow_color: vec4<f32>,
    @location(6) shadow_offset: vec2<f32>,
    @location(7) shadow_blur_radius: f32,
    @location(8) snap: u32,
    @location(9) clip: vec4<f32>    
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texture_coordinates: vec2<f32>,
    @location(1) border_color: vec4<f32>,
    @location(2) pos: vec2<f32>,
    @location(3) scale: vec2<f32>,
    @location(4) border_radius: vec4<f32>,
    @location(5) border_width: f32,
    @location(6) shadow_color: vec4<f32>,
    @location(7) shadow_offset: vec2<f32>,
    @location(8) shadow_blur_radius: f32,
    @location(9) clip: vec4<f32>,    
}

@vertex
fn solid_vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    var bounds = shadow_expanded_bounds(input.pos, input.scale, input.shadow_offset, input.shadow_blur_radius) * globals.scale;
    var pos: vec2<f32> = bounds.xy;
    var scale: vec2<f32> = bounds.zw;

    var pos_snap = vec2<f32>(0.0, 0.0);
    var scale_snap = vec2<f32>(0.0, 0.0);

    if bool(input.snap) {
        pos_snap = round(pos + vec2(0.001, 0.001)) - pos;
        scale_snap = round(pos + scale + vec2(0.001, 0.001)) - pos - pos_snap - scale;
    }

    var min_border_radius = min(input.scale.x, input.scale.y) * 0.5;
    var border_radius: vec4<f32> = vec4<f32>(
        min(input.border_radius.x, min_border_radius),
        min(input.border_radius.y, min_border_radius),
        min(input.border_radius.z, min_border_radius),
        min(input.border_radius.w, min_border_radius)
    );

    var transform: mat4x4<f32> = mat4x4<f32>(
        vec4<f32>(scale.x + scale_snap.x + 1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, scale.y + scale_snap.y + 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(pos + pos_snap - vec2<f32>(0.5, 0.5), 0.0, 1.0)
    );

    var texture_coordinates = vec2<f32>(0.0, 0.0);

    if (input.vertex_index == 0 || input.vertex_index == 5) {
        texture_coordinates = vec2<f32>(1.0, 1.0);
    } else 
    if (input.vertex_index == 1) {
        texture_coordinates = vec2<f32>(1.0, 0.0);
    } else {

    } if (input.vertex_index == 4) {
        texture_coordinates = vec2<f32>(0.0, 1.0);
    }

    out.position = globals.transform * transform * vec4<f32>(vertex_position(input.vertex_index), 0.0, 1.0);
    out.texture_coordinates = texture_coordinates;
    out.border_color = premultiply(input.border_color);
    out.pos = input.pos * globals.scale + pos_snap;
    out.scale = input.scale * globals.scale + scale_snap;
    out.border_radius = border_radius * globals.scale;
    out.border_width = input.border_width * globals.scale;
    out.shadow_color = premultiply(input.shadow_color);
    out.shadow_offset = input.shadow_offset * globals.scale;
    out.shadow_blur_radius = input.shadow_blur_radius * globals.scale;
    out.clip = input.clip;    

    return out;
}

@fragment
fn solid_fs_main(
    input: VertexOutput
) -> @location(0) vec4<f32> {
    if (
        input.position[0] < input.clip[0] ||
        input.position[0] > input.clip[2] ||
        input.position[1] < input.clip[1] ||
        input.position[1] > input.clip[3]
    ) {
        discard;
    }

    var mixed_color: vec4<f32> = textureSample(image_texture, image_sampler, input.texture_coordinates);    

    let local_pixel_pos = input.position.xy - globals.viewport_offset;

    var dist = rounded_box_sdf(
        -(local_pixel_pos - input.pos - input.scale * 0.5) * 2.0,
        input.scale,
        input.border_radius * 2.0
    ) / 2.0;

    if (input.border_width > 0.0) {
        mixed_color = mix(
            mixed_color,
            input.border_color,
            clamp(0.5 + dist + input.border_width, 0.0, 1.0)
        );
    }

    var quad_alpha: f32 = clamp(0.5-dist, 0.0, 1.0);

    let quad_color = mixed_color * quad_alpha;

    if input.shadow_color.a > 0.0 {
        return mix_shadow(
            quad_color,
            quad_alpha,
            input.position.xy,
            input.pos,
            input.scale,
            input.border_radius,
            input.shadow_color,
            input.shadow_offset,
            input.shadow_blur_radius
        );
    }

    return quad_color;
}
