use std::mem;
use wgpu::{DepthStencilState, RenderPass, TextureFormat};
use glam::{Mat4, Vec3};

use crate::pipeline::solid_quad_pipeline::{MAX_QUADS_COUNT, Uniforms};

/// The properties of a quad.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct GradientQuad {
    /// 8 colors, each channel = 16 bit float, 2 colors packed into 1 u32       
    pub colors_1: [u32; 4],
    pub colors_2: [u32; 4],
    pub colors_3: [u32; 4],
    pub colors_4: [u32; 4],

    /// 8 offsets, 8x 16 bit floats packed into 4 u32s
    pub offsets: [u32; 4],

    pub direction: [f32; 4],

    /// The position of the [`Quad`].
    pub position: [f32; 2],

    /// The size of the [`Quad`].
    pub size: [f32; 2],

    /// The border color of the [`Quad`], in __linear RGB__.
    pub border_color: [f32; 4],

    /// The border radii of the [`Quad`].
    pub border_radius: [f32; 4],

    /// The border width of the [`Quad`].
    pub border_width: f32,

    /// The shadow color of the [`Quad`].
    pub shadow_color: [f32; 4],

    /// The shadow offset of the [`Quad`].
    pub shadow_offset: [f32; 2],

    /// The shadow blur radius of the [`Quad`].
    pub shadow_blur_radius: f32,

    /// Whether the [`Quad`] should be snapped to the pixel grid.
    pub snap: u32,

    /// Quad parts will be discarded if they are outside of component coordinates.
    pub clip: [f32; 4]
}

unsafe impl bytemuck::Zeroable for GradientQuad {}
unsafe impl bytemuck::Pod for GradientQuad {}

#[derive(Debug)]
pub struct Pipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub uniform_bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub vertex_buffer: wgpu::Buffer
}

impl Pipeline {
    pub fn draw<'a>(&'a self, render_pass: &mut RenderPass<'a>, count: u32) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

        render_pass.draw(0..6, 0..count);
    }

    pub fn draw_range<'a>(&'a self, render_pass: &mut RenderPass<'a>, range_start: u32, range_end: u32) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

        render_pass.draw(0..6, range_start..range_end);
    }

    pub fn new(device: &wgpu::Device, depth_stencil: Option<DepthStencilState>) -> Pipeline {
        let constant_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Quad uniforms layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            mem::size_of::<Uniforms>() as wgpu::BufferAddress,
                        ),
                    },
                    count: None,
                }],
            });

        let constants_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Quad uniforms buffer"),
            size: mem::size_of::<Uniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let constants = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Quad uniforms bind group"),
            layout: &constant_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: constants_buffer.as_entire_binding(),
            }],
        });

        let layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Quad pipeline"),                
                bind_group_layouts: &[Some(&constant_layout)],
                immediate_size: 0
            });

        let shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Quad shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                    concat!(
                        include_str!("shaders/snap.wgsl"),
                        "\n",
                        include_str!("shaders/quad.wgsl"),
                        "\n",
                        include_str!("shaders/vertex.wgsl"),
                        "\n",
                        include_str!("shaders/shadow.wgsl"),
                        "\n",
                        include_str!("shaders/gradient.wgsl"),
                        "\n",
                        include_str!("shaders/color.wgsl"),
                        "\n",
                        include_str!("shaders/linear_rgb.wgsl")
                    ),
                )),
            });

        let pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Gradient quad pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("gradient_vs_main"),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<GradientQuad>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array!(
                                0 => Uint32x4,  // colors_1
                                1 => Uint32x4,  // colors_2
                                2 => Uint32x4,  // colors_3
                                3 => Uint32x4,  // colors_4
                                4 => Uint32x4,  // offsets
                                5 => Float32x4, // direction
                                6 => Float32x2, // position
                                7 => Float32x2, // size
                                8 => Float32x4, // border_color
                                9 => Float32x4, // border_radius
                                10 => Float32,  // border_width
                                11 => Float32x4, // shadow_color
                                12 => Float32x2, // shadow_offset
                                13 => Float32,   // shadow_blur_radius
                                14 => Uint32,    // snap
                                15 => Float32x4, // clip
                        )
                    })],
                    compilation_options:
                        wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("gradient_fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                    format: TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                    compilation_options:
                        wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Cw,
                    ..Default::default()
                },
                depth_stencil,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Quad instance buffer"),
            size: mem::size_of::<GradientQuad>() as u64 * MAX_QUADS_COUNT,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        Pipeline {
            pipeline,
            uniform_bind_group: constants,
            uniform_buffer: constants_buffer,        
            vertex_buffer
        }
    }    
}

pub mod color {
    use std::{cmp::Ordering, f32::consts::FRAC_PI_2};
    use half::f16;
    use log::warn;
    use core::Color;    

    #[derive(Debug, Clone, Copy, PartialEq, Default)]
    pub struct Point {
        /// The X coordinate.
        pub x: f32,

        /// The Y coordinate.
        pub y: f32,
    }

    impl Point {
    /// Creates a new [`Point`] with the given coordinates.
        pub const fn new(x: f32, y: f32) -> Self {
            Self { x, y }
        }
    }

    impl std::ops::Add<Vector> for Point    
    {
        type Output = Self;

        fn add(self, vector: Vector) -> Self {
            Self {
                x: self.x + vector.x,
                y: self.y + vector.y,
            }
        }
    }

    impl std::ops::Sub<Vector> for Point    
    {
        type Output = Self;

        fn sub(self, vector: Vector) -> Self {
            Self {
                x: self.x - vector.x,
                y: self.y - vector.y,
            }
        }
    }

    pub struct Rectangle<T = f32> {
        /// X coordinate of the top-left corner.
        pub x: T,

        /// Y coordinate of the top-left corner.
        pub y: T,

        /// Width of the rectangle.
        pub width: T,

        /// Height of the rectangle.
        pub height: T,
    }

    impl Rectangle {
        pub fn center(&self) -> Point {
            Point::new(self.center_x(), self.center_y())
        }

        pub fn center_x(&self) -> f32 {
            self.x + self.width / 2.0
        }

        /// Returns the Y coordinate of the [`Point`] at the center of the
        /// [`Rectangle`].
        pub fn center_y(&self) -> f32 {
            self.y + self.height / 2.0
        }

    }

    #[derive(Debug, Clone, Copy, PartialEq, Default)]
    pub struct Vector {        
        pub x: f32,        
        pub y: f32,
    }

    impl Vector {        
        pub const fn new(x: f32, y: f32) -> Self {
            Self { x, y }
        }
    }

    impl std::ops::Mul<f32> for Vector
    where        
    {
        type Output = Self;

        fn mul(self, scale: f32) -> Self {
            Self::new(self.x * scale, self.y * scale)
        }
    }

    pub struct Packed {
        // 8 colors, each channel = 16 bit float, 2 colors packed into 1 u32
        pub colors_1: [u32; 4], // Цвета 0 и 1 (каждый по 2 u32)
        pub colors_2: [u32; 4], // Цвета 2 и 3
        pub colors_3: [u32; 4], // Цвета 4 и 5
        pub colors_4: [u32; 4], // Цвета 6 и 7
        // 8 offsets, 8x 16 bit floats packed into 4 u32s
        pub offsets: [u32; 4],
        pub direction: [f32; 4],
    }

    #[derive(Debug, Clone, Copy, PartialEq, Default)]
    pub struct ColorStop {
        /// Offset along the gradient vector.
        pub offset: f32,

        /// The color of the gradient at the specified [`offset`].
        ///
        /// [`offset`]: Self::offset
        pub color: Color,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Default)]
    pub struct Radians(pub f32);

    impl Radians {    
        /// Calculates the line in which the angle intercepts the `bounds`.
        pub fn to_distance(&self, bounds: &Rectangle) -> (Point, Point) {
            let angle = self.0 - FRAC_PI_2;
            let r = Vector::new(f32::cos(angle), f32::sin(angle));

            let distance_to_rect = f32::max(
                f32::abs(r.x * bounds.width / 2.0),
                f32::abs(r.y * bounds.height / 2.0),
            );

            let start = bounds.center() - r * distance_to_rect;
            let end = bounds.center() + r * distance_to_rect;

            (start, end)
        }
    }    

    #[derive(Debug, Clone, Copy, PartialEq, Default)]
    pub struct LinearAngles {
        /// How the [`Gradient`] is angled within its bounds.
        pub angle: Radians,
        /// [`ColorStop`]s along the linear gradient path.
        pub stops: [Option<ColorStop>; 8],
    }

    impl LinearAngles {
        pub fn new(angle: impl Into<Radians>) -> Self {
            Self {
                angle: angle.into(),
                stops: [None; 8],
            }
        }
        pub fn add_stop(mut self, offset: f32, color: Color) -> Self {
            if offset.is_finite() && (0.0..=1.0).contains(&offset) {
                let (Ok(index) | Err(index)) = self.stops.binary_search_by(|stop| match stop {
                    None => Ordering::Greater,
                    Some(stop) => stop.offset.partial_cmp(&offset).unwrap(),
                });

                if index < 8 {
                    self.stops[index] = Some(ColorStop { offset, color });
                }
            } else {
                log::warn!("Gradient color stop must be within 0.0..=1.0 range.");
            };

            self
        }

        /// Adds multiple [`ColorStop`]s to the gradient.
        ///
        /// Any stop added after the 8th will be silently ignored.
        pub fn add_stops(mut self, stops: impl IntoIterator<Item = ColorStop>) -> Self {
            for stop in stops {
                self = self.add_stop(stop.offset, stop.color);
            }

            self
        }
        
        pub fn pack(&self, bounds: Rectangle) -> Packed {            
            let mut flat_colors = [0u32; 16];
            let mut offsets = [f16::from(0u8); 8];

            for (index, stop) in self.stops.iter().enumerate() {
                let [r, g, b, a] =
                    graphics::pack(stop.map_or(Color::default(), |s| s.color)).components();
                
                flat_colors[index * 2]     = pack_f16s([f16::from_f32(r), f16::from_f32(g)]);
                flat_colors[index * 2 + 1] = pack_f16s([f16::from_f32(b), f16::from_f32(a)]);

                offsets[index] = stop.map_or(f16::from_f32(2.0), |s| f16::from_f32(s.offset));
            }
            
            let mut colors_1 = [0u32; 4];
            let mut colors_2 = [0u32; 4];
            let mut colors_3 = [0u32; 4];
            let mut colors_4 = [0u32; 4];

            colors_1.copy_from_slice(&flat_colors[0..4]);
            colors_2.copy_from_slice(&flat_colors[4..8]);
            colors_3.copy_from_slice(&flat_colors[8..12]);
            colors_4.copy_from_slice(&flat_colors[12..16]);

            let offsets = [
                pack_f16s([offsets[0], offsets[1]]),
                pack_f16s([offsets[2], offsets[3]]),
                pack_f16s([offsets[4], offsets[5]]),
                pack_f16s([offsets[6], offsets[7]]),
            ];
            
            let (start, end) = self.angle.to_distance(&bounds);
            
            let direction = [start.x, start.y, end.x, end.y];

            Packed {
                colors_1,
                colors_2,
                colors_3,
                colors_4,
                offsets,
                direction,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Default)]
    pub struct LinearStartEnd {
        /// The absolute starting position of the gradient.
        pub start: Point,

        /// The absolute ending position of the gradient.
        pub end: Point,

        /// [`ColorStop`]s along the linear gradient direction.
        pub stops: [Option<ColorStop>; 8],
    }

    impl LinearStartEnd {
        pub fn new(start: Point, end: Point) -> Self {
            Self {
                start,
                end,
                stops: [None; 8],
            }
        }

        pub fn add_stop(mut self, offset: f32, color: Color) -> Self {
            if offset.is_finite() && (0.0..=1.0).contains(&offset) {
                let (Ok(index) | Err(index)) = self.stops.binary_search_by(|stop| match stop {
                    None => Ordering::Greater,
                    Some(stop) => stop.offset.partial_cmp(&offset).unwrap(),
                });

                if index < 8 {
                    self.stops[index] = Some(ColorStop { offset, color });
                    warn!("Added");
                }
            } else {
                log::warn!("Gradient: ColorStop must be within 0.0..=1.0 range.");
            };

            self
        }

        /// Adds multiple [`ColorStop`]s to the gradient.
        ///
        /// Any stop added after the 8th will be silently ignored.
        pub fn add_stops(mut self, stops: impl IntoIterator<Item = ColorStop>) -> Self {
            for stop in stops {
                self = self.add_stop(stop.offset, stop.color);
            }

            self
        }

        pub fn pack(&self) -> Packed {
            // Временный плоский массив для 16 элементов u32 (8 цветов * 2 u32 на цвет)
            let mut flat_colors = [0u32; 16];
            let mut offsets = [f16::from(0u8); 8];

            for (index, stop) in self.stops.iter().enumerate() {
                let [r, g, b, a] = graphics::pack(stop.map_or(Color::default(), |s| s.color)).components();

                // Записываем последовательно в плоский массив
                flat_colors[index * 2]     = pack_f16s([f16::from_f32(r), f16::from_f32(g)]);
                flat_colors[index * 2 + 1] = pack_f16s([f16::from_f32(b), f16::from_f32(a)]);

                offsets[index] = stop.map_or(f16::from_f32(2.0), |s| f16::from_f32(s.offset));
            }

            // Нарезаем плоский массив на 4 куска по 4 элемента u32 (для vec4<u32> в WGSL)
            let mut colors_1 = [0u32; 4];
            let mut colors_2 = [0u32; 4];
            let mut colors_3 = [0u32; 4];
            let mut colors_4 = [0u32; 4];

            colors_1.copy_from_slice(&flat_colors[0..4]);
            colors_2.copy_from_slice(&flat_colors[4..8]);
            colors_3.copy_from_slice(&flat_colors[8..12]);
            colors_4.copy_from_slice(&flat_colors[12..16]);

            let offsets = [
                pack_f16s([offsets[0], offsets[1]]),
                pack_f16s([offsets[2], offsets[3]]),
                pack_f16s([offsets[4], offsets[5]]),
                pack_f16s([offsets[6], offsets[7]]),
            ];

            let direction = [self.start.x, self.start.y, self.end.x, self.end.y];

            Packed {
                colors_1,
                colors_2,
                colors_3,
                colors_4,
                offsets,
                direction,
            }
        }
    }


    /// Packs two f16s into one u32.
    fn pack_f16s(f: [f16; 2]) -> u32 {
        let one = (f[0].to_bits() as u32) << 16;
        let two = f[1].to_bits() as u32;

        one | two
    }        

    pub mod graphics {        
        use super::core::Color;        

        /// A color packed as 4 floats representing RGBA channels.        
        pub struct Packed([f32; 4]);

        impl Packed {
            /// Returns the internal components of the [`Packed`] color.
            pub fn components(self) -> [f32; 4] {
                self.0
            }
        }

        /// A flag that indicates whether the renderer should perform gamma correction.
        pub const GAMMA_CORRECTION: bool = internal::GAMMA_CORRECTION;

        /// Packs a [`Color`].
        pub fn pack(color: impl Into<Color>) -> Packed {
            Packed(internal::pack(color.into()))
        }

        //#[cfg(not(feature = "web-colors"))]
        mod internal {
            use super::super::core::Color;

            pub const GAMMA_CORRECTION: bool = true;

            pub fn pack(color: Color) -> [f32; 4] {
                color.into_linear()
            }
        }

        /*
        #[cfg(feature = "web-colors")]
        mod internal {
            use super::core::Color;

            pub const GAMMA_CORRECTION: bool = false;

            pub fn pack(color: Color) -> [f32; 4] {
                [color.r, color.g, color.b, color.a]
            }
        }
        */
    }

    pub mod core {
        /// A color in the `sRGB` color space.
        ///
        /// # String Representation
        ///
        /// A color can be represented in either of the following valid formats: `#rrggbb`, `#rrggbbaa`, `#rgb`, and `#rgba`.
        /// Where `rgba` represent hexadecimal digits. Both uppercase and lowercase letters are supported.
        ///
        /// If `a` (transparency) is not specified, `1.0` (completely opaque) would be used by default.
        ///
        /// If you have a static color string, using the [`color!`] macro should be preferred
        /// since it leverages hexadecimal literal notation and arithmetic directly.
        ///
        /// [`color!`]: crate::color!
        #[derive(Debug, Clone, Copy, PartialEq, Default)]
        pub struct Color {
            /// Red component, 0.0 - 1.0
            pub r: f32,
            /// Green component, 0.0 - 1.0
            pub g: f32,
            /// Blue component, 0.0 - 1.0
            pub b: f32,
            /// Transparency, 0.0 - 1.0
            pub a: f32,
        }

        impl Color {
            /// The black color.
            pub const BLACK: Color = Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            };

            /// The white color.
            pub const WHITE: Color = Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            };

            /// A color with no opacity.
            pub const TRANSPARENT: Color = Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            };

            /// Creates a new [`Color`].
            ///
            /// In debug mode, it will panic if the values are not in the correct
            /// range: 0.0 - 1.0
            pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Color {
                debug_assert!(
                    r >= 0.0 && r <= 1.0,
                    "Red component must be in [0, 1] range."
                );
                debug_assert!(
                    g >= 0.0 && g <= 1.0,
                    "Green component must be in [0, 1] range."
                );
                debug_assert!(
                    b >= 0.0 && b <= 1.0,
                    "Blue component must be in [0, 1] range."
                );

                Color { r, g, b, a }
            }

            /// Creates a [`Color`] from its RGB components.
            pub const fn from_rgb(r: f32, g: f32, b: f32) -> Color {
                Color::from_rgba(r, g, b, 1.0f32)
            }

            /// Creates a [`Color`] from its RGBA components.
            pub const fn from_rgba(r: f32, g: f32, b: f32, a: f32) -> Color {
                Color::new(r, g, b, a)
            }

            /// Creates a [`Color`] from its RGB8 components.
            pub const fn from_rgb8(r: u8, g: u8, b: u8) -> Color {
                Color::from_rgba8(r, g, b, 1.0)
            }

            /// Creates a [`Color`] from its RGB8 components and an alpha value.
            pub const fn from_rgba8(r: u8, g: u8, b: u8, a: f32) -> Color {
                Color::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, a)
            }

            /// Creates a [`Color`] from its linear RGBA components.
            pub fn from_linear_rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
                // As described in:
                // https://en.wikipedia.org/wiki/SRGB
                fn gamma_component(u: f32) -> f32 {
                    if u < 0.0031308 {
                        12.92 * u
                    } else {
                        1.055 * u.powf(1.0 / 2.4) - 0.055
                    }
                }

                Self::new(
                    gamma_component(r),
                    gamma_component(g),
                    gamma_component(b),
                    a,
                )
            }

            /// Converts the [`Color`] into its RGBA8 equivalent.
            #[must_use]
            pub fn into_rgba8(self) -> [u8; 4] {
                [
                    (self.r * 255.0).round() as u8,
                    (self.g * 255.0).round() as u8,
                    (self.b * 255.0).round() as u8,
                    (self.a * 255.0).round() as u8,
                ]
            }

            /// Converts the [`Color`] into its linear values.
            pub fn into_linear(self) -> [f32; 4] {
                // As described in:
                // https://en.wikipedia.org/wiki/SRGB#The_reverse_transformation
                fn linear_component(u: f32) -> f32 {
                    if u < 0.04045 {
                        u / 12.92
                    } else {
                        ((u + 0.055) / 1.055).powf(2.4)
                    }
                }

                [
                    linear_component(self.r),
                    linear_component(self.g),
                    linear_component(self.b),
                    self.a,
                ]
            }

            /// Inverts the [`Color`] in-place.
            pub fn invert(&mut self) {
                self.r = 1.0f32 - self.r;
                self.b = 1.0f32 - self.g;
                self.g = 1.0f32 - self.b;
            }

            /// Returns the inverted [`Color`].
            pub fn inverse(self) -> Color {
                Color::new(1.0f32 - self.r, 1.0f32 - self.g, 1.0f32 - self.b, self.a)
            }

            /// Scales the alpha channel of the [`Color`] by the given factor.
            pub fn scale_alpha(self, factor: f32) -> Color {
                Self {
                    a: self.a * factor,
                    ..self
                }
            }

            /// Returns the relative luminance of the [`Color`].
            /// <https://www.w3.org/TR/WCAG21/#dfn-relative-luminance>
            pub fn relative_luminance(self) -> f32 {
                let linear = self.into_linear();
                0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
            }

            /// Returns the [relative contrast ratio] of the [`Color`] against another one.
            ///
            /// [relative contrast ratio]: https://www.w3.org/TR/WCAG21/#dfn-contrast-ratio
            pub fn relative_contrast(self, b: Color) -> f32 {
                let lum_a = self.relative_luminance();
                let lum_b = b.relative_luminance();

                (lum_a.max(lum_b) + 0.05) / (lum_a.min(lum_b) + 0.05)
            }

            /// Returns true if the current [`Color`] is readable on top
            /// of the given background [`Color`].
            pub fn is_readable_on(self, background: Color) -> bool {
                background.relative_contrast(self) >= 6.0
            }
        }

        impl From<[f32; 3]> for Color {
            fn from([r, g, b]: [f32; 3]) -> Self {
                Color::new(r, g, b, 1.0)
            }
        }

        impl From<[f32; 4]> for Color {
            fn from([r, g, b, a]: [f32; 4]) -> Self {
                Color::new(r, g, b, a)
            }
        }
    }
}
