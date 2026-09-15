use std::sync::Arc;
use fern::colors::{Color, ColoredLevelConfig};
use glam::{Mat4, Vec3, vec3};
use gpu_api_relay::model_bindless_data::{InstanceData, NodeData, PrimitiveMeta, SurfaceData, SurfaceCullingTask};
use log::*;
use winit::{dpi::{PhysicalPosition, PhysicalSize}, event::{ElementState, Event, MouseScrollDelta, WindowEvent}, event_loop::{ControlFlow, EventLoop}, window::Window};
use wgpu::{CurrentSurfaceTexture, DeviceDescriptor, ExperimentalFeatures, MemoryHints, RequestAdapterOptions, StoreOp};
#[cfg(target_arch = "wasm32")]
use winit::{event_loop::EventLoopProxy, platform::web::{WindowExtWebSys, EventLoopExtWebSys}};
#[cfg(not(target_arch = "wasm32"))]
use tokio::runtime::Runtime;
use gpu_api::{camera::create_camera, frame_counter::FrameCounter, pipeline::{self, image_pipeline::{self, ImageObject, ImageQuad}, line_pipeline::LineVertex, model_pipeline::{CAMERA_UNIFORM_SIZE, model::{Object, ObjectGroup}}, solid_quad_pipeline::{self, Transformation}, surface_bindless_pipeline::SurfaceBindlessResources}};
use gpu_api_dto::{AnimationComputationMode, AnimationProperty, ViewSource};
use world::world::World;

pub const TARGET_FPS: u32 = 200;
pub const FRAME_CYCLE_LENGTH_FOR_ANIMATION: usize = 200;

#[derive(Debug)]
pub enum AppEvent {
}

pub struct Layout {    
    pub size: PhysicalSize<u32>,    
    pub cursor_physical_position: Option<PhysicalPosition<f64>>    
}

async fn run() {    
    let mut window_attributes = Window::default_attributes();

    window_attributes = window_attributes
        .with_title("Test application")
        .with_inner_size(winit::dpi::LogicalSize::new(1700.0, 950.0));
        
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;        
        use winit::platform::web::WindowAttributesExtWebSys;

        let canvas = web_sys::window()
            .expect("Failed to get window")
            .document()
            .expect("Failed to get window")
            .get_element_by_id("canvas")
            .expect("Failed to get canvas element")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("Failed to cast canvas");

        let _ = canvas.set_attribute("style", "width: 800px;height: 600px;outline: none;");

        window_attributes = window_attributes.with_canvas(Some(canvas));
    }

    let event_loop: EventLoop<AppEvent> = EventLoop::with_user_event().build().expect("Failed to create event loop");    
    let window = Arc::new(event_loop.create_window(window_attributes).expect("Failed to create window"));
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::DX12,
        flags: Default::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let surface = instance.create_surface(window.clone()).expect("Failed to create surface");
    let adapter = instance    
        .request_adapter(&RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            apply_limit_buckets: false,
        })
        .await
        .expect("Failed to find an appropriate adapter");

    //let adapter_limits = adapter.limits();    
    //info!("{:#?}", adapter_limits);    
    
    let (device, queue) = adapter
        .request_device(        
            &DeviceDescriptor {
                label: None,
                //required_features: wgpu::Features::empty(),
                required_features:
                    wgpu::Features::TEXTURE_FORMAT_16BIT_NORM |
                    wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING |
                    wgpu::Features::TEXTURE_BINDING_ARRAY,                    
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits {
                        max_binding_array_elements_per_shader_stage: gpu_api::pipeline::model_bindless_pipeline::MAX_TEXTURES * 8,
                        max_binding_array_sampler_elements_per_shader_stage: gpu_api::pipeline::model_bindless_pipeline::MAX_TEXTURES * 4,
                        ..Default::default()
                    }
                },
                experimental_features: ExperimentalFeatures::disabled(),
                memory_hints: MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
        .await
        .expect("Failed to create device");    

    let size = window.inner_size();
    let scale_factor = window.scale_factor();  

    info!("{:?}, {}", size, scale_factor);   

    let mut layout = Layout {        
        size,     
        cursor_physical_position: None        
    };        

    let mut config = surface
        .get_default_config(&adapter, layout.size.width, layout.size.height)
        .expect("Surface isn't supported by the adapter.");

    config.present_mode = wgpu::PresentMode::Mailbox;
    config.format = wgpu::TextureFormat::Rgba8Unorm;
    config.view_formats.push(wgpu::TextureFormat::Rgba8UnormSrgb);    

    info!("{:#?}", config);

    surface.configure(&device, &config);
    
    let depth_stencil_state = Some(wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::Always),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default()
    });

    let surface_depth_stencil_state = Some(wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::Always),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default()
    });

    let model_depth_stencil_state = Some(wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::Less),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default()
    });

    let image_pipeline = pipeline::image_pipeline::Pipeline::new(&device, depth_stencil_state.clone());

    let angle_xz = 0.4;
    let angle_y = 1.4;
    let dist = 30.0;    

    let mut camera = create_camera(layout.size.width as f32, layout.size.height as f32, angle_xz, angle_y, dist, 0.0, 0.0, 0.0);

    let camera_uniform = camera.get_uniform();
    
    let model_pipeline = pipeline::model_pipeline::new(&device, &config, &camera_uniform, model_depth_stencil_state.clone());    
    
    let (model_data, loaded_images) = model_load::load("test", "../models/knight/knight.gltf", false, true, vec![], true);
    
    let view_source = ViewSource {
        x: 0.0,
        y: 0.0,
        z: 0.0,        
        scale_x: 3.0,
        scale_y: 3.0,
        scale_z: 3.0,
        rotation_y: 0.0
    };

    let mut init_data = pipeline::model_pipeline::model::InitData {
        vertices: Vec::new(),
        indices: Vec::new(),
        factors: Vec::new(),
        materials: Vec::new(),
        instances: Vec::new(),
        joints: Vec::new(),
        nodes: Vec::new(),
    };
    
    let position = vec3(view_source.x, view_source.y, view_source.z);
    let object = Object::new(&device, &queue, &model_pipeline, model_data, vec![view_source], loaded_images, FRAME_CYCLE_LENGTH_FOR_ANIMATION, &mut init_data);

    let surface_resources = SurfaceBindlessResources::new(&device, &queue, &camera_uniform, surface_depth_stencil_state, &init_data.materials);
    
    let mut test_world = world::world::World::new(1, vec3(32.0, 32.0, 32.0));

    test_world.add_object(position, InstanceData {
        model_matrix: object.model_instances[0].model_matrix,
        is_animated: 1,
        node_index: 0,
        joints_offset: 10 * 128,
        material_index: 0,
        primitive_index: 0,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
        aabb_min: [-5.0, -5.0, -5.0],
        _pad_aabb1: 0,
        aabb_max: [5.0, 5.0, 5.0],
        _pad_aabb2: 0,
    });

    let registered_primitives = vec![
        PrimitiveMeta {
            id: 0,
            base_vertex: 0,
            lod_index_counts: [init_data.indices.len() as u32, init_data.indices.len() as u32, init_data.indices.len() as u32],
            lod_first_indices: [0, 0, 0],
            max_global_instances: 10,
        }
    ];

    let indirect_commands = World::generate_initial_indirect_commands(&registered_primitives);

    let model_bindless_resources = pipeline::model_bindless_pipeline::ModelBindlessResources::new(&device, &queue, &camera_uniform, model_depth_stencil_state,
        registered_primitives.len(),
        &mut init_data
    );

    let mut object_group = ObjectGroup {
        active: true,
        objects: vec![]
    };        

    let frustum = gpu_api_relay::frustum::Frustum::from_view_projection(camera.view_proj);
    let mut frame_data = world::octree::RenderFrameData {
        visible_chunks: Vec::new(),
    };
    test_world.cull(&frustum, camera.position, &mut frame_data);

    let mut surface_data = SurfaceData::default();
    //generate_surface_meshlets(Vec3::ZERO, 0, 0, &mut surface_data);

    surface_resources.init(&queue, &surface_data.vertices, &surface_data.indices, &surface_data.meshlets, &init_data.factors, &surface_data.indirect_commands);
    
    let mut surface_culling_tasks = Vec::new();    
    let mut model_culling_tasks = Vec::new();
    let mut global_instances = Vec::new();

    surface_culling_tasks.push(SurfaceCullingTask {
        start_meshlet_index: 0,
        meshlet_count: surface_data.meshlets.len() as u32,
        indirect_cmd_index: 0,
        _padding: 0,
    });

    test_world.prepare_gpu_indirect_frame(&frame_data, &mut model_culling_tasks, &mut global_instances);
   
    model_bindless_resources.init(&queue, &init_data.vertices, &init_data.indices, &init_data.factors, &indirect_commands);

    object_group.objects.push(object);

    let mut object_groups = vec![];
    object_groups.push(object_group);    
    
    let solid_quad_pipeline = pipeline::solid_quad_pipeline::Pipeline::new(&device, depth_stencil_state.clone());
    let gradient_quad_pipeline = pipeline::gradient_quad_pipeline::Pipeline::new(&device, depth_stencil_state.clone());
    let line_pipeline = pipeline::line_pipeline::Pipeline::new(&device, &camera_uniform, depth_stencil_state);

    let transformation = solid_quad_pipeline::Transformation::orthographic(layout.size.width, layout.size.height);
    let mut quad_uniforms = solid_quad_pipeline::Uniforms::new(transformation, scale_factor as f32, [0.0, 0.0]);

    let clip = [0.0, 0.0, 950.0, 950.0];    

    let shadow_color = [0.0, 0.0, 0.0, 0.0];
    let shadow_offset = [0.0, 0.0];
    let shadow_blur_radius = 0.0;

    let mut image_objects = vec![];

    let image_bytes = include_bytes!("../../textures/happy-tree.png");
    let img = image::load_from_memory(image_bytes).expect("Failed to load texture");

    let hi_image = ImageObject::new(&device, &queue, &image_pipeline, "hi", true, &img, vec![
        ImageQuad {
            border_color: [0.0, 0.5, 0.0, 1.0],
            border_radius: [10.0, 10.0, 10.0, 10.0],
            border_width: 0.0,
            position: [0.0, 0.0],
            size: [100.0, 100.0],
            clip,            
            shadow_color,
            shadow_offset,
            shadow_blur_radius,
            snap: 0
        },
        ImageQuad {
            border_color: [0.0, 0.5, 0.0, 1.0],
            border_radius: [10.0, 10.0, 10.0, 10.0],
            border_width: 0.0,
            position: [220.0, 220.0],
            size: [200.0, 200.0],
            clip,            
            shadow_color,
            shadow_offset,
            shadow_blur_radius,
            snap: 0
        }
    ]);

    image_objects.push(hi_image);

    let quads = vec![        
        solid_quad_pipeline::SolidQuad {
            border_color: [0.0, 0.5, 0.0, 1.0],
            border_radius: [10.0, 10.0, 10.0, 10.0],
            color: [1.0, 0.0, 0.0, 1.0],
            border_width: 0.0,
            position: [100.0, 100.0],
            size: [100.0, 100.0],
            clip,            
            shadow_color,
            shadow_offset,
            shadow_blur_radius,
            snap: 0
        },
        solid_quad_pipeline::SolidQuad {
            border_color: [0.0, 0.5, 0.0, 1.0],
            border_radius: [15.0, 15.0, 15.0, 15.0],
            color: [1.0, 0.0, 0.0, 1.0],
            border_width: 0.0,
            position: [300.0, 100.0],
            size: [30.0, 30.0],
            clip,            
            shadow_color,
            shadow_offset,
            shadow_blur_radius,
            snap: 0
        },
        solid_quad_pipeline::SolidQuad {
            border_color: [0.0, 0.5, 0.0, 1.0],
            border_radius: [10.0, 10.0, 10.0, 10.0],
            color: [1.0, 1.0, 1.0, 1.0],
            border_width: 1.0,
            position: [500.0, 500.0],
            size: [100.0, 100.0],
            clip,            
            shadow_color,
            shadow_offset,
            shadow_blur_radius,
            snap: 0
        }
    ];

    use pipeline::gradient_quad_pipeline::{GradientQuad, color::{core::Color, Point, LinearStartEnd}};

    let start = Point::new(0.0, 200.0);
    let end = Point::new(70.0, 270.0);

    let g = LinearStartEnd::new(start, end)
        .add_stop(0.0, Color::new(1.0, 0.0, 0.0, 1.0))
        .add_stop(1.0, Color::new(0.0, 0.0, 1.0, 1.0));

    let packed = g.pack();

    let gradient_quads = vec![        
        GradientQuad {            
            colors_1: packed.colors_1,
            colors_2: packed.colors_2,
            colors_3: packed.colors_3,
            colors_4: packed.colors_4,
            offsets: packed.offsets,
            direction: packed.direction,                    
            position: [0.0, 200.0],
            size: [100.0, 100.0],            
            border_color: [0.0, 0.5, 0.0, 1.0],
            border_radius: [10.0, 10.0, 10.0, 10.0],
            border_width: 0.0,
            shadow_color,
            shadow_offset,
            shadow_blur_radius,
            snap: 0,
            clip
        }
    ];

    let line_transform = Transformation::identity();
    let line_uniforms = solid_quad_pipeline::Uniforms::new(line_transform, scale_factor as f32, [0.0, 0.0]);

    let line_data = vec![
        LineVertex { color: [1.0, 0.0, 0.0, 1.0], pos: [0.0, 0.0, 0.0] },
        LineVertex { color: [0.0, 1.0, 0.0, 1.0], pos: [0.5,  0.5, 0.5] },
        LineVertex { color: [0.0, 1.0, 0.0, 1.0], pos: [0.5,  0.0, 0.5] },
    ];

    let line_indices: Vec<u32> = vec![0, 1, 1, 2];

    let mut staging_belt = wgpu::util::StagingBelt::new(device.clone(), 5 * 1024);
    let mut frame_counter = FrameCounter::new(TARGET_FPS);

    event_loop.run(move |event, target| {        
        target.set_control_flow(ControlFlow::Wait);

        match event {
            Event::WindowEvent { event: window_event, window_id } => {
                match window_event {
                    WindowEvent::CloseRequested => {                        
                        info!("Event loop close requested");
                        target.exit();
                    }
                    WindowEvent::Resized(new_size) => {                        
                        info!("Resized");

                        let scale_factor = window.scale_factor();
                        info!("{:?}, {}", new_size, scale_factor);

                        layout.size = new_size;          

                        quad_uniforms = solid_quad_pipeline::Uniforms::new(transformation, scale_factor as f32, [0.0, 0.0]);

                        surface.configure(&device, &config);
                    }
                    WindowEvent::CursorMoved { device_id: _, position, .. } => {                        
                        let norm_x = position.x as f32 / layout.size.width as f32 - 0.5;
                        let norm_y = position.y as f32 / layout.size.height as f32 - 0.5;
                        camera.angle_y = norm_x * 5.0;
                        camera.angle_xz = norm_y;
                    }
                    WindowEvent::MouseInput { device_id: _, state, button, .. } => {
                        match state {
                            ElementState::Pressed => {                                
                            }
                            ElementState::Released => {                                                                                        
                            }                            
                        }
                    }
                    WindowEvent::MouseWheel { device_id: _, delta, phase: _ } => {                        
                        match delta {
                            MouseScrollDelta::LineDelta(_, vertical_delta) => {
                                if vertical_delta > 0.0 {
                                    camera.dist = camera.dist - 10.0;
                                } else {
                                    camera.dist = camera.dist + 10.0;
                                }
                            }
                            MouseScrollDelta::PixelDelta(position) => {
                                if position.y > 0.0 {
                                    camera.dist = camera.dist - 10.0;
                                } else {
                                    camera.dist = camera.dist + 10.0;
                                }
                            }
                        }        
                    }
                    WindowEvent::ModifiersChanged(state) => {
                        info!("Modifiers changed");                        
                    }
                    WindowEvent::KeyboardInput { device_id: _, event, is_synthetic } => {
                        warn!("{:?}", event);                    
                    }
                    WindowEvent::RedrawRequested => {
                        //frame_counter.simple_update();
                        if frame_counter.tick() {
                                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("Redraw")
                                }
                            );
            
                            // Get the next frame
                            let frame = match surface.get_current_texture() {
                                CurrentSurfaceTexture::Success(st) => st,
                                _ => panic!("Failed to get current frame texture from surface")
                            };
                            let mut texture_view_descriptor = wgpu::TextureViewDescriptor::default();
                            texture_view_descriptor.format = Some(wgpu::TextureFormat::Rgba8UnormSrgb);
                            let view = &frame.texture.create_view(&texture_view_descriptor);

                            if image_objects.len() > 0 {
                                {
                                    let mut uniform_buffer = staging_belt.write_buffer(
                                        &mut encoder,
                                        &image_pipeline.uniform_buffer,
                                        0,
                                        wgpu::BufferSize::new(std::mem::size_of::<solid_quad_pipeline::Uniforms>() as u64)
                                            .expect("Failed to create quad uniform buffer size")                                        
                                    );
                
                                    uniform_buffer.copy_from_slice(bytemuck::bytes_of(&quad_uniforms));
                                }
                            }

                            for image_object in &image_objects {
                                let vertex_bytes = bytemuck::cast_slice(&image_object.quads);
                
                                let mut vertex_buffer = staging_belt.write_buffer(
                                    &mut encoder,
                                    &image_object.vertex_buffer,
                                    0,
                                    wgpu::BufferSize::new(vertex_bytes.len() as u64).expect("Failed to create image object buffer size")                                    
                                );
            
                                vertex_buffer.copy_from_slice(vertex_bytes);
                            }

                            {
                                let mut uniform_buffer = staging_belt.write_buffer(
                                    &mut encoder,
                                    &gradient_quad_pipeline.uniform_buffer,
                                    0,
                                    wgpu::BufferSize::new(std::mem::size_of::<solid_quad_pipeline::Uniforms>() as u64).expect("Failed to create gradient quad uniform buffer size")                                    
                                );
            
                                uniform_buffer.copy_from_slice(bytemuck::bytes_of(&quad_uniforms));
                            }
                            
                            {
                                let vertex_bytes = bytemuck::cast_slice(&gradient_quads);
            
                                let mut vertex_buffer = staging_belt.write_buffer(
                                    &mut encoder,
                                    &gradient_quad_pipeline.vertex_buffer,
                                    0,
                                    wgpu::BufferSize::new(vertex_bytes.len() as u64).expect("Failed to create gradient quad buffer size")                                    
                                );
            
                                vertex_buffer.copy_from_slice(vertex_bytes);
                            }
            
                            {
                                let mut uniform_buffer = staging_belt.write_buffer(
                                    &mut encoder,
                                    &solid_quad_pipeline.uniform_buffer,
                                    0,
                                    wgpu::BufferSize::new(std::mem::size_of::<solid_quad_pipeline::Uniforms>() as u64).expect("Failed to create quad uniform buffer size")                                    
                                );
            
                                uniform_buffer.copy_from_slice(bytemuck::bytes_of(&quad_uniforms));
                            }
                            
                            {
                                let vertex_bytes = bytemuck::cast_slice(&quads);
            
                                let mut vertex_buffer = staging_belt.write_buffer(
                                    &mut encoder,
                                    &solid_quad_pipeline.vertex_buffer,
                                    0,
                                    wgpu::BufferSize::new(vertex_bytes.len() as u64).expect("Failed to create quad buffer size")                                    
                                );
            
                                vertex_buffer.copy_from_slice(vertex_bytes);
                            }           

                            {
                                let vertex_bytes = bytemuck::cast_slice(&line_data);
                                let mut vertex_buffer = staging_belt.write_buffer(
                                    &mut encoder,
                                    &line_pipeline.vertex_buffer,
                                    0,
                                    wgpu::BufferSize::new(vertex_bytes.len() as u64).expect("Failed to create line vertex buffer size")
                                );
                                vertex_buffer.copy_from_slice(vertex_bytes);

                                let index_bytes = bytemuck::cast_slice(&line_indices);
                                let mut index_buffer = staging_belt.write_buffer(
                                    &mut encoder,
                                    &line_pipeline.index_buffer,
                                    0,
                                    wgpu::BufferSize::new(index_bytes.len() as u64).expect("Failed to create line index buffer size")
                                );
                                index_buffer.copy_from_slice(index_bytes);
                            }

                            camera.update(layout.size.width as f32, layout.size.height as f32);
            
                            {                                                                                            
                                let camera_uniform = camera.get_uniform();

                                let mut model_camera_slice = staging_belt.write_buffer(
                                    &mut encoder,
                                    &model_pipeline.camera_buffer,
                                    0,
                                    wgpu::BufferSize::new(CAMERA_UNIFORM_SIZE).expect("Failed to allocate model camera slice")                                    
                                );            
                                model_camera_slice.copy_from_slice(bytemuck::bytes_of(&camera_uniform));

                                let mut line_camera_slice = staging_belt.write_buffer(
                                    &mut encoder,
                                    &line_pipeline.camera_buffer,
                                    0,
                                    wgpu::BufferSize::new(CAMERA_UNIFORM_SIZE).expect("Failed to allocate line camera slice")                                    
                                );            
                                line_camera_slice.copy_from_slice(bytemuck::bytes_of(&camera_uniform));
                            }

                            {
                                for object_group in &mut object_groups {
                                    for object in &mut object_group.objects {
                                        if object_group.active == false {
                                            continue;
                                        }
                                        
                                        if object.instances_count == 0 {
                                            continue;
                                        }                                        

                                        let animation_index = 0;

                                        match object.animation_computation_mode {
                                            AnimationComputationMode::NotAnimated => {
                                                for mesh in &object.meshes {
                                                    match &mesh.node_transform {
                                                        Some(node_transform) => {
                                                            let mut node_transform_slice = staging_belt.write_buffer(
                                                                &mut encoder,
                                                                &mesh.node_transform_buffer,
                                                                0,
                                                                wgpu::BufferSize::new(gpu_api::pipeline::model_pipeline::NODE_TRANSFORM_UNIFORM_SIZE).expect("Failed to allocate node transform slice")                                                                
                                                            );
                                        
                                                            node_transform_slice.copy_from_slice(bytemuck::bytes_of(node_transform));
                                                        }
                                                        None => {}
                                                    }
                                                }
                                            }
                                            AnimationComputationMode::ComputeInRealTime => {
                                                for channel in &mut object.animations[animation_index].channels {
                                                    let current_time = channel.start_instant.elapsed().as_secs_f32();

                                                    let mut frame_index = channel.frame_index;
                    
                                                    for timestamp in channel.timestamps.iter().skip(frame_index) {
                                                        if timestamp > &current_time {
                                                            break;
                                                        }
                                                        
                                                        frame_index = frame_index + 1;
                                                    }

                                                    if frame_index == channel.timestamps.len() {
                                                        frame_index = 0;
                                                        channel.frame_index = 0;
                                                        #[cfg(not(target_arch = "wasm32"))] {
                                                            channel.start_instant = std::time::Instant::now();
                                                        }                                                    
                                                        #[cfg(target_arch = "wasm32")] {
                                                            channel.start_instant = web_time::Instant::now();
                                                        }
                                                    }

                                                    let previous_frame_index = match frame_index {
                                                        0 => 0,
                                                        _ => frame_index - 1
                                                    };

                                                    let factor = (current_time - channel.timestamps[previous_frame_index]) / (channel.timestamps[frame_index] - channel.timestamps[previous_frame_index]);

                                                    match &channel.property {
                                                        AnimationProperty::Translation => {
                                                            let translation = channel.translations[previous_frame_index].lerp(channel.translations[frame_index], factor);
                                                            object.nodes[channel.target_index].translation = translation;
                                                        }
                                                        AnimationProperty::Rotation => {
                                                            let rotation = channel.rotations[previous_frame_index].lerp(channel.rotations[frame_index], factor).normalize();
                                                            object.nodes[channel.target_index].rotation = rotation;
                                                        }
                                                        AnimationProperty::Scale => {
                                                            let scale = channel.scales[previous_frame_index].lerp(channel.scales[frame_index], factor);
                                                            object.nodes[channel.target_index].scale = scale;
                                                        }
                                                        AnimationProperty::MorphTargetWeights => {
                                                            let weight_morph = channel.weight_morphs[frame_index];
                                                        }
                                                    }
                                                }
                                                
                                                for node_index in object.node_topological_sorting.iter() {
                                                    match object.node_map.get(node_index) {
                                                        Some(parent_index) => {
                                                            let parent_transform = object.nodes[*parent_index].global_transform_matrix;
                                                            let node = &mut object.nodes[*node_index];
                                            
                                                            let local_transform = glam::Mat4::from_scale_rotation_translation(node.scale, node.rotation, node.translation);

                                                            node.global_transform_matrix = parent_transform * local_transform;
                                                        }
                                                        None => {}
                                                    }                                            
                                                }                                    

                                                let mut joint_matrices: [Mat4; gpu_api::pipeline::model_pipeline::JOINT_MATRICES_COUNT] = [Mat4::IDENTITY; gpu_api::pipeline::model_pipeline::JOINT_MATRICES_COUNT];
                                                
                                                let mut joint_matrix_index = 0;
                                                let skin_index = 0;
                                                
                                                for joint in &object.skins[skin_index].joints {
                                                    let joint_matrix = object.nodes[joint.node_index].global_transform_matrix * joint.inverse_bind_matrix;
                                                    joint_matrices[joint_matrix_index] = joint_matrix;
                                                    joint_matrix_index = joint_matrix_index + 1;
                                                }
                                                
                                                for node in &object.nodes {
                                                    match node.mesh_index {
                                                        Some(mesh_index) => {                                                            
                                                            match &mut object.meshes[mesh_index].node_transform {
                                                                Some(node_transform) => {                                                                    
                                                                    node_transform.transform = node.global_transform_matrix;
                                                                }
                                                                None => {}
                                                            }
                                                        }
                                                        None => {}
                                                    }
                                                }

                                                {                                                                
                                                    let mut joint_matrices_slice = staging_belt.write_buffer(
                                                        &mut encoder,
                                                        &object.joint_matrices_buffer,
                                                        0,
                                                        wgpu::BufferSize::new(gpu_api::pipeline::model_pipeline::JOINT_MATRICES_UNIFORM_SIZE).expect("Failed to allocate joint matrices slice")                                                        
                                                    );
                                
                                                    joint_matrices_slice.copy_from_slice(bytemuck::cast_slice(&joint_matrices));
                                                }

                                                for mesh in &object.meshes {
                                                    match &mesh.node_transform {
                                                        Some(node_transform) => {
                                                            let mut node_transform_slice = staging_belt.write_buffer(
                                                                &mut encoder,
                                                                &mesh.node_transform_buffer,
                                                                0,
                                                                wgpu::BufferSize::new(gpu_api::pipeline::model_pipeline::NODE_TRANSFORM_UNIFORM_SIZE).expect("Failed to allocate node transform slice")                                                                
                                                            );
                                        
                                                            node_transform_slice.copy_from_slice(bytemuck::bytes_of(node_transform));
                                                        }
                                                        None => {}
                                                    }
                                                }
                                            }
                                            AnimationComputationMode::PreComputed => {                                                
                                                if object.animations[animation_index].frame_index == object.animations[animation_index].frame_cycle_count {
                                                    object.animations[animation_index].frame_index = 7;
                                                }                                                                                                
                                                
                                                {
                                                    let mut joint_matrices_slice = staging_belt.write_buffer(
                                                        &mut encoder,
                                                        &object.joint_matrices_buffer,
                                                        0,
                                                        wgpu::BufferSize::new(gpu_api::pipeline::model_pipeline::JOINT_MATRICES_UNIFORM_SIZE).expect("Failed to allocate joint matrices slice")
                                                    );
                                
                                                    joint_matrices_slice.copy_from_slice(bytemuck::cast_slice(&object.animations[animation_index].joint_matrices[object.animations[animation_index].frame_index]));
                                                }

                                                //let q = &object.animations[animation_index].joint_matrices[object.animations[animation_index].frame_index];
                                                
                                                //init_data.joints.clear();
                                                //init_data.joints.extend_from_slice(q);
                                                init_data.nodes.clear();

                                                for mesh in &object.meshes {
                                                    if mesh.node_transform.is_some() {
                                                        let mut node_transform_slice = staging_belt.write_buffer(
                                                            &mut encoder,
                                                            &mesh.node_transform_buffer,
                                                            0,
                                                            wgpu::BufferSize::new(gpu_api::pipeline::model_pipeline::NODE_TRANSFORM_UNIFORM_SIZE).expect("Failed to allocate node transform slice")
                                                        );

                                                        let mesh_node_transform = &object.animations[animation_index].mesh_node_transforms[mesh.index];
                                                        
                                                        let node_transform = &mesh_node_transform.node_transforms[object.animations[animation_index].frame_index];
                                    
                                                        info!("{:?}", node_transform);
                                                        node_transform_slice.copy_from_slice(bytemuck::bytes_of(node_transform));
                                                        init_data.nodes.push(*node_transform);
                                                    }                                                    
                                                }

                                                object.animations[animation_index].frame_index = object.animations[animation_index].frame_index + 1;                                                
                                            }
                                        }
                                        
                                        let mut view_slice = staging_belt.write_buffer(
                                            &mut encoder,
                                            &object.instance_buffer,
                                            0,
                                            wgpu::BufferSize::new(object.model_instance_size).expect("Failed to allocate view slice")                                            
                                        );
                    
                                        view_slice.copy_from_slice(bytemuck::cast_slice(&object.model_instances));
                                    }
                                }
                            }
                                                        
                            surface_resources.load_frame(&queue, &mut encoder, &camera, &mut staging_belt, &surface_culling_tasks);
                            surface_resources.clear_gpu_driven_frame(&mut encoder);
                            
                            {
                                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("Surface Culling Pass"),
                                    timestamp_writes: None,
                                });

                                surface_resources.compute_gpu_driven_frame(&mut compute_pass, surface_data.meshlets.len() as u32);
                            }
                            
                            
                            if init_data.nodes.is_empty() {
                                init_data.nodes.push(NodeData {
                                    info: [0, 0, 0, 0],
                                    transform: Mat4::IDENTITY,
                                });
                            }

                            model_bindless_resources.load_frame(&queue, &mut encoder, &camera, &mut staging_belt, &global_instances, &init_data.nodes,
                                //&init_data.joints,
                                &model_culling_tasks);
                            model_bindless_resources.clear_gpu_driven_frame(&mut encoder);
                            
                            {
                                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("Model Bindless Culling Pass"),
                                    timestamp_writes: None,
                                });

                                model_bindless_resources.compute_gpu_driven_frame(&mut compute_pass, &model_culling_tasks);
                            }
                            

                            {
                                let mut render_pass = encoder.begin_render_pass(
                                    &wgpu::RenderPassDescriptor {
                                        label: Some("Render pass"),
                                        color_attachments: &[
                                            Some(wgpu::RenderPassColorAttachment {
                                                view,
                                                depth_slice: None,
                                                resolve_target: None,
                                                ops: wgpu::Operations {
                                                    load: wgpu::LoadOp::Clear(
                                                        wgpu::Color {
                                                            r: 1.0,
                                                            g: 1.0,
                                                            b: 1.0,
                                                            a: 1.0
                                                        },
                                                    ),
                                                    store: StoreOp::Store
                                                }
                                            })
                                        ],
                                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                            view: &model_pipeline.depth_texture.view,
                                            depth_ops: Some(wgpu::Operations {
                                                load: wgpu::LoadOp::Clear(1.0),
                                                store: wgpu::StoreOp::Store,
                                            }),
                                            stencil_ops: None,
                                        }),
                                        timestamp_writes: None,
                                        occlusion_query_set: None,
                                        multiview_mask: None
                                    }
                                );
            
                                //surface_resources.draw_gpu_driven_frame(&mut render_pass, surface_data.meshlets.len() as u32);
                                model_bindless_resources.draw_gpu_driven_frame(&mut render_pass, &indirect_commands);
                                //model_pipeline.draw(&mut render_pass, &object_groups);
                                line_pipeline.draw(&mut render_pass, line_indices.len() as u32);
                                image_pipeline.draw(&mut render_pass, &image_objects);
                                gradient_quad_pipeline.draw(&mut render_pass, gradient_quads.len() as u32);
                                solid_quad_pipeline.draw(&mut render_pass, quads.len() as u32);
                            }
            
                            staging_belt.finish();
                            queue.submit(Some(encoder.finish()));
                            queue.present(frame);
                            staging_belt.recall();
                        }

                        window.request_redraw();
                    }
                    _ => {}
                }
            }
            Event::UserEvent(_) => {                
            }                        
            _ => {}
        }
    }).expect("Event loop failed");
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));    

    // 2. Настройка логгера через fern
    let mut dispatch = fern::Dispatch::new()
        .format(|out, message, record| {
            let colors = ColoredLevelConfig::new()
                .error(Color::Red)
                .warn(Color::Yellow)
                .info(Color::Green)
                .debug(Color::White);
        
            out.finish(format_args!(
                "[{}] [{}] {}",
                colors.color(record.level()),
                record.target(),
                message
            ))
        })
        .level(log::LevelFilter::Info)
        .level_for("wgpu", log::LevelFilter::Debug);

    // Куда выводим результат?
    #[cfg(target_arch = "wasm32")]
    {
        dispatch = dispatch.chain(fern::Output::call(console_log::log));
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        dispatch = dispatch.chain(std::io::stdout());
    }

    dispatch.apply().expect("Could not initialize logger");
    
    #[cfg(not(target_arch = "wasm32"))]
    {            
        let rt = Runtime::new().expect("Failed to create runtime");        
        rt.block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Warn).expect("Could not initialize logger");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}
