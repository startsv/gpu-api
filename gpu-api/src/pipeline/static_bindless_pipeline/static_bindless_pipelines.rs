use std::borrow::Cow;
use wgpu::TextureFormat;
use crate::pipeline::static_bindless_pipeline::static_bindless_layout::PipelineLayouts;

pub fn create_pipelines(
    device: &wgpu::Device,
    layouts: &PipelineLayouts,    
    depth_stencil: Option<wgpu::DepthStencilState>
) -> (wgpu::ComputePipeline, wgpu::RenderPipeline) {        
    let culling_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Culling Compute Shader Module"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/static_bindless_culling.wgsl"))),
    });    

    let render_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Meshlet Render Shader Module"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/static_bindless.wgsl"))),
    });

    // 2. Сборка Compute Пайплайна
    let culling_compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Culling Compute Pipeline"),
        layout: Some(&layouts.compute_pipeline_layout),
        module: &culling_module,
        entry_point: Some("culling_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // 3. Сборка Графического Render Пайплайна
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Meshlet Render Pipeline"),
        layout: Some(&layouts.render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &render_module,
            entry_point: Some("vs_main"),            
            compilation_options: Default::default(),
            buffers: &[], 
        },
        fragment: Some(wgpu::FragmentState {
            module: &render_module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[
                Some(wgpu::ColorTargetState {
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
                })
            ],
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },        
        depth_stencil,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    (culling_compute_pipeline, render_pipeline)
}
