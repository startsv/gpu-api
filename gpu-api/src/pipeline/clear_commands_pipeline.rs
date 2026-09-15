use wgpu::ComputePass;

pub struct ClearCommandsPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    total_commands_count: u32,
}

impl ClearCommandsPipeline {
    pub fn new(
        device: &wgpu::Device,        
        indirect_commands_buffer: &wgpu::Buffer,
        total_commands_count: u32,
    ) -> Self {        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Clear Commands Shader Module"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/clear_commands.wgsl").into()),
        });
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Clear Commands Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false }, // read_write в WGSL
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Clear Commands Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // 4. Компилируем сам Compute Pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Clear Commands Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // 5. Создаем Bind Group, привязывая наш реальный GPU-буфер команд к слоту @binding(0)
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Clear Commands Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: indirect_commands_buffer.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            bind_group,
            total_commands_count,
        }
    }
    
    pub fn compute(&self, compute_pass: &mut ComputePass) {
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        
        // @workgroup_size(64)
        let workgroup_count = (self.total_commands_count + 63) / 64;

        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);        
    }
}
