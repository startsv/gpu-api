struct DrawIndexedIndirectCmd {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

@group(0) @binding(0) 
var<storage, read_write> indirect_commands: array<DrawIndexedIndirectCmd>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cmd_id = global_id.x;
    let total_commands = arrayLength(&indirect_commands);
    
    if (cmd_id >= total_commands) {
        return;
    }        
    atomicStore(&indirect_commands[cmd_id].instance_count, 0u);
}
