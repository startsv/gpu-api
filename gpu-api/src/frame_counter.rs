#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

pub struct FrameCounter {
    #[cfg(not(target_arch = "wasm32"))]
    pub last_printed_instant: Instant,
    #[cfg(target_arch = "wasm32")]
    pub last_printed_instant: webtime::Instant,
    #[cfg(not(target_arch = "wasm32"))]
    pub last_frame_instant: Instant,
    #[cfg(target_arch = "wasm32")]
    pub last_frame_instant: web_time::Instant,
    pub frame_count: u32,
    pub fps: f64,
    pub delta_time: f64,
    pub target_fps: u32,
}

impl FrameCounter {
    pub fn new(target_fps: u32) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let now = Instant::now();
        #[cfg(target_arch = "wasm32")]
        let now = web_time::Instant::now();

        Self {            
            last_printed_instant: now,            
            last_frame_instant: now,
            frame_count: 0,
            fps: 0.0,
            delta_time: 0.0,
            target_fps,
        }
    }
    
    pub fn update(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        let now = Instant::now();
        #[cfg(target_arch = "wasm32")]
        let now = web_time::Instant::now();                
        self.update_metrics(now);
    }
    
    pub fn tick(&mut self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        let now = Instant::now();
        #[cfg(target_arch = "wasm32")]
        let now = web_time::Instant::now();

        let target_frame_time = Duration::from_secs_f64(1.0 / self.target_fps as f64);
        let elapsed_time = now.duration_since(self.last_frame_instant);
        
        if elapsed_time < target_frame_time {
            return false;
        }
        
        self.update_metrics(now);
        true
    }
    
    fn update_metrics(&mut self, now: Instant) {
        let timestep = now.duration_since(self.last_frame_instant);
        self.delta_time = timestep.as_secs_f64();
        self.last_frame_instant = now;

        self.frame_count += 1;

        let time_since_print = now.duration_since(self.last_printed_instant);
        if time_since_print >= Duration::from_secs(1) {
            self.fps = self.frame_count as f64 / time_since_print.as_secs_f64();
            self.frame_count = 0;
            self.last_printed_instant = now;
        }
    }
}
