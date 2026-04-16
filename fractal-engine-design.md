# Fractal Rendering Engine — Scaffold Design Document

## Overview

Build a macOS Metal compute-based fractal raymarching engine with:
- Shader hot-reload from `.metal` files on disk
- Auto-generated ImGui parameter GUI from shader comment metadata
- SDL3 windowing with native Metal layer
- Foundation for hierarchical temporal raymarching pipeline (not implemented in scaffold, but architecture must support it)

The scaffold's success criteria: a window showing a full-screen compute shader output (a simple raymarched SDF), an ImGui overlay with sliders parsed from shader comments, and live shader recompilation on file save with no host app restart.

**Language:** C++ (C++20) with one Objective-C++ file for Metal API calls.  
**Build system:** CMake  
**Target:** macOS 14+, Apple Silicon (M1+), Metal 3  
**Resolution:** Start at 1920×1080 window, architecture must support 3840×2160  

---

## Project Structure

```
fractal-engine/
├── CMakeLists.txt
├── src/
│   ├── main.cpp                 # Entry point, SDL init, main loop
│   ├── metal_backend.h          # Pure C++ interface to Metal (no ObjC types)
│   ├── metal_backend.mm         # All Metal API calls (Objective-C++)
│   ├── shader_manager.h         # Shader hot-reload + param parsing
│   ├── shader_manager.cpp       # File watching, compilation requests
│   ├── gui.h                    # ImGui setup and parameter rendering
│   ├── gui.cpp                  # ImGui frame logic
│   └── types.h                  # Shared types (ShaderParam, FrameUniforms, etc.)
├── shaders/
│   ├── common.metal             # Shared functions (SDF primitives, noise, etc.)
│   ├── raymarch.metal           # Fractal DE + trace kernel
│   ├── reconstruct.metal        # TAA resolve kernel (placeholder in scaffold)
│   └── present.metal            # Final output + post-processing
└── third_party/
    ├── imgui/                   # Dear ImGui (docking branch)
    │   ├── imgui_impl_sdl3.h/cpp
    │   └── imgui_impl_metal.h/mm
    └── SDL3/                    # SDL3 (via CMake FetchContent or submodule)
```

---

## Types (types.h)

```cpp
#pragma once
#include <string>
#include <vector>
#include <array>

// Parsed from shader comment headers
struct ShaderParam {
    enum Type { Float, Int, Float2, Float3, Float4 };
    
    std::string name;          // display name and uniform field name
    Type type;
    float min_val[4];          // per-component min
    float max_val[4];          // per-component max
    float default_val[4];      // per-component default
    float current_val[4];      // current value (mutated by GUI)
    int component_count;       // 1, 2, 3, or 4
};

// Packed uniform buffer uploaded to GPU each frame.
// This struct is mirrored exactly in Metal as `constant FrameUniforms& frame`.
// Alignment: std430-like, match Metal's rules (float4 = 16-byte aligned).
struct alignas(16) FrameUniforms {
    float time;                // seconds since start
    float delta_time;          // seconds since last frame
    uint32_t frame_index;      // monotonic frame counter
    uint32_t _pad0;

    float resolution[2];       // output resolution (full-res pixels)
    float inv_resolution[2];   // 1.0 / resolution

    float mouse[2];            // normalized mouse position [0,1]
    float mouse_click[2];      // position of last click

    // Camera
    float camera_pos[3];
    float _pad1;
    float camera_fwd[3];
    float _pad2;
    float camera_up[3];
    float _pad3;
    float camera_right[3];
    float camera_fov;          // vertical FOV in radians

    // View-projection matrices (column-major, 4×4)
    float view_proj[16];
    float prev_view_proj[16];
    float inv_view_proj[16];

    // Jitter for TAA (in full-res pixel units, range [-0.5, 0.5])
    float jitter[2];
    float _pad4[2];

    // Shader params: packed sequentially as float4s.
    // Max 32 user params → 32 float4 slots = 512 bytes.
    // Param packing: each param occupies one float4 regardless of actual size.
    // Float → (val, 0, 0, 0), Float3 → (x, y, z, 0), etc.
    float params[32][4];
    uint32_t param_count;
    uint32_t _pad5[3];
};

// Identifies a compiled compute pipeline
struct ComputeKernel {
    std::string name;          // kernel function name
    std::string source_path;   // .metal file path
    uint64_t pipeline_id;      // opaque handle (index into backend's pipeline array)
    bool valid;                // false if compilation failed
    std::string error_msg;     // non-empty if compilation failed
};

// Texture descriptor for backend allocation
struct TextureDesc {
    uint32_t width;
    uint32_t height;
    enum Format { RGBA16Float, RGBA32Float, R32Float, R32Uint } format;
    bool read_write;           // if true, created as read_write in shaders
    std::string name;          // debug label
};
```

---

## Metal Backend (metal_backend.h / metal_backend.mm)

### Public Interface (metal_backend.h)

Pure C++ header. No Objective-C types leak through. Uses pimpl to hide Metal objects.

```cpp
#pragma once
#include "types.h"
#include <memory>
#include <string>
#include <functional>

struct SDL_Window;

class MetalBackend {
public:
    MetalBackend();
    ~MetalBackend();

    // Initialization
    bool init(SDL_Window* window);
    void shutdown();

    // Texture management
    // Returns an opaque texture ID (index). -1 on failure.
    int create_texture(const TextureDesc& desc);
    void destroy_texture(int texture_id);
    // Resize existing texture (preserves ID, reallocates storage)
    void resize_texture(int texture_id, uint32_t width, uint32_t height);

    // Buffer management
    // Returns opaque buffer ID. -1 on failure.
    int create_buffer(size_t size_bytes, const std::string& label);
    void destroy_buffer(int buffer_id);
    void write_buffer(int buffer_id, const void* data, size_t size, size_t offset = 0);
    // Clear buffer to zero
    void clear_buffer(int buffer_id);

    // Shader compilation
    // Compiles MSL source string, extracts kernel function by name.
    // Returns pipeline ID, or -1 on failure. Error string set on failure.
    int compile_kernel(const std::string& msl_source,
                       const std::string& kernel_function_name,
                       std::string& error_out);

    // Dispatches
    struct DispatchParams {
        int pipeline_id;
        uint32_t grid_width;       // total threads X
        uint32_t grid_height;      // total threads Y
        uint32_t threadgroup_w;    // threads per group X (typ. 16)
        uint32_t threadgroup_h;    // threads per group Y (typ. 16)
        // Bindings: texture IDs and buffer IDs in order.
        // Bound as [[texture(0)]], [[texture(1)]], ... and [[buffer(0)]], [[buffer(1)]], ...
        std::vector<int> textures;
        std::vector<int> buffers;
    };

    // Begin a frame (acquire drawable, create command buffer)
    void begin_frame();
    // Dispatch a compute kernel
    void dispatch(const DispatchParams& params);
    // Blit a texture to the screen drawable (for final present)
    void blit_to_screen(int source_texture_id);
    // ImGui rendering pass (called between begin/end frame)
    void render_imgui();
    // End frame (commit command buffer, present drawable)
    void end_frame();

    // Getters
    uint32_t drawable_width() const;
    uint32_t drawable_height() const;
    // Returns the raw MTL device pointer for ImGui Metal backend init.
    // Type-erased as void* to avoid ObjC in header.
    void* raw_device() const;
    void* raw_command_queue() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
```

### Implementation Notes (metal_backend.mm)

The `Impl` struct holds all Objective-C Metal objects:

```objc
struct MetalBackend::Impl {
    id<MTLDevice>             device;
    id<MTLCommandQueue>       command_queue;
    CAMetalLayer*             metal_layer;
    id<CAMetalDrawable>       current_drawable;   // per-frame
    id<MTLCommandBuffer>      current_cmd_buffer;  // per-frame

    // Resource pools
    std::vector<id<MTLTexture>>          textures;
    std::vector<id<MTLBuffer>>           buffers;
    std::vector<id<MTLComputePipelineState>> pipelines;

    // Reusable render pass for ImGui (renders into drawable texture)
    MTLRenderPassDescriptor*  imgui_render_pass;
};
```

Key implementation details:

1. **SDL3 Metal integration:**
   ```objc
   SDL_PropertiesID props = SDL_GetWindowProperties(window);
   // SDL3 creates a CAMetalLayer accessible via properties
   // OR use SDL_Metal_CreateView and extract the layer
   metal_layer = (__bridge CAMetalLayer*)SDL_GetPointerProperty(props, 
       SDL_PROP_WINDOW_COCOA_METAL_VIEW_LAYER_POINTER, NULL);
   metal_layer.device = device;
   metal_layer.pixelFormat = MTLPixelFormatRGBA16Float;
   metal_layer.framebufferOnly = NO;  // needed if blitting from compute
   ```

2. **Shader compilation:**
   ```objc
   int compile_kernel(const std::string& source, const std::string& name, std::string& err) {
       NSError* error = nil;
       MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
       opts.languageVersion = MTLLanguageVersion3_1;
       opts.fastMathEnabled = YES;  // important for DE performance
       
       id<MTLLibrary> lib = [device newLibraryWithSource:@(source.c_str())
                                                 options:opts
                                                   error:&error];
       if (!lib) {
           err = [[error localizedDescription] UTF8String];
           return -1;
       }
       id<MTLFunction> func = [lib newFunctionWithName:@(name.c_str())];
       if (!func) {
           err = "Function '" + name + "' not found in source";
           return -1;
       }
       id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func
                                                                               error:&error];
       if (!pso) {
           err = [[error localizedDescription] UTF8String];
           return -1;
       }
       int id = (int)pipelines.size();
       pipelines.push_back(pso);
       return id;
   }
   ```

3. **Dispatch:**
   ```objc
   void dispatch(const DispatchParams& p) {
       id<MTLComputeCommandEncoder> enc = [current_cmd_buffer computeCommandEncoder];
       [enc setComputePipelineState:pipelines[p.pipeline_id]];
       for (int i = 0; i < p.textures.size(); i++)
           [enc setTexture:textures[p.textures[i]] atIndex:i];
       for (int i = 0; i < p.buffers.size(); i++)
           [enc setBuffer:buffers[p.buffers[i]] offset:0 atIndex:i];
       MTLSize grid = MTLSizeMake(p.grid_width, p.grid_height, 1);
       MTLSize group = MTLSizeMake(p.threadgroup_w, p.threadgroup_h, 1);
       [enc dispatchThreads:grid threadsPerThreadgroup:group];
       [enc endEncoding];
   }
   ```

4. **Pipeline replacement on hot-reload:**
   When a shader recompiles, the new pipeline gets a *new* ID. The caller (ShaderManager) holds the current pipeline ID per kernel name. On successful recompile, the caller swaps its stored ID to the new one. Old pipelines are retained in the vector (they're refcounted by Metal internally; could add a GC pass but not worth it for a dev tool). On failed recompile, the old ID stays valid.

5. **blit_to_screen:** Use an `MTLBlitCommandEncoder` to copy from the compute output texture to the current drawable texture. If sizes match, `copyFromTexture:toTexture:` is zero-cost on Apple Silicon (just remaps the page tables). If sizes differ, use a trivial fragment shader blit or `MTLBlitCommandEncoder` with a source/dest region.

---

## Shader Manager (shader_manager.h / shader_manager.cpp)

### Responsibilities
1. Watch `.metal` files for changes (poll-based, every 500ms)
2. Parse `// @param` comment headers from shader source
3. Request compilation via MetalBackend
4. Track current pipeline ID per kernel name
5. Report errors to GUI

### Shader Param Comment Format

```metal
// @param <name> <type> <min...> <max...> <default...>
//
// Types and their argument counts:
//   float   → min max default              (3 values)
//   int     → min max default              (3 values)
//   float2  → min.x min.y max.x max.y default.x default.y  (6 values)
//   float3  → min.x min.y min.z max.x max.y max.z def.x def.y def.z (9 values)
//   float4  → (12 values, same pattern)
//
// Special type modifiers:
//   color3  → same as float3 but renders as ImGui::ColorEdit3
//   color4  → same as float4 but renders as ImGui::ColorEdit4
```

Example shader header:
```metal
// @param fold_scale float 1.0 3.0 2.0
// @param iterations int 1 20 12
// @param fog_density float 0.001 0.1 0.02
// @param base_color color3 0 0 0 1 1 1 0.8 0.3 0.1
// @param camera_speed float 0.1 10.0 2.0

#include <metal_stdlib>
using namespace metal;
// ... shader code
```

### File Watcher

Simple poll-based watcher. No need for FSEvents/kqueue for a dev tool.

```cpp
class ShaderManager {
public:
    ShaderManager(MetalBackend& backend, const std::string& shader_dir);

    // Call once per frame. Checks file timestamps, recompiles if changed.
    // Returns true if any shader was recompiled this frame.
    bool poll_and_reload();

    // Get current pipeline ID for a kernel. -1 if not loaded or failed.
    int get_pipeline(const std::string& kernel_name) const;

    // Get parsed params for a shader file (by filename, e.g. "raymarch.metal")
    const std::vector<ShaderParam>& get_params(const std::string& filename) const;

    // Get last error for a shader file. Empty if no error.
    const std::string& get_error(const std::string& filename) const;

    // Register a shader file and its kernel function name.
    // Call at startup for each shader.
    void register_shader(const std::string& filename, const std::string& kernel_name);

private:
    struct ShaderEntry {
        std::string filename;
        std::string kernel_name;
        std::string full_path;
        int pipeline_id = -1;
        uint64_t last_modified = 0;    // filesystem timestamp
        std::vector<ShaderParam> params;
        std::string error;
    };

    MetalBackend& backend;
    std::string shader_dir;
    std::vector<ShaderEntry> entries;

    // Read file, parse params, compile, update entry
    void reload_shader(ShaderEntry& entry);
    // Parse // @param lines from source string
    std::vector<ShaderParam> parse_params(const std::string& source);
};
```

### Include Handling

The `common.metal` file contains shared SDF functions, noise, etc. Metal's runtime compilation supports `#include` if you provide an `MTLCompileOptions` with a library search path, but this is fragile. Simpler approach: **string concatenation**.

```cpp
void reload_shader(ShaderEntry& entry) {
    std::string common_src = read_file(shader_dir + "/common.metal");
    std::string shader_src = read_file(entry.full_path);
    std::string combined = common_src + "\n" + shader_src;
    
    entry.params = parse_params(shader_src);  // parse from shader only, not common
    
    std::string err;
    int new_pipeline = backend.compile_kernel(combined, entry.kernel_name, err);
    if (new_pipeline >= 0) {
        entry.pipeline_id = new_pipeline;
        entry.error.clear();
    } else {
        entry.error = err;
        // Keep old pipeline_id — don't break rendering on compile error
    }
}
```

**Important:** When reporting compilation errors, line numbers will be offset by the length of `common.metal`. Either subtract that offset when displaying errors, or use `#line` directives:
```cpp
std::string combined = common_src + "\n#line 1\n" + shader_src;
```

---

## GUI (gui.h / gui.cpp)

### Responsibilities
1. Initialize ImGui with SDL3 + Metal backends
2. Render shader params as appropriate widgets
3. Display shader compilation errors
4. Display frame timing / debug info
5. Provide camera control (WASD + mouse, or a simple orbital camera)

### Auto-Generated Parameter GUI

```cpp
void render_shader_params(std::vector<ShaderParam>& params) {
    for (auto& p : params) {
        switch (p.type) {
            case ShaderParam::Float:
                ImGui::SliderFloat(p.name.c_str(), &p.current_val[0],
                                   p.min_val[0], p.max_val[0]);
                break;
            case ShaderParam::Int:
                // current_val stored as float, cast for display
                {
                    int v = (int)p.current_val[0];
                    ImGui::SliderInt(p.name.c_str(), &v,
                                     (int)p.min_val[0], (int)p.max_val[0]);
                    p.current_val[0] = (float)v;
                }
                break;
            case ShaderParam::Float3:
                if (p.is_color) {
                    ImGui::ColorEdit3(p.name.c_str(), p.current_val);
                } else {
                    ImGui::SliderFloat3(p.name.c_str(), p.current_val,
                                        p.min_val[0], p.max_val[0]);
                }
                break;
            // Float2, Float4 similarly
        }
    }
}
```

### Error Display

```cpp
void render_shader_errors(const ShaderManager& sm) {
    // Red text panel at top of screen for any shader with errors
    for (const auto& entry : sm.entries()) {
        if (!entry.error.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0.3, 0.3, 1));
            ImGui::TextWrapped("[%s] %s", entry.filename.c_str(), entry.error.c_str());
            ImGui::PopStyleColor();
        }
    }
}
```

### Camera

Simple FPS camera. WASD + mouse-drag for rotation. Scroll for speed adjustment.

```cpp
struct Camera {
    float pos[3]   = {0, 0, -3};
    float yaw      = 0;         // radians
    float pitch     = 0;         // radians
    float fov       = 1.2f;      // radians, ~70 degrees
    float speed     = 2.0f;
    float sensitivity = 0.003f;

    void update(float dt, const SDL_Event* events, int event_count);
    void get_vectors(float* fwd, float* up, float* right) const;
    void get_view_matrix(float* out_4x4) const;
    void get_projection_matrix(float aspect, float near, float far, float* out_4x4) const;
    void get_view_proj(float aspect, float near, float far, float* out_4x4) const;
};
```

Camera should be disabled when ImGui wants input (`ImGui::GetIO().WantCaptureMouse / WantCaptureKeyboard`).

---

## Main Loop (main.cpp)

```cpp
int main(int argc, char* argv[]) {
    // 1. SDL Init
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("fractal-engine",
        1920, 1080,
        SDL_WINDOW_METAL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);

    // 2. Metal backend init
    MetalBackend backend;
    backend.init(window);

    // 3. ImGui init
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplSDL3_InitForMetal(window);
    ImGui_ImplMetal_Init((__bridge id<MTLDevice>)backend.raw_device());

    // 4. Shader manager init
    ShaderManager shaders(backend, "shaders/");
    shaders.register_shader("raymarch.metal", "raymarch_kernel");
    shaders.register_shader("present.metal", "present_kernel");
    // reconstruct.metal registered but starts as placeholder

    // 5. Create textures
    uint32_t w = backend.drawable_width();
    uint32_t h = backend.drawable_height();
    uint32_t half_w = w / 2, half_h = h / 2;

    int tex_current_color = backend.create_texture({half_w, half_h, TextureDesc::RGBA16Float, true, "current_color"});
    int tex_current_depth = backend.create_texture({half_w, half_h, TextureDesc::R32Float, true, "current_depth"});
    int tex_output        = backend.create_texture({w, h, TextureDesc::RGBA16Float, true, "output"});
    // History textures for TAA (placeholder, allocated now for pipeline readiness)
    int tex_history_a     = backend.create_texture({w, h, TextureDesc::RGBA16Float, true, "history_a"});
    int tex_history_b     = backend.create_texture({w, h, TextureDesc::RGBA16Float, true, "history_b"});
    bool ping = false;  // ping-pong index for history

    // 6. Create uniform buffer
    int buf_uniforms = backend.create_buffer(sizeof(FrameUniforms), "uniforms");

    // 7. Camera
    Camera camera;

    // 8. Timing
    uint64_t start_time = SDL_GetPerformanceCounter();
    uint64_t freq = SDL_GetPerformanceFrequency();
    uint32_t frame_index = 0;
    float prev_time = 0;

    // Main loop
    bool running = true;
    while (running) {
        // --- Events ---
        SDL_Event event;
        std::vector<SDL_Event> events;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT) running = false;
            events.push_back(event);
        }

        // --- Timing ---
        float time = (float)(SDL_GetPerformanceCounter() - start_time) / (float)freq;
        float dt = time - prev_time;
        prev_time = time;

        // --- Hot reload ---
        shaders.poll_and_reload();

        // --- Camera update ---
        if (!ImGui::GetIO().WantCaptureMouse && !ImGui::GetIO().WantCaptureKeyboard) {
            camera.update(dt, events.data(), (int)events.size());
        }

        // --- Handle resize ---
        uint32_t new_w = backend.drawable_width();
        uint32_t new_h = backend.drawable_height();
        if (new_w != w || new_h != h) {
            w = new_w; h = new_h;
            half_w = w / 2; half_h = h / 2;
            backend.resize_texture(tex_current_color, half_w, half_h);
            backend.resize_texture(tex_current_depth, half_w, half_h);
            backend.resize_texture(tex_output, w, h);
            backend.resize_texture(tex_history_a, w, h);
            backend.resize_texture(tex_history_b, w, h);
        }

        // --- Build uniforms ---
        FrameUniforms uniforms = {};
        uniforms.time = time;
        uniforms.delta_time = dt;
        uniforms.frame_index = frame_index;
        uniforms.resolution[0] = (float)w;
        uniforms.resolution[1] = (float)h;
        uniforms.inv_resolution[0] = 1.0f / (float)w;
        uniforms.inv_resolution[1] = 1.0f / (float)h;
        // Camera vectors
        camera.get_vectors(uniforms.camera_pos, uniforms.camera_fwd,
                           uniforms.camera_up);
        // NOTE: camera_pos should be set separately, get_vectors signature
        // needs to be: get_vectors(fwd, up, right), pos is camera.pos directly
        memcpy(uniforms.camera_pos, camera.pos, sizeof(float) * 3);
        float fwd[3], up[3], right[3];
        camera.get_vectors(fwd, up, right);
        memcpy(uniforms.camera_fwd, fwd, sizeof(float) * 3);
        memcpy(uniforms.camera_up, up, sizeof(float) * 3);
        memcpy(uniforms.camera_right, right, sizeof(float) * 3);
        uniforms.camera_fov = camera.fov;
        // View-projection matrices
        float vp[16], prev_vp[16], inv_vp[16];
        camera.get_view_proj((float)w / (float)h, 0.001f, 1000.0f, vp);
        memcpy(uniforms.view_proj, vp, sizeof(float) * 16);
        // prev_view_proj: store last frame's VP (need to keep a copy)
        // inv_view_proj: invert vp
        // ... (matrix math — use a small inline mat4 utility, no library needed)

        // Jitter (Halton 2,3 sequence, in full-res pixel units, centered)
        uniforms.jitter[0] = halton(frame_index, 2) - 0.5f;
        uniforms.jitter[1] = halton(frame_index, 3) - 0.5f;

        // Pack shader params into uniform buffer
        const auto& params = shaders.get_params("raymarch.metal");
        uniforms.param_count = (uint32_t)params.size();
        for (int i = 0; i < (int)params.size() && i < 32; i++) {
            memcpy(uniforms.params[i], params[i].current_val, sizeof(float) * 4);
        }

        backend.write_buffer(buf_uniforms, &uniforms, sizeof(uniforms));

        // --- Render ---
        backend.begin_frame();

        // Pass 1: Raymarch (half-res)
        int rm_pipeline = shaders.get_pipeline("raymarch_kernel");
        if (rm_pipeline >= 0) {
            backend.dispatch({
                .pipeline_id = rm_pipeline,
                .grid_width = half_w,
                .grid_height = half_h,
                .threadgroup_w = 16,
                .threadgroup_h = 16,
                .textures = {tex_current_color, tex_current_depth},
                .buffers = {buf_uniforms}
            });
        }

        // Pass 2: Present / resolve to full-res output
        // In scaffold, this is a simple bilinear upsample + post-processing.
        // Will become the full TAA reconstruct pass later.
        int present_pipeline = shaders.get_pipeline("present_kernel");
        if (present_pipeline >= 0) {
            int history_read = ping ? tex_history_a : tex_history_b;
            int history_write = ping ? tex_history_b : tex_history_a;
            backend.dispatch({
                .pipeline_id = present_pipeline,
                .grid_width = w,
                .grid_height = h,
                .threadgroup_w = 16,
                .threadgroup_h = 16,
                .textures = {tex_current_color, tex_current_depth,
                             history_read, tex_output},
                .buffers = {buf_uniforms}
            });
            ping = !ping;
        }

        // Blit output to screen
        backend.blit_to_screen(tex_output);

        // ImGui
        ImGui_ImplMetal_NewFrame(/* render pass descriptor */);
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        // GUI content
        ImGui::Begin("Fractal Engine");
        ImGui::Text("%.1f fps (%.2f ms)", 1.0f / dt, dt * 1000.0f);
        render_shader_errors(shaders);
        ImGui::Separator();
        render_shader_params(shaders.get_params_mut("raymarch.metal"));
        ImGui::End();

        ImGui::Render();
        backend.render_imgui();

        backend.end_frame();

        frame_index++;
    }

    // Cleanup
    ImGui_ImplMetal_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    backend.shutdown();
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
```

---

## Starter Shaders

### common.metal

```metal
#include <metal_stdlib>
using namespace metal;

// ---- Uniform struct (must mirror FrameUniforms in types.h exactly) ----
struct FrameUniforms {
    float  time;
    float  delta_time;
    uint   frame_index;
    uint   _pad0;

    float2 resolution;
    float2 inv_resolution;

    float2 mouse;
    float2 mouse_click;

    float3 camera_pos;    float _pad1;
    float3 camera_fwd;    float _pad2;
    float3 camera_up;     float _pad3;
    float3 camera_right;
    float  camera_fov;

    float4x4 view_proj;
    float4x4 prev_view_proj;
    float4x4 inv_view_proj;

    float2 jitter;
    float2 _pad4;

    float4 params[32];
    uint   param_count;
    uint3  _pad5;
};

// ---- SDF Primitives ----
float sd_sphere(float3 p, float r) {
    return length(p) - r;
}

float sd_box(float3 p, float3 b) {
    float3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

// ---- Fractal Utilities ----
// Box fold: reflects p into [-fold_limit, fold_limit] per component
float3 box_fold(float3 p, float fold_limit) {
    return clamp(p, -fold_limit, fold_limit) * 2.0 - p;
}

// Sphere fold: inverts through a sphere
void sphere_fold(thread float3& p, thread float& dr, float min_r2, float fixed_r2) {
    float r2 = dot(p, p);
    if (r2 < min_r2) {
        float ratio = fixed_r2 / min_r2;
        p *= ratio;
        dr *= ratio;
    } else if (r2 < fixed_r2) {
        float ratio = fixed_r2 / r2;
        p *= ratio;
        dr *= ratio;
    }
}

// ---- Halton sequence (for CPU-side jitter, but useful in shader too) ----
float halton(uint index, uint base) {
    float f = 1.0;
    float r = 0.0;
    uint i = index;
    while (i > 0) {
        f /= float(base);
        r += f * float(i % base);
        i /= base;
    }
    return r;
}

// ---- Ray generation ----
struct Ray {
    float3 origin;
    float3 direction;
};

Ray make_camera_ray(float2 pixel_pos, constant FrameUniforms& frame) {
    float2 ndc = (pixel_pos * frame.inv_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;  // Metal texture coords: y-down

    float aspect = frame.resolution.x * frame.inv_resolution.y;
    float half_fov = tan(frame.camera_fov * 0.5);

    float3 dir = normalize(
        frame.camera_fwd +
        frame.camera_right * ndc.x * half_fov * aspect +
        frame.camera_up * ndc.y * half_fov
    );

    Ray r;
    r.origin = frame.camera_pos;
    r.direction = dir;
    return r;
}
```

### raymarch.metal

```metal
// @param fold_scale float 1.0 3.0 2.0
// @param min_radius float 0.01 1.0 0.25
// @param fixed_radius float 0.5 3.0 1.0
// @param iterations int 1 20 10
// @param de_scale float 0.5 4.0 2.5
// @param fog_density float 0.001 0.2 0.03
// @param fog_color color3 0 0 0 1 1 1 0.02 0.02 0.05
// @param surface_color color3 0 0 0 1 1 1 0.9 0.6 0.3

// Distance estimator: Mandelbox
float DE(float3 p, constant FrameUniforms& frame) {
    float fold_scale   = frame.params[0].x;
    float min_radius2  = frame.params[1].x * frame.params[1].x;
    float fixed_radius2 = frame.params[2].x * frame.params[2].x;
    int   iters        = int(frame.params[3].x);
    float scale        = frame.params[4].x;

    float3 z = p;
    float dr = 1.0;

    for (int i = 0; i < iters; i++) {
        z = box_fold(z, 1.0);
        sphere_fold(z, dr, min_radius2, fixed_radius2);
        z = z * scale + p;
        dr = dr * abs(scale) + 1.0;
    }

    return length(z) / abs(dr);
}

kernel void raymarch_kernel(
    texture2d<float, access::write>   out_color  [[texture(0)]],
    texture2d<float, access::write>   out_depth  [[texture(1)]],
    constant FrameUniforms&           frame      [[buffer(0)]],
    uint2                             gid        [[thread_position_in_grid]]
) {
    uint2 half_res = uint2(out_color.get_width(), out_color.get_height());
    if (gid.x >= half_res.x || gid.y >= half_res.y) return;

    // Compute full-res pixel position with jitter
    float2 full_pixel = (float2(gid) + 0.5) * 2.0 + frame.jitter;

    Ray ray = make_camera_ray(full_pixel, frame);

    // Raymarch
    float t = 0.0;
    float d = 0.0;
    int max_steps = 128;
    float min_dist = 0.0001;
    float max_dist = 100.0;

    for (int i = 0; i < max_steps; i++) {
        float3 p = ray.origin + ray.direction * t;
        d = DE(p, frame);
        if (d < min_dist * t) break;  // distance-proportional threshold
        if (t > max_dist) break;
        t += d;
    }

    // Output
    float3 color = float3(0.0);
    float depth = max_dist;

    if (t < max_dist) {
        depth = t;
        float3 hit_pos = ray.origin + ray.direction * t;

        // Simple normal via central differences
        float eps = 0.0001 * t;
        float3 n = normalize(float3(
            DE(hit_pos + float3(eps, 0, 0), frame) - DE(hit_pos - float3(eps, 0, 0), frame),
            DE(hit_pos + float3(0, eps, 0), frame) - DE(hit_pos - float3(0, eps, 0), frame),
            DE(hit_pos + float3(0, 0, eps), frame) - DE(hit_pos - float3(0, 0, eps), frame)
        ));

        // Basic hemisphere lighting
        float3 light_dir = normalize(float3(0.5, 1.0, -0.3));
        float ndl = max(dot(n, light_dir), 0.0);
        float ambient = 0.15;

        float3 surface_col = frame.params[7].xyz;  // surface_color
        color = surface_col * (ndl + ambient);
    }

    // Fog
    float fog_density = frame.params[5].x;
    float3 fog_color  = frame.params[6].xyz;
    float fog = 1.0 - exp(-t * fog_density);
    color = mix(color, fog_color, fog);

    out_color.write(float4(color, 1.0), gid);
    out_depth.write(float4(depth, 0, 0, 0), gid);
}
```

### present.metal

```metal
// @param noise_amount float 0.0 0.1 0.02
// @param exposure float 0.1 5.0 1.0

// Simple hash for noise
float hash12(float2 p) {
    float3 p3 = fract(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

kernel void present_kernel(
    texture2d<float, access::read>    in_color    [[texture(0)]],
    texture2d<float, access::read>    in_depth    [[texture(1)]],
    texture2d<float, access::read>    history     [[texture(2)]],
    texture2d<float, access::write>   output      [[texture(3)]],
    constant FrameUniforms&           frame       [[buffer(0)]],
    uint2                             gid         [[thread_position_in_grid]]
) {
    uint2 full_res = uint2(output.get_width(), output.get_height());
    if (gid.x >= full_res.x || gid.y >= full_res.y) return;

    // For scaffold: simple bilinear fetch from half-res source
    // This will become the full TAA reconstruction pass.
    float2 half_uv = float2(gid) * 0.5;
    uint2 half_coord = uint2(clamp(half_uv, float2(0), float2(in_color.get_width() - 1, in_color.get_height() - 1)));

    float4 color = in_color.read(half_coord);

    // Exposure
    float exposure = frame.params[1].x;  // present shader params start after raymarch params
    // NOTE: present.metal has its own param indices starting from 0 in its own param block.
    // The main loop should pack present params into a separate section or use a second buffer.
    // For scaffold simplicity, hardcode exposure and noise:
    // TODO: separate param buffers per shader
    color.rgb *= 1.0;  // placeholder for exposure

    // Film grain noise
    float noise_amount = 0.02;
    float noise = hash12(float2(gid) + frame.time * 137.0) * 2.0 - 1.0;
    color.rgb += noise * noise_amount;

    // Clamp output
    color = max(color, float4(0.0));
    color.a = 1.0;

    output.write(color, gid);
}
```

---

## Build System (CMakeLists.txt)

```cmake
cmake_minimum_required(VERSION 3.24)
project(fractal-engine LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD 20)

# --- SDL3 ---
include(FetchContent)
FetchContent_Declare(
    SDL3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG        main  # or a specific release tag
)
set(SDL_SHARED OFF)
set(SDL_STATIC ON)
FetchContent_MakeAvailable(SDL3)

# --- Dear ImGui ---
# Clone imgui into third_party/imgui manually or via FetchContent.
# Using manual approach here for simplicity.
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/third_party/imgui)
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
    ${IMGUI_DIR}/backends/imgui_impl_metal.mm
)

# --- Main target ---
add_executable(fractal-engine
    src/main.cpp
    src/metal_backend.mm
    src/shader_manager.cpp
    src/gui.cpp
    ${IMGUI_SOURCES}
)

target_include_directories(fractal-engine PRIVATE
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    src/
)

target_link_libraries(fractal-engine PRIVATE
    SDL3::SDL3-static
    "-framework Metal"
    "-framework MetalKit"
    "-framework QuartzCore"
    "-framework Cocoa"
    "-framework IOKit"
)

# Copy shaders to build directory for hot-reload
add_custom_command(TARGET fractal-engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_SOURCE_DIR}/shaders
        $<TARGET_FILE_DIR:fractal-engine>/shaders
)
```

---

## State Persistence (state_serializer.h / state_serializer.cpp)

### Overview

All application state (camera pose, shader parameters) is persisted to a JSON file next to the executable (`state.json`). State is saved automatically whenever any value changes and loaded on startup.

### What is Serialized

```json
{
  "camera": {
    "pos": [0, 0, -3],
    "yaw": 0.0,
    "pitch": 0.0,
    "fov": 1.2,
    "speed": 2.0,
    "sensitivity": 0.003
  },
  "shaders": {
    "raymarch.metal": {
      "radius": [1.0, 0, 0, 0],
      "surface_color": [0.9, 0.6, 0.3, 0]
    }
  }
}
```

Shader params are keyed by name so they survive reordering in the shader source. Values are stored as 4-element arrays (the full float4 slot) regardless of component count.

### Change Detection

Each frame, compare current camera + param values against the last-saved snapshot. If anything differs, write the file. This is cheap (memcmp on small structs) and avoids unnecessary disk writes when idle.

### Dependencies

- **nlohmann/json** (header-only, via FetchContent)

---

## Utility Functions Needed

These are small enough to go in a `math_util.h` or inline in `types.h`:

```cpp
// Halton sequence (low-discrepancy, for jitter)
float halton(uint32_t index, uint32_t base) {
    float f = 1.0f, r = 0.0f;
    uint32_t i = index;
    while (i > 0) {
        f /= (float)base;
        r += f * (float)(i % base);
        i /= base;
    }
    return r;
}

// Minimal 4x4 matrix math (column-major, matching Metal)
// Only need: identity, perspective, look_at, multiply, invert
// ~100 lines total. No need for GLM or similar.
namespace mat4 {
    void identity(float* out);
    void perspective(float fov_y, float aspect, float near, float far, float* out);
    void look_at(const float* eye, const float* center, const float* up, float* out);
    void multiply(const float* a, const float* b, float* out);
    void invert(const float* m, float* out);
}
```

---

## Implementation Order for Agent

1. **CMakeLists.txt + fetch dependencies.** Get it building with an empty main that opens an SDL3 window. Verify Metal layer is created.

2. **metal_backend.mm — init, begin_frame, end_frame, blit_to_screen.** Create device, command queue, get drawable, present. At this point you should see a cleared window.

3. **metal_backend.mm — compile_kernel, dispatch.** Compile a trivial kernel that writes solid red to a texture. Dispatch it. Blit to screen. Verify red screen.

4. **ImGui integration.** Init ImGui SDL3 + Metal backends. Render a test window. Verify overlay on top of the red screen.

5. **shader_manager.cpp — file loading, param parsing, compilation.** Load `raymarch.metal` and `present.metal`. Parse params. Compile. Dispatch raymarch → present → screen. You should see the Mandelbox.

6. **shader_manager.cpp — file watcher.** Poll timestamps, recompile on change. Verify: edit a color in `raymarch.metal`, save, see it update live.

7. **gui.cpp — auto-generated param sliders.** Wire parsed params to ImGui. Verify: move a slider, see the fractal change.

8. **Camera.** WASD + mouse. Wire into uniforms. Fly around the Mandelbox.

9. **Jitter + history buffer ping-pong.** Not TAA yet, just verify jitter is applied (image should subtly shimmer) and history textures are allocated and swappable.

At step 9 the scaffold is complete. Everything after this is the actual renderer work (TAA reconstruction, hierarchical traversal, breadcrumbs) which builds on this foundation.

---

## Notes for Agent

- **All Metal API calls go in metal_backend.mm. Nowhere else.** The rest of the codebase is pure C++.
- **Do not use any math library (GLM, Eigen, etc.).** The matrix math needed is ~4 functions. Write them inline.
- **nlohmann/json is the JSON library (via FetchContent).** Used for state persistence (camera + shader params). The `@param` comment parser remains simple string splitting — no JSON there.
- **Do not use any file watcher library.** `std::filesystem::last_write_time` polled at 2Hz is sufficient.
- **The `FrameUniforms` struct must be byte-identical in C++ and Metal.** Use `alignas(16)` on the C++ side and verify with `static_assert(sizeof(FrameUniforms) == expected)`. Metal's `float4x4` is 64 bytes, column-major. The C++ side stores these as `float[16]` column-major.
- **The shaders/ directory is symlinked into the build dir.** Edits to source shaders are instantly visible to the running app. No copy step needed.
- **Metal compilation errors should never crash the app.** Always keep the last-good pipeline and display the error in ImGui.
- **`fastMathEnabled = YES` in compile options.** Essential for fractal DE performance. The precision loss is irrelevant for artistic rendering.
- **Threadgroup size 16×16 = 256 threads.** This is the sweet spot for Apple Silicon compute. Don't go larger without benchmarking.
