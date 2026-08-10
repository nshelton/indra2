#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <SDL3/SDL.h>
#import <SDL3/SDL_metal.h>

#include "metal_backend.h"
#include "imgui.h"
#include "imgui_impl_metal.h"
#include <atomic>
#include <mutex>
#include <vector>
#include <string>

// ---- Impl ----

struct MetalBackend::Impl {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         command_queue;
    CAMetalLayer*               metal_layer;
    SDL_MetalView               metal_view;

    // Per-frame state
    id<CAMetalDrawable>         current_drawable;
    id<MTLCommandBuffer>        current_cmd_buffer;
    id<MTLCommandBuffer>        last_cmd_buffer;   // retained for wait_last_frame
    MTLRenderPassDescriptor*    imgui_render_pass;

    // Resource pools
    std::vector<id<MTLTexture>>              textures;
    std::vector<id<MTLBuffer>>               buffers;
    std::vector<id<MTLComputePipelineState>> pipelines;

    // Track texture descs for resize
    std::vector<TextureDesc> texture_descs;

    // Written by the command buffer completed-handler (GPU-internal queue)
    std::atomic<double> gpu_ms{0.0};

    // Per-pass GPU timing via stage-boundary timestamp samples. One sample
    // buffer per in-flight frame, so the frame being resolved never shares
    // storage with the one being encoded.
    static constexpr int TS_MAX_SAMPLES = 32;   // 16 passes/frame
    id<MTLCounterSampleBuffer> ts_buffers[MAX_FRAMES_IN_FLIGHT] = {};
    bool ts_supported = false;
    int  ts_slot = 0;                    // ring index, advanced in begin_frame
    int  ts_cursor = 0;                  // samples used this frame
    std::vector<std::string> ts_labels;  // label per sampled pass this frame
    std::mutex ts_mutex;
    std::vector<std::pair<std::string, double>> ts_results;  // last completed frame

    // Throttles the CPU to MAX_FRAMES_IN_FLIGHT. Without it the CPU can lap
    // the uniform ring and rewrite a slot the GPU is still reading — which
    // interactively only perturbs a converging average, but makes an offline
    // render irreproducible.
    dispatch_semaphore_t frame_sem = nil;
};

// ---- Construction / destruction ----

MetalBackend::MetalBackend() : impl(std::make_unique<Impl>()) {}
MetalBackend::~MetalBackend() { shutdown(); }

// ---- Init ----

bool MetalBackend::init(SDL_Window* window) {
    impl->device = MTLCreateSystemDefaultDevice();
    if (!impl->device) return false;

    impl->command_queue = [impl->device newCommandQueue];
    impl->frame_sem = dispatch_semaphore_create(MAX_FRAMES_IN_FLIGHT);

    // Timestamp sampling at compute-pass boundaries (Apple silicon supports
    // exactly this point). Missing support degrades to gpu_frame_ms only.
    if ([impl->device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
        id<MTLCounterSet> ts_set = nil;
        for (id<MTLCounterSet> cs in impl->device.counterSets) {
            if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) { ts_set = cs; break; }
        }
        if (ts_set) {
            MTLCounterSampleBufferDescriptor* d = [[MTLCounterSampleBufferDescriptor alloc] init];
            d.counterSet = ts_set;
            d.storageMode = MTLStorageModeShared;
            d.sampleCount = Impl::TS_MAX_SAMPLES;
            bool ok = true;
            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                NSError* err = nil;
                impl->ts_buffers[i] = [impl->device newCounterSampleBufferWithDescriptor:d error:&err];
                if (!impl->ts_buffers[i]) { ok = false; break; }
            }
            impl->ts_supported = ok;
        }
    }

    // Create Metal view and extract layer
    impl->metal_view = SDL_Metal_CreateView(window);
    if (!impl->metal_view) return false;

    impl->metal_layer = (__bridge CAMetalLayer*)SDL_Metal_GetLayer(impl->metal_view);
    impl->metal_layer.device = impl->device;
    impl->metal_layer.pixelFormat = MTLPixelFormatRGBA16Float;
    impl->metal_layer.framebufferOnly = NO;

    // ImGui render pass descriptor (reused each frame)
    impl->imgui_render_pass = [MTLRenderPassDescriptor new];
    impl->imgui_render_pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    impl->imgui_render_pass.colorAttachments[0].storeAction = MTLStoreActionStore;

    return true;
}

void MetalBackend::shutdown() {
    // Drain the pipeline before releasing: libdispatch traps if a semaphore is
    // destroyed while its count is below the value it was created with.
    if (impl->frame_sem) {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            dispatch_semaphore_wait(impl->frame_sem, DISPATCH_TIME_FOREVER);
        }
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            dispatch_semaphore_signal(impl->frame_sem);
        }
        impl->frame_sem = nil;
    }
    impl->last_cmd_buffer = nil;
    if (impl->metal_view) {
        SDL_Metal_DestroyView(impl->metal_view);
        impl->metal_view = nullptr;
    }
    impl->textures.clear();
    impl->buffers.clear();
    impl->pipelines.clear();
    impl->device = nil;
    impl->command_queue = nil;
}

// ---- Textures ----

static MTLPixelFormat to_mtl_format(TextureDesc::Format f) {
    switch (f) {
        case TextureDesc::RGBA16Float: return MTLPixelFormatRGBA16Float;
        case TextureDesc::RGBA32Float: return MTLPixelFormatRGBA32Float;
        case TextureDesc::R32Float:    return MTLPixelFormatR32Float;
        case TextureDesc::R32Uint:     return MTLPixelFormatR32Uint;
    }
    return MTLPixelFormatRGBA16Float;
}

int MetalBackend::create_texture(const TextureDesc& desc) {
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:to_mtl_format(desc.format)
                                                                                  width:desc.width
                                                                                 height:desc.height
                                                                              mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    td.storageMode = MTLStorageModePrivate;

    id<MTLTexture> tex = [impl->device newTextureWithDescriptor:td];
    if (!tex) return -1;
    tex.label = @(desc.name.c_str());

    int id = (int)impl->textures.size();
    impl->textures.push_back(tex);
    impl->texture_descs.push_back(desc);
    return id;
}

void MetalBackend::destroy_texture(int texture_id) {
    if (texture_id >= 0 && texture_id < (int)impl->textures.size()) {
        impl->textures[texture_id] = nil;
    }
}

void MetalBackend::resize_texture(int texture_id, uint32_t width, uint32_t height) {
    if (texture_id < 0 || texture_id >= (int)impl->textures.size()) return;

    auto& desc = impl->texture_descs[texture_id];
    desc.width = width;
    desc.height = height;

    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:to_mtl_format(desc.format)
                                                                                  width:width
                                                                                 height:height
                                                                              mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    td.storageMode = MTLStorageModePrivate;

    id<MTLTexture> tex = [impl->device newTextureWithDescriptor:td];
    tex.label = @(desc.name.c_str());
    impl->textures[texture_id] = tex;
}

// ---- Buffers ----

int MetalBackend::create_buffer(size_t size_bytes, const std::string& label) {
    id<MTLBuffer> buf = [impl->device newBufferWithLength:size_bytes
                                                  options:MTLResourceStorageModeShared];
    if (!buf) return -1;
    buf.label = @(label.c_str());

    int id = (int)impl->buffers.size();
    impl->buffers.push_back(buf);
    return id;
}

void MetalBackend::destroy_buffer(int buffer_id) {
    if (buffer_id >= 0 && buffer_id < (int)impl->buffers.size()) {
        impl->buffers[buffer_id] = nil;
    }
}

void MetalBackend::write_buffer(int buffer_id, const void* data, size_t size, size_t offset) {
    if (buffer_id < 0 || buffer_id >= (int)impl->buffers.size()) return;
    id<MTLBuffer> buf = impl->buffers[buffer_id];
    memcpy((uint8_t*)[buf contents] + offset, data, size);
}

const void* MetalBackend::buffer_contents(int buffer_id) const {
    if (buffer_id < 0 || buffer_id >= (int)impl->buffers.size()) return nullptr;
    return [impl->buffers[buffer_id] contents];
}

// ---- Shader compilation ----

int MetalBackend::compile_kernel(const std::string& msl_source,
                                  const std::string& kernel_function_name,
                                  std::string& error_out) {
    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = MTLLanguageVersion3_1;
    opts.mathMode = MTLMathModeFast;

    id<MTLLibrary> lib = [impl->device newLibraryWithSource:@(msl_source.c_str())
                                                     options:opts
                                                       error:&error];
    if (!lib) {
        error_out = [[error localizedDescription] UTF8String];
        return -1;
    }

    id<MTLFunction> func = [lib newFunctionWithName:@(kernel_function_name.c_str())];
    if (!func) {
        error_out = "Function '" + kernel_function_name + "' not found in source";
        return -1;
    }

    id<MTLComputePipelineState> pso = [impl->device newComputePipelineStateWithFunction:func
                                                                                  error:&error];
    if (!pso) {
        error_out = [[error localizedDescription] UTF8String];
        return -1;
    }

    int pipeline_id = (int)impl->pipelines.size();
    impl->pipelines.push_back(pso);
    return pipeline_id;
}

// ---- ImGui lifecycle ----

void MetalBackend::imgui_init() {
    ImGui_ImplMetal_Init(impl->device);
}

void MetalBackend::imgui_shutdown() {
    ImGui_ImplMetal_Shutdown();
}

void MetalBackend::imgui_new_frame() {
    // NewFrame snapshots the descriptor's texture sampleCount for pipeline
    // creation — on frame 0 no texture is attached yet (render_imgui assigns
    // it later), which fails pipeline creation with rasterSampleCount = 0.
    if (impl->current_drawable) {
        impl->imgui_render_pass.colorAttachments[0].texture = impl->current_drawable.texture;
    }
    ImGui_ImplMetal_NewFrame(impl->imgui_render_pass);
}

// ---- Frame lifecycle ----

void MetalBackend::begin_frame() {
    // Paired with the signal in end_frame's completed handler. Every path out
    // of the frame must reach end_frame or this deadlocks.
    if (impl->frame_sem) dispatch_semaphore_wait(impl->frame_sem, DISPATCH_TIME_FOREVER);
    impl->current_drawable = [impl->metal_layer nextDrawable];
    impl->current_cmd_buffer = [impl->command_queue commandBuffer];

    // The wait above guarantees the frame that last used this ring slot's
    // timestamp buffer has fully retired (and resolved it).
    impl->ts_slot = (impl->ts_slot + 1) % MAX_FRAMES_IN_FLIGHT;
    impl->ts_cursor = 0;
    impl->ts_labels.clear();
}

void MetalBackend::dispatch(const DispatchParams& params) {
    if (params.pipeline_id < 0 || params.pipeline_id >= (int)impl->pipelines.size()) return;

    // Bracket the pass with timestamp samples when the device can.
    id<MTLComputeCommandEncoder> enc;
    if (impl->ts_supported && impl->ts_cursor + 2 <= Impl::TS_MAX_SAMPLES) {
        MTLComputePassDescriptor* pd = [MTLComputePassDescriptor computePassDescriptor];
        MTLComputePassSampleBufferAttachmentDescriptor* att = pd.sampleBufferAttachments[0];
        att.sampleBuffer = impl->ts_buffers[impl->ts_slot];
        att.startOfEncoderSampleIndex = impl->ts_cursor;
        att.endOfEncoderSampleIndex = impl->ts_cursor + 1;
        impl->ts_cursor += 2;
        impl->ts_labels.push_back(params.label ? params.label : "pass");
        enc = [impl->current_cmd_buffer computeCommandEncoderWithDescriptor:pd];
    } else {
        enc = [impl->current_cmd_buffer computeCommandEncoder];
    }
    [enc setComputePipelineState:impl->pipelines[params.pipeline_id]];

    for (int i = 0; i < (int)params.textures.size(); i++) {
        [enc setTexture:impl->textures[params.textures[i]] atIndex:i];
    }
    for (int i = 0; i < (int)params.buffers.size(); i++) {
        [enc setBuffer:impl->buffers[params.buffers[i]] offset:0 atIndex:i];
    }

    MTLSize grid = MTLSizeMake(params.grid_width, params.grid_height, 1);
    MTLSize group = MTLSizeMake(params.threadgroup_w, params.threadgroup_h, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
    [enc endEncoding];
}

void MetalBackend::blit_to_screen(int source_texture_id) {
    if (source_texture_id < 0 || source_texture_id >= (int)impl->textures.size()) return;
    if (!impl->current_drawable) return;

    id<MTLTexture> src = impl->textures[source_texture_id];
    id<MTLTexture> dst = impl->current_drawable.texture;

    // An off-size source (render scale, resolution override) covers only
    // part of the drawable, and the ImGui pass loads previous contents — so
    // the uncovered border would keep stale pixels and moving UI windows
    // would smear trails there. Clear the whole drawable first. Exact-size
    // sources overwrite every pixel and skip it.
    if (src.width != dst.width || src.height != dst.height) {
        MTLRenderPassDescriptor* clear_pass = [MTLRenderPassDescriptor renderPassDescriptor];
        clear_pass.colorAttachments[0].texture = dst;
        clear_pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        clear_pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        clear_pass.colorAttachments[0].clearColor = MTLClearColorMake(0.06, 0.06, 0.07, 1.0);
        id<MTLRenderCommandEncoder> clr =
            [impl->current_cmd_buffer renderCommandEncoderWithDescriptor:clear_pass];
        [clr endEncoding];
    }

    // Sampled like the compute passes. Encoder timestamps start when the
    // encoder executes — after the wait for the compositor to release the
    // drawable — so this measures the copy itself, and the vsync stall
    // stays out of every measured bucket (it's the frame-total remainder).
    id<MTLBlitCommandEncoder> blit;
    if (impl->ts_supported && impl->ts_cursor + 2 <= Impl::TS_MAX_SAMPLES) {
        MTLBlitPassDescriptor* pd = [MTLBlitPassDescriptor blitPassDescriptor];
        MTLBlitPassSampleBufferAttachmentDescriptor* att = pd.sampleBufferAttachments[0];
        att.sampleBuffer = impl->ts_buffers[impl->ts_slot];
        att.startOfEncoderSampleIndex = impl->ts_cursor;
        att.endOfEncoderSampleIndex = impl->ts_cursor + 1;
        impl->ts_cursor += 2;
        impl->ts_labels.push_back("blit");
        blit = [impl->current_cmd_buffer blitCommandEncoderWithDescriptor:pd];
    } else {
        blit = [impl->current_cmd_buffer blitCommandEncoder];
    }

    // Centered both ways: a smaller render sits in the middle of the window,
    // a larger one shows its center crop (blits can't scale).
    NSUInteger w = MIN(src.width, dst.width);
    NSUInteger h = MIN(src.height, dst.height);

    [blit copyFromTexture:src
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake((src.width - w) / 2, (src.height - h) / 2, 0)
               sourceSize:MTLSizeMake(w, h, 1)
                toTexture:dst
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:MTLOriginMake((dst.width - w) / 2, (dst.height - h) / 2, 0)];

    [blit endEncoding];
}

void MetalBackend::copy_texture_to_buffer(int texture_id, int buffer_id,
                                          uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                                          uint32_t bytes_per_pixel, uint32_t bytes_per_row) {
    if (texture_id < 0 || texture_id >= (int)impl->textures.size()) return;
    if (buffer_id < 0 || buffer_id >= (int)impl->buffers.size()) return;

    uint32_t row = bytes_per_row ? bytes_per_row : w * bytes_per_pixel;

    id<MTLBlitCommandEncoder> blit = [impl->current_cmd_buffer blitCommandEncoder];
    [blit copyFromTexture:impl->textures[texture_id]
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(x, y, 0)
               sourceSize:MTLSizeMake(w, h, 1)
                 toBuffer:impl->buffers[buffer_id]
        destinationOffset:0
   destinationBytesPerRow:row
 destinationBytesPerImage:row * h];
    [blit endEncoding];
}

void MetalBackend::render_imgui() {
    if (!impl->current_drawable) return;

    impl->imgui_render_pass.colorAttachments[0].texture = impl->current_drawable.texture;

    // Fragment stage only. On a TBDR GPU the vertex stage runs as soon as
    // it's scheduled — potentially frames before the fragment stage, which
    // is what waits for the compositor to release the drawable — so a
    // vertex-start..fragment-end span measures the whole drawable wait
    // (pipeline depth x refresh interval), not this pass's work. Fragment
    // start fires after that wait. The descriptor is reused across frames:
    // rewrite every field, including parking the vertex index back on
    // DontSample so an older configuration can't linger.
    if (impl->ts_supported && impl->ts_cursor + 2 <= Impl::TS_MAX_SAMPLES) {
        MTLRenderPassSampleBufferAttachmentDescriptor* att =
            impl->imgui_render_pass.sampleBufferAttachments[0];
        att.sampleBuffer = impl->ts_buffers[impl->ts_slot];
        att.startOfVertexSampleIndex = MTLCounterDontSample;
        att.endOfVertexSampleIndex = MTLCounterDontSample;
        att.startOfFragmentSampleIndex = impl->ts_cursor;
        att.endOfFragmentSampleIndex = impl->ts_cursor + 1;
        impl->ts_cursor += 2;
        impl->ts_labels.push_back("imgui");
    }

    id<MTLRenderCommandEncoder> enc = [impl->current_cmd_buffer renderCommandEncoderWithDescriptor:impl->imgui_render_pass];

    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), impl->current_cmd_buffer, enc);

    [enc endEncoding];
}

void MetalBackend::end_frame() {
    if (impl->current_drawable) {
        [impl->current_cmd_buffer presentDrawable:impl->current_drawable];
    }
    Impl* imp = impl.get();
    dispatch_semaphore_t sem = impl->frame_sem;
    // Copies captured by the block: the members mutate as the next frame
    // encodes while this handler waits on the GPU.
    int ts_slot = impl->ts_slot;
    int ts_count = impl->ts_cursor / 2;
    std::vector<std::string> ts_labels = impl->ts_labels;
    [impl->current_cmd_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        imp->gpu_ms.store((cb.GPUEndTime - cb.GPUStartTime) * 1000.0);
        // Resolve this frame's pass timestamps (Apple GPUs report them in
        // nanoseconds). Runs before the semaphore signal, so the slot can't
        // be re-encoded while we read it.
        if (imp->ts_supported && ts_count > 0) {
            NSData* data = [imp->ts_buffers[ts_slot]
                resolveCounterRange:NSMakeRange(0, (NSUInteger)ts_count * 2)];
            if (data && data.length >= sizeof(MTLCounterResultTimestamp) * ts_count * 2) {
                auto* ts = (const MTLCounterResultTimestamp*)data.bytes;
                std::vector<std::pair<std::string, double>> res;
                res.reserve(ts_count);
                for (int i = 0; i < ts_count; i++) {
                    uint64_t t0 = ts[i * 2].timestamp;
                    uint64_t t1 = ts[i * 2 + 1].timestamp;
                    bool valid = t0 != MTLCounterErrorValue && t1 != MTLCounterErrorValue && t1 > t0;
                    res.emplace_back(ts_labels[i], valid ? (double)(t1 - t0) / 1.0e6 : 0.0);
                }
                std::lock_guard<std::mutex> lock(imp->ts_mutex);
                imp->ts_results = std::move(res);
            }
        }
        if (sem) dispatch_semaphore_signal(sem);
    }];
    [impl->current_cmd_buffer commit];
    impl->last_cmd_buffer = impl->current_cmd_buffer;
    impl->current_drawable = nil;
    impl->current_cmd_buffer = nil;
}

std::vector<std::pair<std::string, double>> MetalBackend::gpu_pass_ms() const {
    std::lock_guard<std::mutex> lock(impl->ts_mutex);
    return impl->ts_results;
}

void MetalBackend::wait_last_frame() {
    if (!impl->last_cmd_buffer) return;
    [impl->last_cmd_buffer waitUntilCompleted];
}

void MetalBackend::set_vsync(bool enabled) {
    impl->metal_layer.displaySyncEnabled = enabled ? YES : NO;
}

// ---- Getters ----

float MetalBackend::gpu_frame_ms() const {
    return (float)impl->gpu_ms.load();
}

uint32_t MetalBackend::drawable_width() const {
    CGSize size = impl->metal_layer.drawableSize;
    return (uint32_t)size.width;
}

uint32_t MetalBackend::drawable_height() const {
    CGSize size = impl->metal_layer.drawableSize;
    return (uint32_t)size.height;
}

void* MetalBackend::raw_device() const {
    return (__bridge void*)impl->device;
}

void* MetalBackend::raw_command_queue() const {
    return (__bridge void*)impl->command_queue;
}

