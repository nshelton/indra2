#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <SDL3/SDL.h>
#import <SDL3/SDL_metal.h>

#include "metal_backend.h"
#include "imgui.h"
#include "imgui_impl_metal.h"
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
    MTLRenderPassDescriptor*    imgui_render_pass;

    // Resource pools
    std::vector<id<MTLTexture>>              textures;
    std::vector<id<MTLBuffer>>               buffers;
    std::vector<id<MTLComputePipelineState>> pipelines;

    // Track texture descs for resize
    std::vector<TextureDesc> texture_descs;
};

// ---- Construction / destruction ----

MetalBackend::MetalBackend() : impl(std::make_unique<Impl>()) {}
MetalBackend::~MetalBackend() { shutdown(); }

// ---- Init ----

bool MetalBackend::init(SDL_Window* window) {
    impl->device = MTLCreateSystemDefaultDevice();
    if (!impl->device) return false;

    impl->command_queue = [impl->device newCommandQueue];

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
    ImGui_ImplMetal_NewFrame(impl->imgui_render_pass);
}

// ---- Frame lifecycle ----

void MetalBackend::begin_frame() {
    impl->current_drawable = [impl->metal_layer nextDrawable];
    impl->current_cmd_buffer = [impl->command_queue commandBuffer];
}

void MetalBackend::dispatch(const DispatchParams& params) {
    if (params.pipeline_id < 0 || params.pipeline_id >= (int)impl->pipelines.size()) return;

    id<MTLComputeCommandEncoder> enc = [impl->current_cmd_buffer computeCommandEncoder];
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

    id<MTLBlitCommandEncoder> blit = [impl->current_cmd_buffer blitCommandEncoder];

    id<MTLTexture> src = impl->textures[source_texture_id];
    id<MTLTexture> dst = impl->current_drawable.texture;

    // If sizes match, direct copy. Otherwise copy the overlapping region.
    NSUInteger w = MIN(src.width, dst.width);
    NSUInteger h = MIN(src.height, dst.height);

    [blit copyFromTexture:src
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(w, h, 1)
                toTexture:dst
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];

    [blit endEncoding];
}

void MetalBackend::render_imgui() {
    if (!impl->current_drawable) return;

    impl->imgui_render_pass.colorAttachments[0].texture = impl->current_drawable.texture;

    id<MTLRenderCommandEncoder> enc = [impl->current_cmd_buffer renderCommandEncoderWithDescriptor:impl->imgui_render_pass];

    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), impl->current_cmd_buffer, enc);

    [enc endEncoding];
}

void MetalBackend::end_frame() {
    if (impl->current_drawable) {
        [impl->current_cmd_buffer presentDrawable:impl->current_drawable];
    }
    [impl->current_cmd_buffer commit];
    impl->current_drawable = nil;
    impl->current_cmd_buffer = nil;
}

// ---- Getters ----

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

