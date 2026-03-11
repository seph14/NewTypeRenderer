#include "luisaCinderRenderer.h"
#include <luisa/luisa-compute.h>
// #include <luisa/dsl/sugar.h>  // Disabled - not needed for runtime API only
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda.h>
#include <iostream>

namespace luisa::gl_interop {

LuisaCinderRenderer::LuisaCinderRenderer(uint32_t width, uint32_t height, luisa::compute::Device& device)
    : _width(width),
      _height(height),
      _device(device),
      _type(TextureType::Float32){

    std::cout << "LuisaCinderRenderer: Created " << width << "x" << height << " renderer" << std::endl;
}

LuisaCinderRenderer::~LuisaCinderRenderer() {
    destroy_frame_resources();
}

void LuisaCinderRenderer::initialize(TextureType type) {
    _type = type;
    create_frame_resources(type);
}

void LuisaCinderRenderer::create_frame_resources(TextureType type) {
    static const vector<luisa::compute::PixelStorage> luFmt = {
        luisa::compute::PixelStorage::FLOAT4,
        luisa::compute::PixelStorage::HALF4,
        luisa::compute::PixelStorage::BYTE4
    };
    static const vector<uint32_t> ciFmt = {
        GL_RGBA32F,
        GL_RGBA16F,
        GL_RGBA8
    };

    // Format: RGBA32F (4 channels of float)
    const auto format = luFmt[static_cast<int>(type)];
    const auto glFmt  = ciFmt[static_cast<int>(type)];

    for (size_t i = 0; i < FRAME_COUNT; ++i) {
        _frames[i]  = std::make_unique<FrameResource>();
        auto& frame = _frames[i];

        // Create Cinder texture with given type format (important for CUDA interop)
        ci::gl::Texture2d::Format fmt;
        fmt.setInternalFormat(glFmt);
        //fmt.setDataType(GL_FLOAT);
        fmt.wrap(GL_CLAMP_TO_EDGE);
        fmt.setMinFilter(GL_LINEAR);
        fmt.setMagFilter(GL_LINEAR);
        frame->texture = ci::gl::Texture2d::create(_width, _height, fmt);

        // Create LuisaCompute Image as render target
        frame->render_target = _device.create_image<float>(
            format,
            luisa::uint2(_width, _height));

        // Create CUDA-OpenGL interop handler
        frame->interop = std::make_unique<LuisaGLInterop>(
            frame->texture,
            cudaGraphicsMapFlagsWriteDiscard);
        //_frames[i] = frame;

        std::cout << "Created frame " << i << ": texture=" << frame->texture->getId()
                  << ", luisa_image=" << frame->render_target.handle() << std::endl;
    }

    std::cout << "LuisaCinderRenderer: Created " << FRAME_COUNT << " frame resources" << std::endl;
}

void LuisaCinderRenderer::destroy_frame_resources() {
    for (size_t i = 0; i < FRAME_COUNT; ++i) {
        if (_frames[i]) {
            // interop is automatically cleaned up by unique_ptr
            // render_target is automatically cleaned up by Image destructor
            // texture is reference-counted by Cinder
            _frames[i].reset();
        }
    }
}

FrameResource& LuisaCinderRenderer::begin_frame(luisa::compute::Stream& stream) {
    // Advance to next frame (round-robin through frame resources)
    _current_frame_index = (_current_frame_index + 1) % FRAME_COUNT;
    return *_frames[_current_frame_index];
}

void LuisaCinderRenderer::end_frame(luisa::compute::Stream& stream) {
    auto& current_frame = *_frames[_current_frame_index];

    // Map the GL texture for CUDA access
    auto map = current_frame.interop->map(stream);
    auto cuda_array = map.get_mipmapped_array(0, 0);

    // Copy from LuisaCompute Image to CUDA array (and thus to GL texture)
    copy_image_to_gl_array(stream, current_frame.render_target,
                           cuda_array.get(), _width, _height);

    // Unmap happens automatically when 'map' goes out of scope

    // Mark this frame as ready for display
    _ready_frame_index = _current_frame_index;
}

const ci::gl::Texture2dRef& LuisaCinderRenderer::current_texture() const {
    return _frames[_ready_frame_index]->texture;
}

const luisa::compute::Image<float>& LuisaCinderRenderer::current_render_target() const {
    return _frames[_current_frame_index]->render_target;
}

void LuisaCinderRenderer::resize(uint32_t width, uint32_t height) {
    if (width == _width && height == _height) {
        return;  // No resize needed
    }

    std::cout << "LuisaCinderRenderer: Resizing to " << width << "x" << height << std::endl;

    _width = width;
    _height = height;

    // Recreate all frame resources with new size
    destroy_frame_resources();
    create_frame_resources (_type);
}

void copy_image_to_gl_array(
    luisa::compute::Stream& stream,
    const luisa::compute::Image<float>& source,
    cudaArray_t dest_array,
    uint32_t width,
    uint32_t height) {

    // The native_handle() returns the CUDA array (CUarray) directly, not a pointer to CUDATexture
    // From cuda_device.cpp: native_handle = reinterpret_cast<void *>(p->handle())
    // where handle() returns _base_array, which for single-level images IS the CUarray
    CUarray src_array = reinterpret_cast<CUarray>(source.native_handle());

    if (src_array == nullptr) {
        std::cerr << "Warning: LuisaCompute Image has invalid CUDA array (native_handle is null)" << std::endl;
        return;
    }

    // For RGBA32F format: 4 channels * 4 bytes per float = 16 bytes per pixel
    constexpr size_t bytes_per_pixel = 4 * sizeof(float);  // RGBA32F = 16 bytes
    size_t width_in_bytes = width * bytes_per_pixel;

    // Get CUDA stream handle from LuisaCompute Stream
    CUstream cuda_stream = reinterpret_cast<CUstream>(stream.native_handle());

    // Use CUDA Driver API to copy from CUDA array (LuisaCompute) to CUDA array (OpenGL)
    CUDA_MEMCPY2D copy_params = {};
    copy_params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy_params.srcArray = src_array;
    copy_params.srcXInBytes = 0;
    copy_params.srcY = 0;
    copy_params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy_params.dstArray = reinterpret_cast<CUarray>(dest_array);
    copy_params.dstXInBytes = 0;
    copy_params.dstY = 0;
    copy_params.WidthInBytes = width_in_bytes;
    copy_params.Height = height;

    CUresult result = cuMemcpy2DAsync(&copy_params, cuda_stream);

    if (result != CUDA_SUCCESS) {
        const char* error_str = nullptr;
        cuGetErrorString(result, &error_str);
        std::cerr << "cuMemcpy2DAsync failed: " << error_str << " (result=" << result << ")" << std::endl;
    }
}

} // namespace luisa::gl_interop
