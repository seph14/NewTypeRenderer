#include "NewTypeRenderer.h"
#include "cinder/Log.h"
#include "cinder/CinderAssert.h"
#include <luisa/luisa-compute.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda.h>
#include <iostream>

namespace newtype::gl_interop {
using namespace luisa;

// Define static members
luisa::compute::Device   NewTypeRenderer::_device;
luisa::compute::Context* NewTypeRenderer::_context = nullptr;
luisa::compute::Stream   NewTypeRenderer::_stream;

NewTypeRenderer::NewTypeRenderer(uint32_t width, uint32_t height)
    : _width (width),
      _height(height),
      _type  (TextureType::Float32){

    initLuisaContext();
    CI_LOG_I("NewTypeRenderer: Created " << width << "x" << height << " renderer");
}

NewTypeRenderer::NewTypeRenderer(ci::ivec2 size)
    : _width(size.x),
    _height (size.y),
    _type   (TextureType::Float32) {

    initLuisaContext();
    CI_LOG_I("NewTypeRenderer: Created " << size.x << "x" << size.y << " renderer");
}

NewTypeRenderer::~NewTypeRenderer() {
    destroyFrameResources();
    
    if (_context) {
        delete _context;
        _context = nullptr;
    }
}

void NewTypeRenderer::initLuisaContext() {
    // Get the full executable path for LuisaCompute Context
    char exePath[MAX_PATH];
    GetModuleFileNameA(NULL, exePath, MAX_PATH);
    //std::cout << "Executable path: " << exePath << std::endl;

    // Initialize LuisaCompute with full executable path
    _context = new (std::nothrow) compute::Context(exePath);
    CI_ASSERT_MSG(_context != nullptr, "Failed to create LuisaCompute Context");

    // Try to create a device (CUDA backend preferred)
    try {
        _device = _context->create_device("cuda");
        if (_device) {
            _stream = _device.create_stream();
            CI_LOG_I("LuisaCompute CUDA backend initialized successfully");
        } else {
            CI_LOG_E("Failed to create CUDA device");
            return;
        }
    } catch (const std::exception& e) {
        CI_LOG_EXCEPTION("Exception creating CUDA device", e);
    } catch (...) {
        CI_LOG_E("Unknown exception creating CUDA device");
    }

}

void NewTypeRenderer::initialize(TextureType type) {
    _type = type;
    createFrameResources(type);
}

void NewTypeRenderer::createFrameResources(TextureType type) {
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
        fmt.setDataType     (GL_FLOAT);
        fmt.wrap            (GL_CLAMP_TO_EDGE);
        fmt.setMinFilter    (GL_LINEAR);
        fmt.setMagFilter    (GL_LINEAR);
        frame->texture = ci::gl::Texture2d::create(_width, _height, fmt);

        // Create LuisaCompute Image as render target
        frame->render_target = _device.create_image<float>(
            format,
            luisa::uint2(_width, _height));

        // Create CUDA-OpenGL interop handler
        frame->interop = std::make_unique<LuisaGLInterop>(
            frame->texture,
            cudaGraphicsMapFlagsWriteDiscard);
        
        CI_LOG_D("Created frame " << i << ": texture=" << frame->texture->getId()
            << ", luisa_image=" << frame->render_target.handle());
    }

    CI_LOG_D("NewTypeRenderer: Created " << FRAME_COUNT << " frame resources");
}

void NewTypeRenderer::destroyFrameResources() {
    for (size_t i = 0; i < FRAME_COUNT; ++i) {
        if (_frames[i]) {
            // interop is automatically cleaned up by unique_ptr
            // render_target is automatically cleaned up by Image destructor
            // texture is reference-counted by Cinder
            _frames[i].reset();
        }
    }
}

FrameResource& NewTypeRenderer::beginFrame() {
    _current_frame_index = (_current_frame_index + 1) % FRAME_COUNT;
    return *_frames[_current_frame_index];
}

void NewTypeRenderer::endFrame() {
    auto& current_frame = *_frames[_current_frame_index];

    // Map the GL texture for CUDA access
    auto map        = current_frame.interop->map(_stream);
    auto cuda_array = map.get_mipmapped_array(0, 0);
    // Copy from LuisaCompute Image to CUDA array (and thus to GL texture)
    copyImageToGlArray(_stream, current_frame.render_target,
                       cuda_array.get(), _width, _height);

    // Unmap happens automatically when 'map' goes out of scope

    // Mark this frame as ready for display
    _ready_frame_index = _current_frame_index;
}

const ci::gl::Texture2dRef& NewTypeRenderer::currentTexture() const {
    return _frames[_ready_frame_index]->texture;
}

const luisa::compute::Image<float>& NewTypeRenderer::currentRenderTarget() const {
    return _frames[_current_frame_index]->render_target;
}

void NewTypeRenderer::resize(uint32_t width, uint32_t height) {
    if (width == _width && height == _height) 
        return;  // No resize needed

    CI_LOG_I("NewTypeRenderer: Resizing to " << width << "x" << height);
    _width  = width;
    _height = height;

    // Recreate all frame resources with new size
    destroyFrameResources();
    createFrameResources (_type);
}

void copyImageToGlArray(
    luisa::compute::Stream& stream,
    const luisa::compute::Image<float>& source,
    cudaArray_t dest_array,
    uint32_t width,
    uint32_t height) {

    // The native_handle() returns the CUDA array (CUarray) directly, not a pointer to CUDATexture
    // From cuda_device.cpp: native_handle = reinterpret_cast<void *>(p->handle())
    // where handle() returns _base_array, which for single-level images IS the CUarray
    CUarray src_array = reinterpret_cast<CUarray>(source.native_handle());
    CI_ASSERT_MSG(src_array != nullptr, 
        "Warning: LuisaCompute Image has invalid CUDA array (native_handle is null)");
    
    // For RGBA32F format: 4 channels * 4 bytes per float = 16 bytes per pixel
    constexpr size_t bytes_per_pixel = 4 * sizeof(float);  // RGBA32F = 16 bytes
    size_t width_in_bytes            = width * bytes_per_pixel;

    // Get CUDA stream handle from LuisaCompute Stream
    CUstream cuda_stream = reinterpret_cast<CUstream>(stream.native_handle());

    // Use CUDA Driver API to copy from CUDA array (LuisaCompute) to CUDA array (OpenGL)
    CUDA_MEMCPY2D copy_params   = {};
    copy_params.srcMemoryType   = CU_MEMORYTYPE_ARRAY;
    copy_params.srcArray        = src_array;
    copy_params.srcXInBytes     = 0;
    copy_params.srcY            = 0;
    copy_params.dstMemoryType   = CU_MEMORYTYPE_ARRAY;
    copy_params.dstArray        = reinterpret_cast<CUarray>(dest_array);
    copy_params.dstXInBytes     = 0;
    copy_params.dstY            = 0;
    copy_params.WidthInBytes    = width_in_bytes;
    copy_params.Height          = height;
    CUresult result = cuMemcpy2DAsync(&copy_params, cuda_stream);

    if (result != CUDA_SUCCESS) {
        const char* error_str = nullptr;
        cuGetErrorString(result, &error_str);
        std::cerr << "cuMemcpy2DAsync failed: " << error_str << " (result=" << result << ")" << std::endl;
    }
}

} // namespace luisa::gl_interop
