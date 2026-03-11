#pragma once

#include "LuisaGLInterop.h"
#include <luisa/luisa-compute.h>
#include <cinder/gl/Texture.h>
#include <memory>

namespace newtype::gl_interop {

/**
 * @brief Frame resources for triple-buffered rendering
 *
 * Holds a Cinder texture, LuisaCompute render target, and CUDA-OpenGL interop handler
 * for a single frame buffer. Use multiple instances for pipelined rendering.
 */
struct FrameResource {
    ci::gl::Texture2dRef            texture;        // Cinder GL texture for display
    luisa::compute::Image<float>    render_target;  // LuisaCompute Image for rendering
    std::unique_ptr<LuisaGLInterop> interop;        // CUDA-OpenGL interop handler
    uint64_t                        fence_value = 0;// For synchronization
};

enum class TextureType {
    Float32 = 0, Float16 = 1, Int8 = 2
};

/**
 * @brief Manages LuisaCompute to Cinder rendering pipeline
 *
 * This class handles:
 * 1. Creating Cinder textures with proper format for CUDA interop
 * 2. Registering textures with CUDA for zero-copy access
 * 3. Managing LuisaCompute Image render targets
 * 4. Copying rendered data from LuisaCompute to Cinder textures
 *
 * Usage:
 * @code
 *   NewTypeRenderer renderer(1920, 1080, luisa_device);
 *
 *   // In setup:
 *   renderer.initialize();
 *
 *   // In render loop:
 *   auto& frame = renderer.begin_frame(luisa_stream);
 *
 *   // Render to frame.render_target using LuisaCompute
 *   // ...
 *
 *   renderer.end_frame(luisa_stream);
 *
 *   // Draw to screen:
 *   ci::gl::draw(frame.texture);
 * @endcode
 */
class NewTypeRenderer {
public:
    /**
     * @brief Construct renderer with specified dimensions
     * @param width Render target width
     * @param height Render target height
     * @param device LuisaCompute device
     */
    NewTypeRenderer(uint32_t width, uint32_t height);
    NewTypeRenderer(ci::ivec2 size);

    ~NewTypeRenderer();

    // Non-copyable, non-movable (manages GPU resources)
    NewTypeRenderer(const NewTypeRenderer&) = delete;
    NewTypeRenderer& operator=(const NewTypeRenderer&) = delete;
    NewTypeRenderer(NewTypeRenderer&&) = delete;
    NewTypeRenderer& operator=(NewTypeRenderer&&) = delete;

    /**
     * @brief Initialize all GPU resources
     * Call this after construction and before any rendering
     */
    void initialize(TextureType type = TextureType::Float32);

    /**
     * @brief Begin a new frame, returns the frame resources to use
     * @return FrameResource containing render target and texture
     */
    [[nodiscard]] FrameResource& beginFrame();

    /**
     * @brief End the current frame, performs copy from LuisaCompute to GL texture
     */
    void endFrame();

    /**
     * @brief Get the current frame's texture for rendering in Cinder
     */
    [[nodiscard]] const ci::gl::Texture2dRef& currentTexture() const;

    /**
     * @brief Get the current frame's LuisaCompute render target
     */
    [[nodiscard]] const luisa::compute::Image<float>& currentRenderTarget() const;

    /**
     * @brief Resize all frame resources
     * @param width New width
     * @param height New height
     */
    void resize(uint32_t width, uint32_t height);

    /**
     * @brief Get render width
     */
    [[nodiscard]] uint32_t width() const noexcept { return _width; }

    /**
     * @brief Get render height
     */
    [[nodiscard]] uint32_t height() const noexcept { return _height; }

    /**
     * @brief Get the LuisaCompute device
     */
    [[nodiscard]] static luisa::compute::Device& device() noexcept { return _device; }
    /**
     * @brief Get the LuisaCompute context
     */
    [[nodiscard]] static luisa::compute::Context* ctx() noexcept { return _context; }
    /**
     * @brief Get the LuisaCompute stream
     */
    [[nodiscard]] static luisa::compute::Stream& stream() noexcept { return _stream; }

private:
    void initLuisaContext();
    void createFrameResources(TextureType type);
    void destroyFrameResources();
    
    // dimension
    uint32_t _width  = 0;
    uint32_t _height = 0;

    // Double buffering for pipelined rendering
    static constexpr size_t FRAME_COUNT = 2;
    std::array<std::unique_ptr<FrameResource>, FRAME_COUNT> _frames;
    TextureType     _type;
    size_t          _current_frame_index = 0;
    size_t          _ready_frame_index   = 0;  // Frame ready for display

protected:
    // LuisaCompute context and device
    static luisa::compute::Device   _device;
    static luisa::compute::Context* _context;
    static luisa::compute::Stream   _stream;
};

/**
 * @brief Helper to copy a LuisaCompute Image to a CUDA-OpenGL mapped array
 *
 * This function handles the copy operation from a LuisaCompute Image to
 * a CUDA array obtained from CUDA-OpenGL graphics interop.
 *
 * @param stream LuisaCompute stream
 * @param source Source LuisaCompute Image
 * @param dest_array Destination CUDA array (from CUDA-OpenGL interop)
 * @param width Image width
 * @param height Image height
 */
void copyImageToGlArray(
    luisa::compute::Stream& stream,
    const luisa::compute::Image<float>& source,
    cudaArray_t dest_array,
    uint32_t width,
    uint32_t height);

} // namespace luisa::gl_interop
