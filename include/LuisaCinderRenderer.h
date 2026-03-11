#pragma once

#include "LuisaGLInterop.h"
#include <luisa/luisa-compute.h>
#include <cinder/gl/Texture.h>
#include <memory>

namespace luisa::gl_interop {

/**
 * @brief Frame resources for triple-buffered rendering
 *
 * Holds a Cinder texture, LuisaCompute render target, and CUDA-OpenGL interop handler
 * for a single frame buffer. Use multiple instances for pipelined rendering.
 */
struct FrameResource {
    ci::gl::Texture2dRef texture;           // Cinder GL texture for display
    luisa::compute::Image<float> render_target; // LuisaCompute Image for rendering
    std::unique_ptr<LuisaGLInterop> interop;   // CUDA-OpenGL interop handler
    uint64_t fence_value = 0;                 // For synchronization
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
 *   LuisaCinderRenderer renderer(1920, 1080, luisa_device);
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
class LuisaCinderRenderer {
public:
    /**
     * @brief Construct renderer with specified dimensions
     * @param width Render target width
     * @param height Render target height
     * @param device LuisaCompute device
     */
    LuisaCinderRenderer(uint32_t width, uint32_t height, luisa::compute::Device& device);

    ~LuisaCinderRenderer();

    // Non-copyable, non-movable (manages GPU resources)
    LuisaCinderRenderer(const LuisaCinderRenderer&) = delete;
    LuisaCinderRenderer& operator=(const LuisaCinderRenderer&) = delete;
    LuisaCinderRenderer(LuisaCinderRenderer&&) = delete;
    LuisaCinderRenderer& operator=(LuisaCinderRenderer&&) = delete;

    /**
     * @brief Initialize all GPU resources
     * Call this after construction and before any rendering
     */
    void initialize(TextureType type);

    /**
     * @brief Begin a new frame, returns the frame resources to use
     * @param stream LuisaCompute stream for rendering
     * @return FrameResource containing render target and texture
     */
    [[nodiscard]] FrameResource& begin_frame(luisa::compute::Stream& stream);

    /**
     * @brief End the current frame, performs copy from LuisaCompute to GL texture
     * @param stream LuisaCompute stream
     */
    void end_frame(luisa::compute::Stream& stream);

    /**
     * @brief Get the current frame's texture for rendering in Cinder
     */
    [[nodiscard]] const ci::gl::Texture2dRef& current_texture() const;

    /**
     * @brief Get the current frame's LuisaCompute render target
     */
    [[nodiscard]] const luisa::compute::Image<float>& current_render_target() const;

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
    [[nodiscard]] luisa::compute::Device& device() const noexcept { return _device; }

private:
    void create_frame_resources(TextureType type);
    void destroy_frame_resources();

    uint32_t _width = 0;
    uint32_t _height = 0;
    luisa::compute::Device& _device;

    // Triple buffering for pipelined rendering
    static constexpr size_t FRAME_COUNT = 3;
    std::array<std::unique_ptr<FrameResource>, FRAME_COUNT> _frames;
    TextureType _type;
    size_t _current_frame_index = 0;
    size_t _ready_frame_index = 0;  // Frame ready for display
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
void copy_image_to_gl_array(
    luisa::compute::Stream& stream,
    const luisa::compute::Image<float>& source,
    cudaArray_t dest_array,
    uint32_t width,
    uint32_t height);

} // namespace luisa::gl_interop
