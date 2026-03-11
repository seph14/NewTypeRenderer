#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "luisaCinderRenderer.h"

// Workaround for LuisaCompute DSL macro expansion issues
// Define _WITH_FMT as empty to prevent compilation errors in DSL headers
#define _WITH_FMT

#include <luisa/luisa-compute.h>
// #include <luisa/dsl/sugar.h>  // Disabled - DSL requires additional preprocessor setup

using namespace ci;
using namespace ci::app;
using namespace luisa;
using namespace luisa::compute;
using namespace luisa::gl_interop;

class IntegrationCinderLuisaApp : public App {
public:
    void setup() override;
    void update() override;
    void draw() override;
    void cleanup() override;
    void keyDown(KeyEvent event) override;
    void resize() override;

private:
    // LuisaCompute context and device
    Context* luisa_context = nullptr;
    Device luisa_device;
    Stream luisa_stream;

    // Renderer for LuisaCompute to Cinder integration
    std::unique_ptr<LuisaCinderRenderer> renderer;

    // Compute kernel for rendering
    //luisa::compute::Kernel2D<Image<float>, float, luisa::uint2>  render_kernel;
    // Time-varying parameters for animation
    float time = 0.0f;
    int frame_count = 0;
};

// Simple shader: renders a colorful gradient pattern based on position and time
// NOTE: Disabled - requires <luisa/dsl/sugar.h> which needs additional preprocessor setup
/*
void render_gradient(ImageFloat image, float time, luisa::uint2 resolution) {
    Var coord = dispatch_id().xy();

    // Normalize coordinates to [0, 1]
    Var uv = make_float2(coord) / make_float2(resolution);

    // Create a dynamic color pattern
    auto r = 0.5f + 0.5f * cos(uv.x * 3.14159f * 2.0f + time);
    auto g = 0.5f + 0.5f * cos(uv.y * 3.14159f * 2.0f + time * 0.7f);
    auto b = 0.5f + 0.5f * sin(uv.x * uv.y * 3.14159f * 4.0f + time * 1.3f);

    // Add some interesting wave pattern
    auto wave = sin(uv.x * 10.0f + time) * sin(uv.y * 10.0f + time) * 0.2f;

    // Store to image (RGBA format, stored as float4)
    image.write(coord, make_float4(r + wave, g + wave, b + wave, Var<float>(1.0f)));
}
*/

void IntegrationCinderLuisaApp::setup() {
    // Get the full executable path for LuisaCompute Context
    char exePath[MAX_PATH];
    GetModuleFileNameA(NULL, exePath, MAX_PATH);
    std::cout << "Executable path: " << exePath << std::endl;

    // Initialize LuisaCompute with full executable path
    luisa_context = new (std::nothrow) Context(exePath);

    if (!luisa_context) {
        std::cerr << "Failed to create LuisaCompute Context!" << std::endl;
        return;
    }

    // Try to create a device (CUDA backend preferred)
    try {
        luisa_device = luisa_context->create_device("cuda");
        if (luisa_device) {
            luisa_stream = luisa_device.create_stream();
            std::cout << "LuisaCompute CUDA backend initialized successfully!" << std::endl;
        } else {
            std::cerr << "Failed to create CUDA device!" << std::endl;
            return;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception creating CUDA device: " << e.what() << std::endl;
        return;
    } catch (...) {
        std::cerr << "Unknown exception creating CUDA device!" << std::endl;
        return;
    }

    // Create renderer (1280x720 default size)
    renderer = std::make_unique<LuisaCinderRenderer>(
        getWindowWidth(), getWindowHeight(), luisa_device);
    renderer->initialize(TextureType::Float32);

    // Create the compute kernel
   /* auto render_kernel = luisa_device.compile_kernel2D(
        render_gradient,
        luisa_device.impl()->get_library_compiler(),
        "render_gradient",
        luisa_device.impl()->get_shader_compiler()
    );*/

    std::cout << "IntegrationCinderLuisaApp: Setup complete!" << std::endl;
}

void IntegrationCinderLuisaApp::update() {
    frame_count++;
    time = frame_count * 0.016f;  // Approximate time at 60fps
}

void IntegrationCinderLuisaApp::draw() {
    gl::clear(Color(0.1f, 0.1f, 0.15f));

    // Begin frame - get render target
    auto& frame = renderer->begin_frame(luisa_stream);

    // Run compute kernel to render to LuisaCompute Image
    // Kernel renders a colorful animated gradient pattern
    /*luisa_stream << render_kernel(frame.render_target, time,
                                   make_uint2(renderer->width(), renderer->height()))
                   << luisa_stream.synchronize();*/

    // End frame - copy from LuisaCompute to Cinder GL texture
    renderer->end_frame(luisa_stream);

    // Draw the rendered texture to screen
    gl::draw(renderer->current_texture(), getWindowBounds());

    // Draw status text
    gl::drawString("Cinder + LuisaCompute Integration - CUDA-OpenGL Interop",
                   vec2(10, 10), Color(1, 1, 1));
    gl::drawString("Frame: " + std::to_string(frame_count),
                   vec2(10, 30), Color(0.7f, 0.7f, 1.0f));
    gl::drawString("Rendering: GPU Kernel (CUDA) -> GL Texture via cudaGraphicsGLRegisterImage",
                   vec2(10, 50), Color(0.5f, 0.8f, 1.0f));
}

void IntegrationCinderLuisaApp::cleanup() {
    // renderer is automatically cleaned up by unique_ptr
    if (luisa_context) {
        delete luisa_context;
        luisa_context = nullptr;
    }
}

void IntegrationCinderLuisaApp::keyDown(KeyEvent event) {
    if (event.getChar() == 'f') {
        setFullScreen(!isFullScreen());
    }
    if (event.getCode() == KeyEvent::KEY_ESCAPE) {
        quit();
    }
}

void IntegrationCinderLuisaApp::resize() {
    // Handle window resize
    if (renderer) {
        renderer->resize(getWindowWidth(), getWindowHeight());
    }
}

CINDER_APP(IntegrationCinderLuisaApp, RendererGl, [](App::Settings* settings) {
    settings->setWindowSize(1280, 720);
    settings->setResizable(true);
    settings->setTitle("Cinder + LuisaCompute - CUDA-OpenGL Interop Demo");
    settings->setConsoleWindowEnabled();
})
