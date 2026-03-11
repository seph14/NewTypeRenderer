#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "NewTypeRenderer.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace ci;
using namespace ci::app;
using namespace luisa;
using namespace luisa::compute;
using namespace newtype::gl_interop;

class IntegrationCinderLuisaApp : public App {
public:
    void setup() override;
    void update() override;
    void draw() override;
    void cleanup() override;
    void keyDown(KeyEvent event) override;
    void resize() override;

private:
    // Renderer for LuisaCompute to Cinder integration
    std::unique_ptr<NewTypeRenderer> renderer;

    // Compiled shader for rendering
    Shader2D<Image<float>, float, luisa::uint2> render_shader;

    // Time-varying parameters for animation
    float time = 0.0f;
    int frame_count = 0;
};

// Simple shader: renders a colorful gradient pattern based on position and time
Kernel2D render_gradient = [](ImageFloat image, Float time, UInt2 resolution) {
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
    image.write(coord, make_float4(r + wave, g + wave, b + wave, 1.0f));
};

void IntegrationCinderLuisaApp::setup() {
    // Create renderer (1280x720 default size)
    renderer = std::make_unique<NewTypeRenderer>(
        getWindowWidth(), getWindowHeight());
    renderer->initialize(TextureType::Float32);

    // Compile the compute kernel using DSL
    render_shader = NewTypeRenderer::device().compile(render_gradient);

    std::cout << "IntegrationCinderLuisaApp: Setup complete with DSL kernel!" << std::endl;
}

void IntegrationCinderLuisaApp::update() {
    frame_count++;
    time = frame_count * 0.016f;  // Approximate time at 60fps
}

void IntegrationCinderLuisaApp::draw() {
    gl::clear(Color(0.1f, 0.1f, 0.15f));

    // Begin frame - get render target
    auto& frame = renderer->beginFrame();

    // Run compute kernel to render to LuisaCompute Image
    // Kernel renders a colorful animated gradient pattern
    NewTypeRenderer::stream() << render_shader(frame.render_target, time,
                                  luisa::make_uint2(static_cast<uint>(renderer->width()),
                                                    static_cast<uint>(renderer->height())))
                        .dispatch(renderer->width(), renderer->height());

    // End frame - copy from LuisaCompute to Cinder GL texture
    renderer->endFrame();

    // Draw the rendered texture to screen
    gl::draw(renderer->currentTexture(), getWindowBounds());

    // Draw status text
    gl::drawString("Cinder + LuisaCompute Integration - CUDA-OpenGL Interop with DSL",
                   vec2(10, 10), Color(1, 1, 1));
    gl::drawString("Frame: " + std::to_string(frame_count),
                   vec2(10, 30), Color(0.7f, 0.7f, 1.0f));
    gl::drawString("Rendering: GPU Kernel (CUDA) -> GL Texture via cudaGraphicsGLRegisterImage",
                   vec2(10, 50), Color(0.5f, 0.8f, 1.0f));
}

void IntegrationCinderLuisaApp::cleanup() {
    // renderer is automatically cleaned up by unique_ptr
    /*if (luisa_context) {
        delete luisa_context;
        luisa_context = nullptr;
    }*/
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
