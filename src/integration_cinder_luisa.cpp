#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include <luisa/luisa-compute.h>
#include <iostream>
#include <windows.h>

using namespace ci;
using namespace ci::app;
using namespace luisa;
using namespace luisa::compute;

class IntegrationCinderLuisaApp : public App {
public:
    void setup() override;
    void update() override;
    void draw() override;
    void cleanup() override;
    void keyDown(KeyEvent event) override;
    void resize() override;

private:
    // Use pointer for Context since it's not default-constructible
    Context* luisa_context = nullptr;
    Device luisa_device;
    Stream luisa_stream;
    Image<float> render_target;
    int frame_count = 0;
};

void IntegrationCinderLuisaApp::setup() {
    // Get the full executable path for LuisaCompute Context
    char exePath[MAX_PATH];
    GetModuleFileNameA(NULL, exePath, MAX_PATH);
    std::string exeStr = app::getAppPath().string();
    std::cout << "Executable path: " << exePath << "-" << exeStr << std::endl;

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
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception creating CUDA device: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception creating CUDA device!" << std::endl;
    }
}

void IntegrationCinderLuisaApp::update()
{
    frame_count++;
}

void IntegrationCinderLuisaApp::draw()
{
    gl::clear(Color(0.1f, 0.1f, 0.15f));

    // Draw status text
    gl::drawString("Cinder + LuisaCompute Integration",
                   vec2(10, 10), Color(1, 1, 1));
    gl::drawString("Frame: " + std::to_string(frame_count),
                   vec2(10, 30), Color(0.7f, 0.7f, 1.0f));
}

void IntegrationCinderLuisaApp::cleanup()
{
    if (luisa_context) {
        delete luisa_context;
        luisa_context = nullptr;
    }
}

void IntegrationCinderLuisaApp::keyDown(KeyEvent event)
{
    if (event.getChar() == 'f') {
        setFullScreen(!isFullScreen());
    }
    if (event.getCode() == KeyEvent::KEY_ESCAPE) {
        quit();
    }
}

void IntegrationCinderLuisaApp::resize()
{
    // Handle window resize
}

CINDER_APP(IntegrationCinderLuisaApp, RendererGl, [](App::Settings* settings) {
    settings->setWindowSize (1280, 720);
    settings->setResizable  (true);
    settings->setTitle      ("Cinder + LuisaCompute Integration");
    settings->setConsoleWindowEnabled();
})
