# Configuration Guide

This guide explains how to configure the Cinder + LuisaCompute integration template for your development environment.

## Overview

The `IntegrationCinderLuisa.vcxproj.template` file uses MSBuild property macros instead of hardcoded paths. To use the template, you need to define these properties via a `Directory.Build.props` file or environment variables.

## Method 1: Directory.Build.props (Recommended)

Create a `Directory.Build.props` file in your project root (or solution root) with your specific paths:

```xml
<Project>
  <PropertyGroup>
    <!-- Cinder Paths -->
    <CinderRoot>C:\Users\YourName\Projects\Cinder</CinderRoot>
    <CinderInclude>$(CinderRoot)\include</CinderInclude>
    <CinderIncludeAngle>$(CinderRoot)\include\ANGLE</CinderIncludeAngle>
    <CinderLibDebug>$(CinderRoot)\lib\msw\x64\Debug_Shared\v143\</CinderLibDebug>
    <CinderLibRelease>$(CinderRoot)\lib\msw\x64\Release_Shared\v143\</CinderLibRelease>
    <CinderBinDebug>$(CinderRoot)\lib\msw\x64\Debug_Shared\v143</CinderBinDebug>
    <CinderBinRelease>$(CinderRoot)\lib\msw\x64\Release_Shared\v143</CinderBinRelease>

    <!-- LuisaCompute Paths -->
    <LuisaComputeRoot>D:\Projects\LuCompute\LuisaCompute</LuisaComputeRoot>
    <LuisaComputeInclude>$(LuisaComputeRoot)\include</LuisaComputeInclude>
    <LuisaComputeSpdlog>$(LuisaComputeRoot)\src\ext\spdlog\include</LuisaComputeSpdlog>
    <LuisaComputeXxHash>$(LuisaComputeRoot)\src\ext\xxHash</LuisaComputeXxHash>
    <LuisaComputeMagicEnum>$(LuisaComputeRoot)\src\ext\magic_enum\include</LuisaComputeMagicEnum>
    <LuisaComputeEASTL>$(LuisaComputeRoot)\src\ext\EASTL\include</LuisaComputeEASTL>
    <LuisaComputeEABase>$(LuisaComputeRoot)\src\ext\EASTL\packages\EABase\include\Common</LuisaComputeEABase>
    <LuisaComputeReproc>$(LuisaComputeRoot)\src\ext\reproc\reproc\include</LuisaComputeReproc>
    <LuisaComputeReprocXX>$(LuisaComputeRoot)\src\ext\reproc\reproc++\include</LuisaComputeReprocXX>
    <LuisaComputeMarl>$(LuisaComputeRoot)\src\ext\marl\include</LuisaComputeMarl>
    <LuisaComputeHalf>$(LuisaComputeRoot)\src\ext\half\include</LuisaComputeHalf>
    <LuisaComputeStb>$(LuisaComputeRoot)\src\ext\stb</LuisaComputeStb>
    <LuisaComputeLibDebug>$(LuisaComputeRoot)\build-cuda\lib\Debug\</LuisaComputeLibDebug>
    <LuisaComputeLibRelease>$(LuisaComputeRoot)\build-cuda\lib\Release\</LuisaComputeLibRelease>
    <LuisaComputeBinDebug>$(LuisaComputeRoot)\build-cuda\bin\Debug</LuisaComputeBinDebug>
    <LuisaComputeBinRelease>$(LuisaComputeRoot)\build-cuda\bin\Release</LuisaComputeBinRelease>

    <!-- CUDA Paths -->
    <CudaRoot>C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1</CudaRoot>
    <CudaInclude>$(CudaRoot)\include</CudaInclude>
    <CudaLib>$(CudaRoot)\lib\x64\cuda.lib;$(CudaRoot)\lib\x64\cudart.lib</CudaLib>
    <CudaBin>$(CudaRoot)\bin</CudaBin>

    <!-- Platform Toolset -->
    <PlatformToolset>v143</PlatformToolset>
  </PropertyGroup>
</Project>
```

Place this file at:
```
D:\Projects\LuCompute\Directory.Build.props
```

MSBuild will automatically pick up this file for all projects in the directory tree.

## Method 2: Environment Variables

Alternatively, set environment variables (less flexible for team development):

```batch
set CinderRoot=C:\Users\YourName\Projects\Cinder
set LuisaComputeRoot=D:\Projects\LuCompute\LuisaCompute
set CudaRoot=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
```

## Using the Template

1. Copy `IntegrationCinderLuisa.vcxproj.template` to a new `.vcxproj` file
2. Rename the project GUID and root namespace as needed
3. Ensure `Directory.Build.props` exists with your paths
4. Open the solution in Visual Studio

## Required Libraries

Ensure LuisaCompute is built with DSL enabled before using this template:

```batch
cd LuisaCompute
mkdir build-cuda
cd build-cuda
cmake .. -DCMAKE_BUILD_TYPE=Release -DLUISA_COMPUTE_ENABLE_DSL=ON
cmake --build . --config Release
```

## Architecture Overview

```
integration_cinder_luisa/
├── include/
│   ├── NewTypeRenderer.h      # Main renderer class (static context/device/stream)
│   ├── LuisaGLInterop.h       # CUDA-OpenGL interop utilities
│   └── Resources.h            # Windows resources
├── src/
│   ├── integration_cinder_luisa.cpp  # Main app with DSL shader example
│   ├── NewTypeRenderer.cpp    # Renderer implementation
│   └── LuisaGLInterop.cpp     # CUDA-OpenGL interop implementation
└── vc2022/
    ├── IntegrationCinderLuisa.vcxproj.template  # Boilerplate template
    └── Resources.rc           # Resource file
```

## Key Design Patterns

- **Static Context Pattern**: `NewTypeRenderer` holds static `Device`, `Context`, and `Stream` shared across all instances
- **Double Buffering**: Two `FrameResource` instances for pipelined rendering
- **CUDA-OpenGL Interop**: Zero-copy via `cudaGraphicsGLRegisterImage` + `cuMemcpy2DAsync`
- **DSL Support**: Compute shaders written in C++ DSL (requires `/Zc:preprocessor` flag)

## Build Configurations

| Configuration | Runtime Library | LuisaCompute Libs |
|--------------|----------------|-------------------|
| Debug x64    | MultiThreadedDebugDLL | luisa-*.lib from Debug |
| Release x64  | MultiThreadedDLL | luisa-*.lib from Release |
