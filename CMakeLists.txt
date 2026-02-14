cmake_minimum_required(VERSION 3.21)
project(nature-of-nature LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# --- WebGPU (wgpu-native backend via eliemichel's distribution) ---
FetchContent_Declare(
    webgpu
    URL https://github.com/eliemichel/WebGPU-distribution/archive/refs/tags/wgpu-v0.19.4.1.zip
)
FetchContent_MakeAvailable(webgpu)

# --- GLFW ---
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glfw
    URL https://github.com/glfw/glfw/archive/refs/tags/3.4.zip
)
FetchContent_MakeAvailable(glfw)

# --- glfw3webgpu (GLFW ↔ WebGPU surface helper) ---
FetchContent_Declare(
    glfw3webgpu
    URL https://github.com/eliemichel/glfw3webgpu/archive/refs/tags/v1.0.1.zip
)
FetchContent_MakeAvailable(glfw3webgpu)

# --- Dear ImGui ---
FetchContent_Declare(
    imgui
    URL https://github.com/ocornut/imgui/archive/refs/tags/v1.91.8.zip
)
FetchContent_MakeAvailable(imgui)

# ImGui library (core + GLFW + WebGPU backends)
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_wgpu.cpp
)
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw webgpu)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_WEBGPU_BACKEND_WGPU=1)

# --- stb (header-only) ---
FetchContent_Declare(
    stb
    URL https://github.com/nothings/stb/archive/refs/heads/master.zip
)
FetchContent_MakeAvailable(stb)

# --- Main executable ---
file(GLOB_RECURSE SOURCES src/*.cpp src/*.h)
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE webgpu glfw glfw3webgpu imgui)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${stb_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Copy wgpu-native shared lib next to executable
target_copy_webgpu_binaries(${PROJECT_NAME})

# Ensure the executable finds libwgpu_native.dylib next to it
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_RPATH "@executable_path"
    INSTALL_RPATH "@executable_path"
)

# Copy shaders to build dir
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders
)
