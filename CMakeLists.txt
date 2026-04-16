cmake_minimum_required(VERSION 3.24)
project(fractal-engine LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD 20)

# --- SDL3 via FetchContent ---
include(FetchContent)
FetchContent_Declare(
    SDL3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG        release-3.2.0
    GIT_SHALLOW    TRUE
)
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
set(SDL_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(SDL3)

# --- Dear ImGui via FetchContent ---
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.92.6-docking
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(imgui)

# --- nlohmann/json via FetchContent ---
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(json)

set(IMGUI_DIR ${imgui_SOURCE_DIR})
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
    ${IMGUI_DIR}/backends/imgui_impl_metal.mm
)

# --- Main target ---
add_executable(fractal-engine
    src/main.cpp
    src/metal_backend.mm
    src/shader_manager.cpp
    src/gui.cpp
    src/state_serializer.cpp
    ${IMGUI_SOURCES}
)

target_include_directories(fractal-engine PRIVATE
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    src/
)

target_link_libraries(fractal-engine PRIVATE
    SDL3::SDL3-static
    nlohmann_json::nlohmann_json
    "-framework Metal"
    "-framework MetalKit"
    "-framework QuartzCore"
    "-framework Cocoa"
    "-framework IOKit"
)

# Symlink shaders into build directory for hot-reload
add_custom_command(TARGET fractal-engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_SOURCE_DIR}/shaders
        $<TARGET_FILE_DIR:fractal-engine>/shaders
)
