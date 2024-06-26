cmake_minimum_required(VERSION 3.15)

project("AI Dojo" LANGUAGES CXX)

set(CUDNN_LIBRARY_PATH "$ENV{cuDNN}/lib/x64")
set(CUDNN_INCLUDE_PATH "$ENV{cuDNN}/include")
set(CMAKE_PREFIX_PATH "$ENV{LIBS_PATH}/LibTorch/Torch-1.10.0/libtorch")

find_package(Torch REQUIRED)

include_directories("${TORCH_INCLUDE_DIRS}")

include_directories(${CMAKE_SOURCE_DIR}/Utility)

# Add source to this project's executable.
add_executable(AIBasics "main.cpp" "tensor_basics.h" "tensor_basics.cpp" "lin_reg.h" "lin_reg.cpp" "log_reg.h" "log_reg.cpp" "ffnet.h" "ffnet.cpp")

# Link Torch Libraries
target_link_libraries(AIBasics "${TORCH_LIBRARIES}")
target_link_libraries(AIBasics ${CMAKE_SOURCE_DIR}/Utility EnvUtil)

file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET AIBasics
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:AIBasics>)