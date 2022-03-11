cmake_minimum_required(VERSION 3.15)

project("AI Dojo" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDNN_LIBRARY_PATH "$ENV{cuDNN}/lib/x64")
set(CUDNN_INCLUDE_PATH "$ENV{cuDNN}/include")
set(CMAKE_PREFIX_PATH "$ENV{LIBS_PATH}/LibTorch/Torch-1.10.0/libtorch")

find_package(Torch REQUIRED)

include_directories("${TORCH_INCLUDE_DIRS}")

include_directories(${CMAKE_SOURCE_DIR}/Utility)

# Add source to this project's executable.
add_executable(AIInter "main.cpp" "cnn.h" "cnn.cpp" "rnn.h" "rnn.cpp" "bi_rnn.h" "bi_rnn.cpp" )

# Link Torch Libraries
target_link_libraries(AIInter "${TORCH_LIBRARIES}")
target_link_libraries(AIInter ${CMAKE_SOURCE_DIR}/Utility EnvUtil)

file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET AIInter
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:AIInter>)