cmake_minimum_required(VERSION 2.8)
project(online_models)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11 -fopenmp")
if (APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -undefined dynamic_lookup")
elseif (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()
set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
execute_process(COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_include())" OUTPUT_VARIABLE tf_inc)
include_directories(${tf_inc})


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/)
add_library(user_ops SHARED csv_iter_op.cc)
enable_testing()
add_test(NAME test_csv_iter COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/csv_iter_op_test.py)
