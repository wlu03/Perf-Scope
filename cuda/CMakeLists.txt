cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()
project(PerfScopeCuda LANGUAGES CUDA)

find_package(CUDAToolkit REQUIRED)
add_executable(perf-scope-cuda src/main.cu src/transpose.cu src/gemm.cu src/triad.cu src/softmax.cu)
target_compile_features(perf-scope-cuda PRIVATE cuda_std_17)
target_link_libraries(perf-scope-cuda PRIVATE CUDA::cublas)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME cuda_cli_help COMMAND perf-scope-cuda --help)
  add_test(
    NAME rejects_invalid_shape
    COMMAND perf-scope-cuda transpose naive
      --shape 32,0 --dtype fp32 --seed 42 --warmups 0 --iterations 1
  )
  set_tests_properties(rejects_invalid_shape PROPERTIES PASS_REGULAR_EXPRESSION "invalid --shape")
  foreach(variant naive tiled)
    foreach(shape 128,256 67,99)
      string(REPLACE "," "x" shape_label "${shape}")
      add_test(
        NAME transpose_${variant}_${shape_label}
        COMMAND perf-scope-cuda transpose ${variant}
          --shape ${shape} --dtype fp32 --seed 42 --warmups 2 --iterations 5
      )
      set_tests_properties(
        transpose_${variant}_${shape_label}
        PROPERTIES LABELS gpu PASS_REGULAR_EXPRESSION "\"correct\":true"
      )
    endforeach()
  endforeach()
  foreach(variant naive tiled cublas)
    foreach(shape 16,16,32 17,19,31)
      string(REPLACE "," "x" shape_label "${shape}")
      add_test(
        NAME gemm_${variant}_${shape_label}
        COMMAND perf-scope-cuda gemm ${variant}
          --shape ${shape} --dtype fp32 --seed 42 --warmups 2 --iterations 5
      )
      set_tests_properties(
        gemm_${variant}_${shape_label}
        PROPERTIES LABELS gpu PASS_REGULAR_EXPRESSION "\"correct\":true"
      )
    endforeach()
  endforeach()
  add_test(
    NAME triad_partial_block
    COMMAND perf-scope-cuda triad streaming
      --shape 10003 --dtype fp32 --seed 42 --warmups 2 --iterations 5
  )
  set_tests_properties(triad_partial_block PROPERTIES LABELS gpu PASS_REGULAR_EXPRESSION "\"correct\":true")
  foreach(variant fused multi-pass)
    foreach(shape 11,1 17,259)
      string(REPLACE "," "x" shape_label "${shape}")
      add_test(
        NAME softmax_${variant}_${shape_label}
        COMMAND perf-scope-cuda softmax ${variant}
          --shape ${shape} --dtype fp32 --seed 42 --warmups 2 --iterations 5
      )
      set_tests_properties(
        softmax_${variant}_${shape_label}
        PROPERTIES LABELS gpu PASS_REGULAR_EXPRESSION "\"correct\":true"
      )
    endforeach()
  endforeach()
endif()
