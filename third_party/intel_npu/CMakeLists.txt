if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonIntelNPU ${CMAKE_CURRENT_SOURCE_DIR}/triton_intel_npu.cc)
  target_link_libraries(TritonIntelNPU PUBLIC PRIVATE Python3::Module pybind11::headers)
endif()