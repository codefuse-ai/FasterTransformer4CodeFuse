add_library(th_nccl_inherit_utils STATIC nccl_inherit_utils.cc)
set_property(TARGET th_nccl_inherit_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(th_nccl_inherit_utils PRIVATE "${TORCH_LIBRARIES}" th_utils nccl_utils)

add_library(pybind_callback_utils STATIC pybind_callback_utils.cc)
set_property(TARGET pybind_callback_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(pybind_callback_utils PRIVATE "${TORCH_LIBRARIES}" th_utils)