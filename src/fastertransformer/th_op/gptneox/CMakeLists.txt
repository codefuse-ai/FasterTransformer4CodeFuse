# Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if (BUILD_PYBIND)
    add_definitions("-DBUILD_PYBIND")

    find_package(pybind11 REQUIRED)
    find_package(Torch REQUIRED)

    add_subdirectory(utils)

    if(TORCH_PYTHON_LIBRARIES)
        set(TORCH_LIBRARIES ${TORCH_PYTHON_LIBRARIES} ${TORCH_LIBRARIES})
    endif()

    message("TORCH_LIBRARIES: ${TORCH_LIBRARIES}")

    add_library(th_gptneox SHARED GptNeoXOp.cc)
    set_property(TARGET th_gptneox PROPERTY POSITION_INDEPENDENT_CODE  ON)
    target_link_libraries(th_gptneox PRIVATE "${TORCH_LIBRARIES}" GptNeoX th_utils nccl_utils pybind11::module th_nccl_inherit_utils pybind_callback_utils)

else()

    add_library(th_gptneox STATIC GptNeoXOp.cc)
    set_property(TARGET th_gptneox PROPERTY POSITION_INDEPENDENT_CODE  ON)
    target_link_libraries(th_gptneox PRIVATE "${TORCH_LIBRARIES}" GptNeoX th_utils nccl_utils)

endif()
