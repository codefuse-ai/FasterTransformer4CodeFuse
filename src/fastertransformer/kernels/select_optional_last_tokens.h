/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/utils/cuda_utils.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformer {

template<typename T>
void invokeSelectOptionalLastTokens(T*           logits,                // [batch_size, beam_width, vocab_size_padded]
                                    const int*   optional_last_tokens,  // [batch_size, max_optional_last_tokens_count]
                                    const int    batch_size,
                                    const int    beam_width,
                                    const int    vocab_size_padded,
                                    const int    max_optional_last_tokens_count,
                                    cudaStream_t stream);

}  // namespace fastertransformer