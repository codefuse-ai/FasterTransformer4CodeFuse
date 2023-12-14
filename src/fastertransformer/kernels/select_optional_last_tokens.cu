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

#include "src/fastertransformer/kernels/select_optional_last_tokens.h"

namespace fastertransformer {

template<typename T>
__global__ void
select_optional_last_tokens(T*         logits,                       // [batch_size, beam_width, vocab_size_padded]
                            const int* optional_last_tokens,         // [batch_size, max_optional_last_tokens_count]
                            const int  batch_size,
                            const int  beam_width,
                            const int  vocab_size_padded,
                            const int  max_optional_last_tokens_count,
                            const int  shared_size)
{
    const int batch_idx  = blockIdx.x;
    const int beam_idx   = blockIdx.y;
    const int block_idx  = blockIdx.z;
    const int thread_idx = threadIdx.x;

    const int threads_per_block = blockDim.x;
    const int blocks_per_beam   = gridDim.z;
    const int threads_per_beam  = threads_per_block * blocks_per_beam;

    const int batch_beam_idx   = batch_idx * beam_width + beam_idx;
    const int block_thread_idx = block_idx * blocks_per_beam + thread_idx;

    const int vocab_size_per_block = shared_size * 8;
    const int block_field_start_idx = vocab_size_per_block * block_idx;

    const int thread_field_length = vocab_size_per_block / threads_per_block;

    const int thread_field_start_idx = block_field_start_idx + thread_field_length * thread_idx;
    const int thread_field_end_idx   = block_field_start_idx + thread_field_length * (thread_idx + 1);

    // vocab_size_per_block bits
    extern __shared__ char smem[];

    // init to 0
    memset(smem, 0, shared_size);

    __syncthreads();

    int offset = max_optional_last_tokens_count * batch_idx;
    for (int idx = offset; idx < offset + max_optional_last_tokens_count; idx ++) {

        int last_token = optional_last_tokens[idx];

        if (last_token >= 0 && last_token >= thread_field_start_idx && last_token < thread_field_end_idx) {
            int smem_idx_bit   = last_token - block_field_start_idx;
            int smem_idx_char  = smem_idx_bit / 8;
            int smem_idx_inner = smem_idx_bit % 8;
            smem[smem_idx_char] |= (1 << smem_idx_inner);
        }
    }

    __syncthreads();

    offset = batch_beam_idx * vocab_size_padded;
    for (int token_id = thread_field_start_idx; token_id < vocab_size_padded && token_id < thread_field_end_idx; token_id ++) {

        int smem_idx_bit   = token_id - block_field_start_idx;
        int smem_idx_char  = smem_idx_bit / 8;
        int smem_idx_inner = smem_idx_bit % 8;
        if ((smem[smem_idx_char] & (1 << smem_idx_inner)) == 0) {
            logits[offset + token_id] = static_cast<T>(-INFINITY);
        }
    }
}

template<typename T>
void invokeSelectOptionalLastTokens(T*           logits,                // [batch_size, beam_width, vocab_size_padded]
                                    const int*   optional_last_tokens,  // [batch_size, max_optional_last_tokens_count]
                                    const int    batch_size,
                                    const int    beam_width,
                                    const int    vocab_size_padded,
                                    const int    max_optional_last_tokens_count,
                                    cudaStream_t stream)
{
    const int blocks_per_beam   = 64;
    const int threads_per_block = 128;
    const int threads_per_beam  = threads_per_block * blocks_per_beam;

    const int divisor = threads_per_beam * 8;

    const int vocab_size_tmp       = (vocab_size_padded + divisor - 1) / divisor * divisor;
    const int vocab_size_per_block = vocab_size_tmp / blocks_per_beam;
    const int shared_size          = vocab_size_per_block / 8;

    dim3 grid(batch_size, beam_width, blocks_per_beam);
    dim3 block(threads_per_block);

    select_optional_last_tokens<<<grid, block, shared_size, stream>>>(logits,
                                                                      optional_last_tokens,
                                                                      batch_size,
                                                                      beam_width,
                                                                      vocab_size_padded,
                                                                      max_optional_last_tokens_count,
                                                                      shared_size);

    sync_check_cuda_error();
}

template void
invokeSelectOptionalLastTokens(half*        logits,                // [batch_size, beam_width, vocab_size_padded]
                               const int*   optional_last_tokens,  // [batch_size, max_optional_last_tokens_count]
                               const int    batch_size,
                               const int    beam_width,
                               const int    vocab_size_padded,
                               const int    max_optional_last_tokens_count,
                               cudaStream_t stream);
#ifdef ENABLE_BF16
template void
invokeSelectOptionalLastTokens(__nv_bfloat16* logits,                // [batch_size, beam_width, vocab_size_padded]
                               const int*     optional_last_tokens,  // [batch_size, max_optional_last_tokens_count]
                               const int      batch_size,
                               const int      beam_width,
                               const int      vocab_size_padded,
                               const int      max_optional_last_tokens_count,
                               cudaStream_t   stream);
#endif
template void
invokeSelectOptionalLastTokens(float*       logits,                // [batch_size, beam_width, vocab_size_padded]
                               const int*   optional_last_tokens,  // [batch_size, max_optional_last_tokens_count]
                               const int    batch_size,
                               const int    beam_width,
                               const int    vocab_size_padded,
                               const int    max_optional_last_tokens_count,
                               cudaStream_t stream);

}  // namespace fastertransformer