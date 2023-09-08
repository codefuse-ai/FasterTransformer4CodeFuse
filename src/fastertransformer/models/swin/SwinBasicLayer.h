/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "SwinBlock.h"
#include "src/fastertransformer/kernels/image_merge_kernels.h"
namespace fastertransformer {
template<typename T>
class SwinTransformerBasicLayer: public BaseLayer {

private:
    int   max_batch_   = 1;
    int   window_size_ = 7;
    float mlp_ratio_   = 4.0f;
    bool  qkv_bias_    = true;
    float qk_scale_    = 1.0f;
    float layernorm_eps_;
    int   version_ = 1;

    T*                       buf_          = nullptr;
    T *                      block_output_ = nullptr, *merge_layernorm_buf_ = nullptr;
    SwinTransformerBlock<T>* block_ = nullptr;

public:
    // dim & input_resolution will be used to malloc the max buf size
    SwinTransformerBasicLayer(int              max_batch,
                              int              window_size,
                              float            mlp_ratio,
                              float            layernorm_eps,
                              cudaStream_t     stream,
                              cublasMMWrapper* cublas_wrapper,
                              IAllocator*      allocator,
                              bool             is_free_buffer_after_forward,
                              bool             qkv_bias,
                              float            qk_scale,
                              int              version);

    void allocateBuffer() override;
    void allocateBuffer(int batch, int input_resolution, int dim);
    void freeBuffer();

    ~SwinTransformerBasicLayer();

    // input is [B, H, W, C]
    // merge_layernorm_buf is [B, H/2, W/2, 4*C]
    // output is [B, H/2, W/2, 2*C]
    void patchMerge(T*       output,
                    T*       merge_layernorm_buf,
                    const T* input,
                    const T* gamma,
                    const T* beta,
                    const T* weight,
                    int      batch,
                    int      input_resolution,
                    int      dim);

    void forward(TensorMap*                          output_tensors,
                 TensorMap*                          input_tensors,
                 SwinTransformerBasicLayerWeight<T>& swin_basic_layer_weights);

};  // class SwinTransformerBasicLayer
}  // namespace fastertransformer
