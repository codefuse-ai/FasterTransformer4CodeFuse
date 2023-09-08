/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <vector>

// #include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/layers/FfnLayerINT8.h"
#include "src/fastertransformer/layers/attention_layers_int8/FusedAttentionLayerINT8.h"
#include "src/fastertransformer/layers/attention_layers_int8/UnfusedAttentionLayerINT8.h"
#include "src/fastertransformer/models/vit_int8/ViTINT8Weight.h"
#include "src/fastertransformer/utils/conv2d.h"

namespace fastertransformer {

template<typename T>
class ViTTransformerINT8: public BaseLayer {
private:
    size_t                 max_batch_size_ = 0;
    size_t                 img_size_       = 224;
    size_t                 chn_num_        = 3;
    size_t                 class_num_      = 1000;
    size_t                 patch_size_     = 16;  // preproc patch size
    size_t                 max_seq_len_;
    size_t                 request_seq_len_;
    size_t                 embed_dim_;   // patch conv out units, size_per_head = embed_dim / head_num
    size_t                 head_num_;    // mha head num
    size_t                 head_dim_;    // mha head size
    size_t                 inter_size_;  // FF internal size
    size_t                 num_layer_;
    size_t                 nopad_token_num_;
    bool                   with_cls_token_;
    int                    sm_;
    float                  q_scaling_;
    static constexpr float layernorm_eps_ = 1e-6f;
    AttentionType          attention_type_;
    int                    int8_mode_;
    cudnnHandle_t          cudnn_handle_;

    BaseAttentionLayer<T>* attention_layer_;
    FfnLayerINT8<T>*       ffn_layer_;

    bool is_allocate_buffer_ = false;

    void allocateBuffer();
    void freeBuffer();
    bool resetBatch(size_t batch_size);
    bool resetSeqLen(size_t seq_len);
    bool setSeqLenVec(size_t batch_size);
    void setDefaultMask(size_t batch_size);
    void setDefaultPaddingOffset(size_t batch_size);
    void patchEmbed(T*        output,
                    const T*  input,
                    const T*  kernel,
                    const T*  bias,
                    const T*  cls_embed,
                    const T*  pos_embed,
                    const int batch,
                    const int img_size,
                    const int patch_size,
                    const int seq_len,
                    const int in_chans,
                    const int embed_dim);
    void initialize();

    void allocateBuffer(size_t batch_size);

protected:
    // size_t* token_num_ = nullptr;
    T*      embed_buf_1_            = nullptr;
    T*      embed_buf_2_            = nullptr;
    T*      embed_buf_3_            = nullptr;
    T*      embed_buf_4_            = nullptr;
    T*      mask_buf_               = nullptr;
    int*    trt_mha_padding_offset_ = nullptr;
    int*    seq_len_vec_            = nullptr;
    int*    padding_offset_         = nullptr;
    size_t* h_pinned_token_num_ptr_ = nullptr;

public:
    ViTTransformerINT8(size_t           max_batch_size,
                       size_t           img_size,
                       size_t           chn_num,
                       size_t           patch_size,
                       size_t           embed_dim,
                       size_t           head_num,
                       size_t           inter_size,
                       size_t           num_layer,
                       bool             with_cls_token,
                       int              sm,
                       float            q_scaling,
                       int              int8_mode,
                       cudaStream_t     stream,
                       cudnnHandle_t    cudnn_handle,
                       cublasMMWrapper* cublas_wrapper,
                       IAllocator*      allocator,
                       bool             is_free_buffer_after_forward,
                       AttentionType    attention_type);

    ViTTransformerINT8(ViTTransformerINT8<T> const& vit_layer);

    ~ViTTransformerINT8();

    void forward(std::vector<Tensor>*       output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const ViTINT8Weight<T>*    weights);
};

}  // namespace fastertransformer
