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

#include "T5AdapterWeight.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/utils/IA3.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include <string>

namespace fastertransformer {

template<typename T>
struct T5DecoderLayerWeight {

    T5DecoderLayerWeight() = default;
    T5DecoderLayerWeight(size_t head_num,
                         size_t size_per_head,
                         size_t d_model,
                         size_t inter_size,
                         size_t mem_d_model,
                         size_t tensor_para_size,
                         size_t tensor_para_rank,
                         bool   t5_with_bias,
                         bool   use_gated_activation,
                         size_t ia3_num_tasks,
                         size_t adapter_inter_size);
    ~T5DecoderLayerWeight();
    T5DecoderLayerWeight(const T5DecoderLayerWeight& other);
    T5DecoderLayerWeight& operator=(const T5DecoderLayerWeight& other);

    LayerNormWeight<T> pre_layernorm_weights;
    AttentionWeight<T> self_attention_weights;
    LayerNormWeight<T> self_attn_layernorm_weights;
    AttentionWeight<T> cross_attention_weights;
    LayerNormWeight<T> cross_attn_layernorm_weights;
    FfnWeight<T>       ffn_weights;
    bool               t5_with_bias_;
    bool               use_gated_activation_;

    T5AdapterWeight<T> adapter_weights_;

    void loadModel(std::string dir_path, FtCudaDataType model_file_type);

    void setT5WithBias(bool t5_with_bias_para, bool use_gated_activation_para);

    bool has_adapters() const
    {
        return adapter_weights_.enabled();
    }

private:
    void setWeightPtr();
    void mallocWeights();
    void initialize();

    size_t head_num_;
    size_t size_per_head_;
    size_t d_model_;
    size_t inter_size_;
    size_t mem_d_model_;
    size_t tensor_para_size_;
    size_t tensor_para_rank_;
    size_t ia3_num_tasks_;
    bool   is_maintain_buffer = false;

    int real_weights_num_;

    const static int weights_num_ = 24;
    T*               weights_ptr[weights_num_];
    size_t           weights_size[weights_num_];

    T*     ia3_weights_ptr_[IA3_ADAPTER_MAX_NUM_DECODER];
    size_t ia3_weights_size_[IA3_ADAPTER_MAX_NUM_DECODER];
    bool   maintain_ia3_buffer_ = false;
};

}  // namespace fastertransformer
