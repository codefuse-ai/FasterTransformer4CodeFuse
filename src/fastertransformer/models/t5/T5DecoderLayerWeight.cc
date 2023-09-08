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

#include "src/fastertransformer/models/t5/T5DecoderLayerWeight.h"
#include "src/fastertransformer/utils/IA3.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
T5DecoderLayerWeight<T>::T5DecoderLayerWeight(const size_t head_num,
                                              const size_t size_per_head,
                                              const size_t d_model,
                                              const size_t inter_size,
                                              const size_t mem_d_model,
                                              const size_t tensor_para_size,
                                              const size_t tensor_para_rank,
                                              const bool   t5_with_bias,
                                              const bool   use_gated_activation,
                                              const size_t ia3_num_tasks,
                                              const size_t adapter_inter_size):
    adapter_weights_{d_model, adapter_inter_size, tensor_para_size, tensor_para_rank},
    head_num_(head_num),
    size_per_head_(size_per_head),
    d_model_(d_model),
    inter_size_(inter_size),
    mem_d_model_(mem_d_model),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    t5_with_bias_(t5_with_bias),
    use_gated_activation_(use_gated_activation),
    ia3_num_tasks_(ia3_num_tasks)
{
    real_weights_num_ = (11 + (use_gated_activation ? 1 : 0)) * (t5_with_bias ? 2 : 1);

    FT_LOG_DEBUG("T5DecoderLayerWeight " + std::string(__func__) + " start");

    initialize();
    mallocWeights();
    setWeightPtr();

    FT_LOG_DEBUG("T5DecoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
void T5DecoderLayerWeight<T>::initialize()
{
    FT_LOG_DEBUG("T5DecoderLayerWeight " + std::string(__func__) + " start");

    weights_size[0] = d_model_;
    weights_size[1] = d_model_ * 3 * (head_num_ / tensor_para_size_) * size_per_head_;
    weights_size[2] = (head_num_ / tensor_para_size_) * size_per_head_ * d_model_;
    weights_size[3] = d_model_;
    weights_size[4] = d_model_ * (head_num_ / tensor_para_size_) * size_per_head_;
    weights_size[5] = mem_d_model_ * (head_num_ / tensor_para_size_) * size_per_head_;
    weights_size[6] = mem_d_model_ * (head_num_ / tensor_para_size_) * size_per_head_;
    weights_size[7] = (head_num_ / tensor_para_size_) * size_per_head_ * d_model_;
    weights_size[8] = d_model_;
    if (use_gated_activation_) {
        weights_size[9]  = d_model_ * (inter_size_ / tensor_para_size_);
        weights_size[10] = d_model_ * (inter_size_ / tensor_para_size_);  // for gated activation
        weights_size[11] = (inter_size_ / tensor_para_size_) * d_model_;
    }
    else {
        weights_size[9]  = d_model_ * (inter_size_ / tensor_para_size_);
        weights_size[10] = (inter_size_ / tensor_para_size_) * d_model_;
    }

    if (t5_with_bias_) {
        if (use_gated_activation_) {
            weights_size[12] = d_model_;
            weights_size[13] = 3 * (head_num_ / tensor_para_size_) * size_per_head_;
            weights_size[14] = d_model_;
            weights_size[15] = d_model_;
            weights_size[16] = (head_num_ / tensor_para_size_) * size_per_head_;
            weights_size[17] = (head_num_ / tensor_para_size_) * size_per_head_;
            weights_size[18] = (head_num_ / tensor_para_size_) * size_per_head_;
            weights_size[19] = d_model_;
            weights_size[20] = d_model_;
            weights_size[21] = (inter_size_ / tensor_para_size_);
            weights_size[22] = (inter_size_ / tensor_para_size_);  // for gated activation
            weights_size[23] = d_model_;
        }
        else {
            weights_size[11] = d_model_;
            weights_size[12] = 3 * (head_num_ / tensor_para_size_) * size_per_head_;
            weights_size[13] = d_model_;
            weights_size[14] = d_model_;
            weights_size[15] = (head_num_ / tensor_para_size_) * size_per_head_;
            weights_size[16] = (head_num_ / tensor_para_size_) * size_per_head_;
            weights_size[17] = (head_num_ / tensor_para_size_) * size_per_head_;
            weights_size[18] = d_model_;
            weights_size[19] = d_model_;
            weights_size[20] = (inter_size_ / tensor_para_size_);
            weights_size[21] = d_model_;
        }
    }

    if (ia3_num_tasks_ > 0) {
        const size_t attention_adapter_size = ia3_num_tasks_ * (head_num_ / tensor_para_size_) * size_per_head_;
        const size_t mlp_adapter_size       = ia3_num_tasks_ * (inter_size_ / tensor_para_size_);

        // Self-attention K/V
        ia3_weights_size_[0] = attention_adapter_size;
        ia3_weights_size_[1] = attention_adapter_size;
        // Cross-attention K/V
        ia3_weights_size_[2] = attention_adapter_size;
        ia3_weights_size_[3] = attention_adapter_size;
        // MLP
        ia3_weights_size_[4] = mlp_adapter_size;
    }

    FT_LOG_DEBUG("T5DecoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
T5DecoderLayerWeight<T>::~T5DecoderLayerWeight()
{
    FT_LOG_DEBUG("T5DecoderLayerWeight " + std::string(__func__) + " start");

    if (is_maintain_buffer == true) {
        for (int i = 0; i < weights_num_; i++) {
            deviceFree(weights_ptr[i]);
        }

        pre_layernorm_weights.gamma                           = nullptr;
        self_attention_weights.query_weight.kernel            = nullptr;
        self_attention_weights.attention_output_weight.kernel = nullptr;
        self_attn_layernorm_weights.gamma                     = nullptr;

        cross_attention_weights.query_weight.kernel            = nullptr;
        cross_attention_weights.key_weight.kernel              = nullptr;
        cross_attention_weights.value_weight.kernel            = nullptr;
        cross_attention_weights.attention_output_weight.kernel = nullptr;
        cross_attn_layernorm_weights.gamma                     = nullptr;

        ffn_weights.intermediate_weight.kernel  = nullptr;
        ffn_weights.intermediate_weight2.kernel = nullptr;
        ffn_weights.output_weight.kernel        = nullptr;

        pre_layernorm_weights.beta                          = nullptr;
        self_attention_weights.query_weight.bias            = nullptr;
        self_attention_weights.attention_output_weight.bias = nullptr;
        self_attn_layernorm_weights.beta                    = nullptr;

        cross_attention_weights.query_weight.bias            = nullptr;
        cross_attention_weights.key_weight.bias              = nullptr;
        cross_attention_weights.value_weight.bias            = nullptr;
        cross_attention_weights.attention_output_weight.bias = nullptr;
        cross_attn_layernorm_weights.beta                    = nullptr;

        ffn_weights.intermediate_weight.bias  = nullptr;
        ffn_weights.intermediate_weight2.bias = nullptr;
        ffn_weights.output_weight.bias        = nullptr;
        is_maintain_buffer                    = false;
    }

    if (maintain_ia3_buffer_) {
        for (int i = 0; i < IA3_ADAPTER_MAX_NUM_DECODER; i++) {
            deviceFree(ia3_weights_ptr_[i]);
        }
        self_attention_weights.ia3_key_weight.kernel    = nullptr;
        self_attention_weights.ia3_value_weight.kernel  = nullptr;
        cross_attention_weights.ia3_key_weight.kernel   = nullptr;
        cross_attention_weights.ia3_value_weight.kernel = nullptr;
        ffn_weights.ia3_weight.kernel                   = nullptr;
    }

    FT_LOG_DEBUG("T5DecoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
T5DecoderLayerWeight<T>::T5DecoderLayerWeight(const T5DecoderLayerWeight& other):
    adapter_weights_{other.adapter_weights_},
    head_num_(other.head_num_),
    size_per_head_(other.size_per_head_),
    d_model_(other.d_model_),
    inter_size_(other.inter_size_),
    mem_d_model_(other.mem_d_model_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    t5_with_bias_(other.t5_with_bias_),
    use_gated_activation_(other.use_gated_activation_),
    real_weights_num_(other.real_weights_num_),
    ia3_num_tasks_(other.ia3_num_tasks_)
{

    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    if (ia3_num_tasks_ > 0) {
        for (int i = 0; i < IA3_ADAPTER_MAX_NUM_DECODER; i++) {
            cudaD2Dcpy(ia3_weights_ptr_[i], other.ia3_weights_ptr_[i], ia3_weights_size_[i]);
        }
    }
    setWeightPtr();
}

template<typename T>
T5DecoderLayerWeight<T>& T5DecoderLayerWeight<T>::operator=(const T5DecoderLayerWeight& other)
{
    if (this == &other)
        return *this;

    adapter_weights_      = other.adapter_weights_;
    head_num_             = other.head_num_;
    size_per_head_        = other.size_per_head_;
    d_model_              = other.d_model_;
    inter_size_           = other.inter_size_;
    mem_d_model_          = other.mem_d_model_;
    tensor_para_size_     = other.tensor_para_size_;
    tensor_para_rank_     = other.tensor_para_rank_;
    t5_with_bias_         = other.t5_with_bias_;
    use_gated_activation_ = other.use_gated_activation_;
    real_weights_num_     = other.real_weights_num_;
    ia3_num_tasks_        = other.ia3_num_tasks_;

    initialize();
    mallocWeights();
    for (int i = 0; i < real_weights_num_; i++) {
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }
    if (ia3_num_tasks_ > 0) {
        for (int i = 0; i < IA3_ADAPTER_MAX_NUM_DECODER; i++) {
            cudaD2Dcpy(ia3_weights_ptr_[i], other.ia3_weights_ptr_[i], ia3_weights_size_[i]);
        }
    }
    setWeightPtr();

    return *this;
}

template<typename T>
void T5DecoderLayerWeight<T>::setWeightPtr()
{
    pre_layernorm_weights.gamma                           = weights_ptr[0];
    self_attention_weights.query_weight.kernel            = weights_ptr[1];
    self_attention_weights.attention_output_weight.kernel = weights_ptr[2];
    self_attn_layernorm_weights.gamma                     = weights_ptr[3];

    cross_attention_weights.query_weight.kernel            = weights_ptr[4];
    cross_attention_weights.key_weight.kernel              = weights_ptr[5];
    cross_attention_weights.value_weight.kernel            = weights_ptr[6];
    cross_attention_weights.attention_output_weight.kernel = weights_ptr[7];
    cross_attn_layernorm_weights.gamma                     = weights_ptr[8];

    if (use_gated_activation_) {
        ffn_weights.intermediate_weight.kernel  = weights_ptr[9];
        ffn_weights.intermediate_weight2.kernel = weights_ptr[10];
        ffn_weights.output_weight.kernel        = weights_ptr[11];
    }
    else {
        ffn_weights.intermediate_weight.kernel = weights_ptr[9];
        ffn_weights.output_weight.kernel       = weights_ptr[10];
    }
    if (t5_with_bias_) {
        if (use_gated_activation_) {
            pre_layernorm_weights.beta                          = weights_ptr[12];
            self_attention_weights.query_weight.bias            = weights_ptr[13];
            self_attention_weights.attention_output_weight.bias = weights_ptr[14];
            self_attn_layernorm_weights.beta                    = weights_ptr[15];

            cross_attention_weights.query_weight.bias            = weights_ptr[16];
            cross_attention_weights.key_weight.bias              = weights_ptr[17];
            cross_attention_weights.value_weight.bias            = weights_ptr[18];
            cross_attention_weights.attention_output_weight.bias = weights_ptr[19];
            cross_attn_layernorm_weights.beta                    = weights_ptr[20];

            ffn_weights.intermediate_weight.bias  = weights_ptr[21];
            ffn_weights.intermediate_weight2.bias = weights_ptr[22];
            ffn_weights.output_weight.bias        = weights_ptr[23];
        }
        else {
            pre_layernorm_weights.beta                          = weights_ptr[11];
            self_attention_weights.query_weight.bias            = weights_ptr[12];
            self_attention_weights.attention_output_weight.bias = weights_ptr[13];
            self_attn_layernorm_weights.beta                    = weights_ptr[14];

            cross_attention_weights.query_weight.bias            = weights_ptr[15];
            cross_attention_weights.key_weight.bias              = weights_ptr[16];
            cross_attention_weights.value_weight.bias            = weights_ptr[17];
            cross_attention_weights.attention_output_weight.bias = weights_ptr[18];
            cross_attn_layernorm_weights.beta                    = weights_ptr[19];

            ffn_weights.intermediate_weight.bias = weights_ptr[20];
            ffn_weights.output_weight.bias       = weights_ptr[21];
        }
    }

    if (ia3_num_tasks_ > 0) {
        self_attention_weights.ia3_key_weight.kernel    = ia3_weights_ptr_[0];
        self_attention_weights.ia3_value_weight.kernel  = ia3_weights_ptr_[1];
        cross_attention_weights.ia3_key_weight.kernel   = ia3_weights_ptr_[2];
        cross_attention_weights.ia3_value_weight.kernel = ia3_weights_ptr_[3];
        ffn_weights.ia3_weight.kernel                   = ia3_weights_ptr_[4];
    }
}

template<typename T>
void T5DecoderLayerWeight<T>::mallocWeights()
{
    for (int i = 0; i < real_weights_num_; i++) {
        deviceMalloc(&weights_ptr[i], weights_size[i]);
    }
    if (ia3_num_tasks_ > 0) {
        for (int i = 0; i < IA3_ADAPTER_MAX_NUM_DECODER; i++) {
            deviceMalloc(&ia3_weights_ptr_[i], ia3_weights_size_[i]);
        }
    }
    is_maintain_buffer = true;
}

template<typename T>
void T5DecoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_LOG_DEBUG("T5DecoderLayerWeight " + std::string(__func__) + " start");

    FT_CHECK(is_maintain_buffer == true);
    const auto tp_rank = std::to_string(tensor_para_rank_);

    loadWeightFromBin<T>(
        weights_ptr[0], {weights_size[0]}, dir_path + "layer.0.layer_norm.weight.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[1],
                         {weights_size[1]},
                         dir_path + "layer.0.SelfAttention.qkv.weight." + tp_rank + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[2],
                         {weights_size[2]},
                         dir_path + "layer.0.SelfAttention.o.weight." + tp_rank + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[3], {weights_size[3]}, dir_path + "layer.1.layer_norm.weight.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[4],
                         {weights_size[4]},
                         dir_path + "layer.1.EncDecAttention.q.weight." + tp_rank + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[5],
                         {weights_size[5]},
                         dir_path + "layer.1.EncDecAttention.k.weight." + tp_rank + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[6],
                         {weights_size[6]},
                         dir_path + "layer.1.EncDecAttention.v.weight." + tp_rank + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[7],
                         {weights_size[7]},
                         dir_path + "layer.1.EncDecAttention.o.weight." + tp_rank + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[8], {weights_size[8]}, dir_path + "layer.2.layer_norm.weight.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[9],
                         {weights_size[9]},
                         dir_path + "layer.2.DenseReluDense.wi.weight." + tp_rank + ".bin",
                         model_file_type);

    const int gated_activation_weight_offset = use_gated_activation_ ? 1 : 0;
    if (use_gated_activation_) {
        loadWeightFromBin<T>(weights_ptr[10],
                             {weights_size[10]},
                             dir_path + "layer.2.DenseReluDense.wi2.weight." + tp_rank + ".bin",
                             model_file_type);
    }
    loadWeightFromBin<T>(weights_ptr[10 + gated_activation_weight_offset],
                         {weights_size[10 + gated_activation_weight_offset]},
                         dir_path + "layer.2.DenseReluDense.wo.weight." + tp_rank + ".bin",
                         model_file_type);

    if (t5_with_bias_) {
        loadWeightFromBin<T>(weights_ptr[11 + gated_activation_weight_offset],
                             {weights_size[11 + gated_activation_weight_offset]},
                             dir_path + "layer.0.layer_norm.bias.bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[12 + gated_activation_weight_offset],
                             {weights_size[12 + gated_activation_weight_offset]},
                             dir_path + "layer.0.SelfAttention.qkv.bias." + tp_rank + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[13 + gated_activation_weight_offset],
                             {weights_size[13 + gated_activation_weight_offset]},
                             dir_path + "layer.0.SelfAttention.o.bias.bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[14 + gated_activation_weight_offset],
                             {weights_size[14 + gated_activation_weight_offset]},
                             dir_path + "layer.1.layer_norm.bias.bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[15 + gated_activation_weight_offset],
                             {weights_size[15 + gated_activation_weight_offset]},
                             dir_path + "layer.1.EncDecAttention.q.bias." + tp_rank + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[16 + gated_activation_weight_offset],
                             {weights_size[16 + gated_activation_weight_offset]},
                             dir_path + "layer.1.EncDecAttention.k.bias." + tp_rank + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[17 + gated_activation_weight_offset],
                             {weights_size[17 + gated_activation_weight_offset]},
                             dir_path + "layer.1.EncDecAttention.v.bias." + tp_rank + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[18 + gated_activation_weight_offset],
                             {weights_size[18 + gated_activation_weight_offset]},
                             dir_path + "layer.1.EncDecAttention.o.bias.bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[19 + gated_activation_weight_offset],
                             {weights_size[19 + gated_activation_weight_offset]},
                             dir_path + "layer.2.layer_norm.bias.bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[20 + gated_activation_weight_offset],
                             {weights_size[20 + gated_activation_weight_offset]},
                             dir_path + "layer.2.DenseReluDense.wi.bias." + tp_rank + ".bin",
                             model_file_type);
        if (use_gated_activation_) {
            loadWeightFromBin<T>(weights_ptr[22],
                                 {weights_size[22]},
                                 dir_path + "layer.2.DenseReluDense.wi2.bias." + tp_rank + ".bin",
                                 model_file_type);
            loadWeightFromBin<T>(
                weights_ptr[23], {weights_size[23]}, dir_path + "layer.2.DenseReluDense.wo.bias.bin", model_file_type);
        }
        else {
            loadWeightFromBin<T>(
                weights_ptr[21], {weights_size[21]}, dir_path + "layer.2.DenseReluDense.wo.bias.bin", model_file_type);
        }
    }

    if (ia3_num_tasks_ > 0) {
        loadWeightFromBin<T>(ia3_weights_ptr_[0],
                             {ia3_weights_size_[0]},
                             dir_path + "layer.0.SelfAttention.k.ia3.weight." + tp_rank + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(ia3_weights_ptr_[1],
                             {ia3_weights_size_[1]},
                             dir_path + "layer.0.SelfAttention.v.ia3.weight." + tp_rank + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(ia3_weights_ptr_[2],
                             {ia3_weights_size_[2]},
                             dir_path + "layer.1.EncDecAttention.k.ia3.weight." + tp_rank + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(ia3_weights_ptr_[3],
                             {ia3_weights_size_[3]},
                             dir_path + "layer.1.EncDecAttention.v.ia3.weight." + tp_rank + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(ia3_weights_ptr_[4],
                             {ia3_weights_size_[4]},
                             dir_path + "layer.2.DenseReluDense.ia3.weight." + tp_rank + ".bin",
                             model_file_type);
    }

    adapter_weights_.loadModel(dir_path, model_file_type);

    FT_LOG_DEBUG("T5DecoderLayerWeight " + std::string(__func__) + " end");
}

template<typename T>
void T5DecoderLayerWeight<T>::setT5WithBias(bool t5_with_bias_para, bool use_gated_activation_para)
{
    t5_with_bias_         = t5_with_bias_para;
    use_gated_activation_ = use_gated_activation_para;
}

template struct T5DecoderLayerWeight<float>;
template struct T5DecoderLayerWeight<half>;
#ifdef ENABLE_BF16
template struct T5DecoderLayerWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
