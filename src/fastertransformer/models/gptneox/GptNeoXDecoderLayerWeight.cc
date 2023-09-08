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

#include "src/fastertransformer/models/gptneox/GptNeoXDecoderLayerWeight.h"
#include "src/fastertransformer/kernels/transpose_int8_kernels.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
GptNeoXDecoderLayerWeight<T>::GptNeoXDecoderLayerWeight(const int  hidden_units,
                                                        const int  inter_size,
                                                        const int  tensor_para_size,
                                                        const int  tensor_para_rank,
                                                        const bool use_gptj_residual,
                                                        const int  int8_mode):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    use_gptj_residual_(use_gptj_residual),
    int8_mode_(int8_mode)
{
    mallocWeights();
    setWeightPtr();

    FT_CHECK_WITH_INFO(!(std::is_same<T, float>::value && int8_mode_ == 1),
                       "Weight only quant does not work with FP32 compute.");
}

template<typename T>
GptNeoXDecoderLayerWeight<T>::GptNeoXDecoderLayerWeight(const int int8_mode): int8_mode_(int8_mode)
{
}

template<typename T>
GptNeoXDecoderLayerWeight<T>::~GptNeoXDecoderLayerWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < 12; i++) {
            if (!use_gptj_residual_ && i != attention_dense_bias_weight_id) {
                cudaFree(weights_ptr[i]);
            }
        }

        pre_layernorm_weights.beta                            = nullptr;
        pre_layernorm_weights.gamma                           = nullptr;
        self_attention_weights.query_weight.kernel            = nullptr;
        self_attention_weights.query_weight.bias              = nullptr;
        self_attention_weights.attention_output_weight.kernel = nullptr;
        self_attention_weights.attention_output_weight.bias   = nullptr;
        post_attention_layernorm_weights.beta                 = nullptr;
        post_attention_layernorm_weights.gamma                = nullptr;

        ffn_weights.intermediate_weight.kernel = nullptr;
        ffn_weights.intermediate_weight.bias   = nullptr;
        ffn_weights.output_weight.kernel       = nullptr;
        ffn_weights.output_weight.bias         = nullptr;

        if (int8_mode_ != 0) {
            for (int i = 0; i < int8_weights_ptr.size(); i++) {
                if (int8_weights_ptr[i] != nullptr) {
                    deviceFree(int8_weights_ptr[i]);
                }
            }

            if (int8_mode_ == 1) {
                for (int i = 0; i < scale_ptr.size(); i++) {
                    if (weight_only_scale_ptr[i] != nullptr) {
                        deviceFree(weight_only_scale_ptr[i]);
                    }
                }
            }
            self_attention_weights.query_weight.int8_kernel                             = nullptr;
            self_attention_weights.query_weight.weight_only_quant_scale                 = nullptr;
            self_attention_weights.attention_output_weight.int8_kernel                  = nullptr;
            self_attention_weights.attention_output_weight.weight_only_quant_scale      = nullptr;
            ffn_weights.intermediate_weight.int8_kernel                                 = nullptr;
            ffn_weights.intermediate_weight.weight_only_quant_scale                     = nullptr;
            ffn_weights.output_weight.int8_kernel                                       = nullptr;
            ffn_weights.output_weight.weight_only_quant_scale                           = nullptr;
        }

        is_maintain_buffer                     = false;
    }
}

template<typename T>
void GptNeoXDecoderLayerWeight<T>::copyFrom(const GptNeoXDecoderLayerWeight& other)
{
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
    if (!use_gptj_residual_) {
        cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    }
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], hidden_units_);
    cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], hidden_units_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);

    if (int8_mode_ == 0) {
        cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], inter_size_ / tensor_para_size_ * hidden_units_);
    }
    else {
        cudaD2Dcpy(int8_weights_ptr[0], other.int8_weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[1], other.int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(int8_weights_ptr[2], other.int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[3], other.int8_weights_ptr[3], inter_size_ / tensor_para_size_ * hidden_units_);

        if (int8_mode_ == 1) {
            cudaD2Dcpy(weight_only_scale_ptr[0], other.weight_only_scale_ptr[0], 3 * hidden_units_ / tensor_para_size_);
            cudaD2Dcpy(weight_only_scale_ptr[1], other.weight_only_scale_ptr[1], hidden_units_);
            cudaD2Dcpy(weight_only_scale_ptr[2], other.weight_only_scale_ptr[2], inter_size_ / tensor_para_size_);
            cudaD2Dcpy(weight_only_scale_ptr[3], other.weight_only_scale_ptr[3], hidden_units_);
        }
    }

    // TODO zgh: modify it to support int8
}

template<typename T>
GptNeoXDecoderLayerWeight<T>::GptNeoXDecoderLayerWeight(const GptNeoXDecoderLayerWeight& other):
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    use_gptj_residual_(other.use_gptj_residual_),
    int8_mode_(other.int8_mode_)
{
    mallocWeights();
    copyFrom(other);
    setWeightPtr();
}

template<typename T>
GptNeoXDecoderLayerWeight<T>& GptNeoXDecoderLayerWeight<T>::operator=(const GptNeoXDecoderLayerWeight& other)
{
    hidden_units_      = other.hidden_units_;
    inter_size_        = other.inter_size_;
    tensor_para_size_  = other.tensor_para_size_;
    tensor_para_rank_  = other.tensor_para_rank_;
    use_gptj_residual_ = other.use_gptj_residual_;
    int8_mode_ = other.int8_mode_;

    mallocWeights();
    copyFrom(other);
    setWeightPtr();
    return *this;
}

template<typename T>
void GptNeoXDecoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_CHECK(is_maintain_buffer == true);
    const std::string rank_spec = std::to_string(tensor_para_rank_);

    loadWeightFromBin<T>(
        weights_ptr[0], {(size_t)hidden_units_}, dir_path + ".input_layernorm.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[1], {(size_t)hidden_units_}, dir_path + ".input_layernorm.weight.bin", model_file_type);

    loadWeightFromBin<T>(weights_ptr[3],
                         {(size_t)(3 * hidden_units_ / tensor_para_size_)},
                         dir_path + ".attention.query_key_value.bias." + rank_spec + ".bin",
                         model_file_type);

    if (!use_gptj_residual_) {
        loadWeightFromBin<T>(
            weights_ptr[5], {(size_t)hidden_units_}, dir_path + ".attention.dense.bias.bin", model_file_type);
    }

    loadWeightFromBin<T>(weights_ptr[7],
                         {(size_t)(inter_size_ / tensor_para_size_)},
                         dir_path + ".mlp.dense_h_to_4h.bias." + rank_spec + ".bin",
                         model_file_type);
    if (use_gptj_residual_) {
        loadWeightFromBin<T>(
            weights_ptr[9], {(size_t)hidden_units_}, dir_path + ".mlp.attention.bias.sum.bin", model_file_type);
    }
    else {
        loadWeightFromBin<T>(
            weights_ptr[9], {(size_t)hidden_units_}, dir_path + ".mlp.dense_4h_to_h.bias.bin", model_file_type);
    }
    loadWeightFromBin<T>(
        weights_ptr[10], {(size_t)hidden_units_}, dir_path + ".post_attention_layernorm.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[11], {(size_t)hidden_units_}, dir_path + ".post_attention_layernorm.weight.bin", model_file_type);
    
    if (int8_mode_ == 0) {
        loadWeightFromBin<T>(weights_ptr[2],
                            {(size_t)hidden_units_, (size_t)(3 * hidden_units_ / tensor_para_size_)},
                            dir_path + ".attention.query_key_value.weight." + rank_spec + ".bin",
                            model_file_type);
        loadWeightFromBin<T>(weights_ptr[4],
                            {(size_t)(hidden_units_ / tensor_para_size_), (size_t)hidden_units_},
                            dir_path + ".attention.dense.weight." + rank_spec + ".bin",
                            model_file_type);
        loadWeightFromBin<T>(weights_ptr[6],
                                {(size_t)hidden_units_, (size_t)(inter_size_ / tensor_para_size_)},
                                dir_path + ".mlp.dense_h_to_4h.weight." + rank_spec + ".bin",
                                model_file_type);
        loadWeightFromBin<T>(weights_ptr[8],
                            {(size_t)(inter_size_ / tensor_para_size_), (size_t)hidden_units_},
                            dir_path + ".mlp.dense_4h_to_h.weight." + rank_spec + ".bin",
                            model_file_type);
    }
    else if (int8_mode_ == 1) {
        loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[0],
                                                     weight_only_scale_ptr[0],
                                                     {hidden_units_, 3 * hidden_units_ / tensor_para_size_},
                                                     dir_path + ".attention.query_key_value.weight."
                                                         + std::to_string(tensor_para_rank_) + ".bin",
                                                     model_file_type);

        loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[1],
                                                     weight_only_scale_ptr[1],
                                                     {hidden_units_ / tensor_para_size_, hidden_units_},
                                                     dir_path + ".attention.dense.weight."
                                                         + std::to_string(tensor_para_rank_) + ".bin",
                                                     model_file_type);

        loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[2],
                                                     weight_only_scale_ptr[2],
                                                     {hidden_units_, inter_size_ / tensor_para_size_},
                                                     dir_path + ".mlp.dense_h_to_4h.weight."
                                                         + std::to_string(tensor_para_rank_) + ".bin",
                                                     model_file_type);

        loadWeightFromBinAndQuantizeForWeightOnly<T>(int8_weights_ptr[3],
                                                     weight_only_scale_ptr[3],
                                                     {inter_size_ / tensor_para_size_, hidden_units_},
                                                     dir_path + ".mlp.dense_4h_to_h.weight."
                                                         + std::to_string(tensor_para_rank_) + ".bin",
                                                     model_file_type);
    }
}

template<typename T>
void GptNeoXDecoderLayerWeight<T>::setWeightPtr()
{
    pre_layernorm_weights.beta                            = weights_ptr[0];
    pre_layernorm_weights.gamma                           = weights_ptr[1];
    self_attention_weights.query_weight.kernel            = weights_ptr[2];
    self_attention_weights.query_weight.bias              = weights_ptr[3];
    self_attention_weights.attention_output_weight.kernel = weights_ptr[4];
    self_attention_weights.attention_output_weight.bias   = use_gptj_residual_ ? nullptr : weights_ptr[5];

    ffn_weights.intermediate_weight.kernel = weights_ptr[6];
    ffn_weights.intermediate_weight.bias   = weights_ptr[7];
    ffn_weights.output_weight.kernel       = weights_ptr[8];
    ffn_weights.output_weight.bias         = weights_ptr[9];

    post_attention_layernorm_weights.beta  = weights_ptr[10];
    post_attention_layernorm_weights.gamma = weights_ptr[11];

    if (int8_mode_ != 0) {
        self_attention_weights.query_weight.int8_kernel                 = int8_weights_ptr[0];
        self_attention_weights.attention_output_weight.int8_kernel      = int8_weights_ptr[1];
        ffn_weights.intermediate_weight.int8_kernel                     = int8_weights_ptr[2];
        ffn_weights.output_weight.int8_kernel                           = int8_weights_ptr[3];

        if (int8_mode_ == 1) {
            self_attention_weights.query_weight.weight_only_quant_scale                 = weight_only_scale_ptr[0];
            self_attention_weights.attention_output_weight.weight_only_quant_scale      = weight_only_scale_ptr[1];
            ffn_weights.intermediate_weight.weight_only_quant_scale                     = weight_only_scale_ptr[2];
            ffn_weights.output_weight.weight_only_quant_scale                           = weight_only_scale_ptr[3];
        }
    }

    is_maintain_buffer                     = true;
}

template<typename T>
void GptNeoXDecoderLayerWeight<T>::mallocWeights()
{
    deviceMalloc(&weights_ptr[0], hidden_units_);
    deviceMalloc(&weights_ptr[1], hidden_units_);
    deviceMalloc(&weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
    if (!use_gptj_residual_) {
        deviceMalloc(&weights_ptr[5], hidden_units_);
    }

    deviceMalloc(&weights_ptr[7], inter_size_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[9], hidden_units_);
    deviceMalloc(&weights_ptr[10], hidden_units_);
    deviceMalloc(&weights_ptr[11], hidden_units_);

    if (int8_mode_ == 0) {
        deviceMalloc(&weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        deviceMalloc(&weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
        deviceMalloc(&weights_ptr[6], hidden_units_ * inter_size_ / tensor_para_size_);
        deviceMalloc(&weights_ptr[8], inter_size_ / tensor_para_size_ * hidden_units_);
    }
    else {
        // Alloc FFN and Attention int8 weights
        deviceMalloc(&int8_weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        deviceMalloc(&int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_);
        deviceMalloc(&int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_);
        deviceMalloc(&int8_weights_ptr[3], inter_size_ / tensor_para_size_ * hidden_units_);

        if (int8_mode_ == 1) {
            // Alloc scales for weight only quant for attention and FFN weights
            deviceMalloc(&weight_only_scale_ptr[0], 3 * hidden_units_ / tensor_para_size_);
            deviceMalloc(&weight_only_scale_ptr[1], hidden_units_);
            deviceMalloc(&weight_only_scale_ptr[2], inter_size_ / tensor_para_size_);
            deviceMalloc(&weight_only_scale_ptr[3], hidden_units_);
        }
    }
}

template struct GptNeoXDecoderLayerWeight<float>;
template struct GptNeoXDecoderLayerWeight<half>;

}  // namespace fastertransformer
