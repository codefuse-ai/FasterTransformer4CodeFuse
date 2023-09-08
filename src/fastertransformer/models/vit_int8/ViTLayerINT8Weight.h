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

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnINT8Weight.h"
#include "src/fastertransformer/layers/attention_layers_int8/AttentionINT8Weight.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

#define WEIGHT_N 16

template<typename T>
struct ViTLayerINT8Weight {

    ViTLayerINT8Weight() = default;
    ViTLayerINT8Weight(const int embed_dim, const int inter_size, int layer_idx, const bool hold_buffer):
        embed_dim_(embed_dim), inter_size_(inter_size), layer_idx_(layer_idx)
    {
        weights_size[0]  = embed_dim_ * embed_dim_;
        weights_size[1]  = embed_dim_;
        weights_size[2]  = embed_dim_ * embed_dim_;
        weights_size[3]  = embed_dim_;
        weights_size[4]  = embed_dim_ * embed_dim_;
        weights_size[5]  = embed_dim_;
        weights_size[6]  = embed_dim_ * embed_dim_;
        weights_size[7]  = embed_dim_;
        weights_size[8]  = embed_dim_;
        weights_size[9]  = embed_dim_;
        weights_size[10] = embed_dim_ * inter_size_;
        weights_size[11] = inter_size_;
        weights_size[12] = inter_size_ * embed_dim_;
        weights_size[13] = embed_dim_;
        weights_size[14] = embed_dim_;
        weights_size[15] = embed_dim_;
        if (hold_buffer) {
            for (int i = 0; i < WEIGHT_N; i++) {
                deviceMalloc(&weights_ptr[i], weights_size[i]);
            }

            scale_list_.size_ = ACTIVATION_AMAX_NUM + 9 * embed_dim + INT8O_GEMM_NUM + TRT_AMAX_NUM + SCALE_RESERVE_NUM;
            scale_list_.p3_offset_ = ACTIVATION_AMAX_NUM + 9 * embed_dim;
            scale_list_.p4_offset_ = ACTIVATION_AMAX_NUM + 9 * embed_dim + INT8O_GEMM_NUM;
            deviceMalloc(&scale_list_ptr[0], scale_list_.size_);
            scale_list_ptr[1] = (float*)malloc(sizeof(float) * scale_list_.size_);
            setWeightPtr();
        }
    }

    ~ViTLayerINT8Weight()
    {
        if (is_maintain_buffer == true) {
            for (int i = 0; i < WEIGHT_N; i++) {
                deviceFree(weights_ptr[i]);
            }

            deviceFree(scale_list_ptr[0]);
            free(scale_list_ptr[1]);
            attention_weights.query_weight.kernel            = nullptr;
            attention_weights.query_weight.bias              = nullptr;
            attention_weights.key_weight.kernel              = nullptr;
            attention_weights.key_weight.bias                = nullptr;
            attention_weights.value_weight.kernel            = nullptr;
            attention_weights.value_weight.bias              = nullptr;
            attention_weights.attention_output_weight.kernel = nullptr;
            attention_weights.attention_output_weight.bias   = nullptr;
            attn_layernorm_weights.gamma                     = nullptr;
            attn_layernorm_weights.beta                      = nullptr;
            ffn_weights.intermediate_weight.kernel           = nullptr;
            ffn_weights.intermediate_weight.bias             = nullptr;
            ffn_weights.output_weight.kernel                 = nullptr;
            ffn_weights.output_weight.bias                   = nullptr;
            ffn_layernorm_weights.gamma                      = nullptr;
            ffn_layernorm_weights.beta                       = nullptr;
            is_maintain_buffer                               = false;
        }
    }

    ViTLayerINT8Weight(const ViTLayerINT8Weight& other): embed_dim_(other.embed_dim_), inter_size_(other.inter_size_)
    {
        memcpy(weights_size, other.weights_size, sizeof(size_t) * WEIGHT_N);
        layer_idx_ = other.layer_idx_;
        if (other.is_maintain_buffer) {
            for (int i = 0; i < WEIGHT_N; i++) {
                if (!is_maintain_buffer) {
                    deviceMalloc(&weights_ptr[i], weights_size[i]);
                }
                cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
            }

            scale_list_.size_      = other.scale_list_.size_;
            scale_list_.p3_offset_ = other.scale_list_.p3_offset_;
            scale_list_.p4_offset_ = other.scale_list_.p4_offset_;
            deviceMalloc(&scale_list_ptr[0], scale_list_.size_);
            cudaD2Dcpy(scale_list_ptr[0], other.scale_list_ptr[0], scale_list_.size_);
            scale_list_ptr[1] = (float*)malloc(sizeof(float) * scale_list_.size_);
            memcpy(scale_list_ptr[1], other.scale_list_ptr[1], sizeof(float) * scale_list_.size_);
            setWeightPtr();
        }
    }

    ViTLayerINT8Weight& operator=(const ViTLayerINT8Weight& other)
    {
        embed_dim_  = other.embed_dim_;
        inter_size_ = other.inter_size_;
        layer_idx_  = other.layer_idx_;
        memcpy(weights_size, other.weights_size, sizeof(size_t) * WEIGHT_N);
        if (other.is_maintain_buffer) {
            for (int i = 0; i < WEIGHT_N; i++) {
                if (!is_maintain_buffer) {
                    deviceMalloc(&weights_ptr[i], weights_size[i]);
                }
                cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
            }
            scale_list_.size_      = other.scale_list_.size_;
            scale_list_.p3_offset_ = other.scale_list_.p3_offset_;
            scale_list_.p4_offset_ = other.scale_list_.p4_offset_;
            deviceMalloc(&scale_list_ptr[0], scale_list_.size_);
            cudaD2Dcpy(scale_list_ptr[0], other.scale_list_ptr[0], scale_list_.size_);
            scale_list_ptr[1] = (float*)malloc(sizeof(float) * scale_list_.size_);
            memcpy(scale_list_ptr[1], other.scale_list_ptr[1], sizeof(float) * scale_list_.size_);
            setWeightPtr();
        }

        return *this;
    }

    inline size_t GetWeightCount()
    {
        return WEIGHT_N + 2;
    }

    size_t GetSerializeSize()
    {
        size_t count;
        for (int i = 0; i < WEIGHT_N; i++) {
            count += weights_size[i];
        }

        return sizeof(T) * count + 2 * scale_list_.size_ * sizeof(float);
    }

    void CopyWeightsFromHostBuffers(const T* const*& w)
    {
        cudaMemcpy(
            const_cast<T*>(attn_layernorm_weights.gamma), *w++, sizeof(T) * weights_size[8], cudaMemcpyHostToDevice);
        cudaMemcpy(
            const_cast<T*>(attn_layernorm_weights.beta), *w++, sizeof(T) * weights_size[9], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(attention_weights.query_weight.kernel),
                   *w++,
                   sizeof(T) * weights_size[0],
                   cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(attention_weights.query_weight.bias),
                   *w++,
                   sizeof(T) * weights_size[1],
                   cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(attention_weights.key_weight.kernel),
                   *w++,
                   sizeof(T) * weights_size[2],
                   cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(attention_weights.key_weight.bias),
                   *w++,
                   sizeof(T) * weights_size[3],
                   cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(attention_weights.value_weight.kernel),
                   *w++,
                   sizeof(T) * weights_size[4],
                   cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(attention_weights.value_weight.bias),
                   *w++,
                   sizeof(T) * weights_size[5],
                   cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(attention_weights.attention_output_weight.kernel),
                   *w++,
                   sizeof(T) * weights_size[6],
                   cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(attention_weights.attention_output_weight.bias),
                   *w++,
                   sizeof(T) * weights_size[7],
                   cudaMemcpyHostToDevice);
        cudaMemcpy(
            const_cast<T*>(ffn_layernorm_weights.gamma), *w++, sizeof(T) * weights_size[14], cudaMemcpyHostToDevice);
        cudaMemcpy(
            const_cast<T*>(ffn_layernorm_weights.beta), *w++, sizeof(T) * weights_size[15], cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(ffn_weights.intermediate_weight.kernel),
                   *w++,
                   sizeof(T) * weights_size[10],
                   cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(ffn_weights.intermediate_weight.bias),
                   *w++,
                   sizeof(T) * weights_size[11],
                   cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<T*>(ffn_weights.output_weight.kernel),
                   *w++,
                   sizeof(T) * weights_size[12],
                   cudaMemcpyHostToDevice);
        cudaMemcpy(
            const_cast<T*>(ffn_weights.output_weight.bias), *w++, sizeof(T) * weights_size[13], cudaMemcpyHostToDevice);

        cudaMemcpy(
            const_cast<float*>(scale_list_ptr[0]), *w++, sizeof(float) * scale_list_.size_, cudaMemcpyHostToDevice);
        cudaMemcpy(
            const_cast<float*>(scale_list_ptr[1]), *w++, sizeof(float) * scale_list_.size_, cudaMemcpyHostToHost);
    }

    void serialize(void* buffer)
    {
        char* tmp_buf = (char*)buffer;
        for (int i = 0; i < WEIGHT_N; i++) {
            cudaMemcpy(tmp_buf, weights_ptr[i], sizeof(T) * weights_size[i], cudaMemcpyDeviceToHost);
            tmp_buf += sizeof(T) * weights_size[i];
        }
        cudaMemcpy(tmp_buf, scale_list_ptr[0], sizeof(float) * scale_list_.size_, cudaMemcpyDeviceToHost);
        tmp_buf += sizeof(float) * scale_list_.size_;
        cudaMemcpy(tmp_buf, scale_list_ptr[1], sizeof(float) * scale_list_.size_, cudaMemcpyHostToHost);
        tmp_buf += sizeof(float) * scale_list_.size_;
    }

    void deserialize(const void* buffer)
    {
        if (!is_maintain_buffer) {
            return;
        }

        char* tmp_buf = (char*)buffer;
        for (int i = 0; i < WEIGHT_N; i++) {
            cudaMemcpy(weights_ptr[i], tmp_buf, sizeof(T) * weights_size[i], cudaMemcpyHostToDevice);
            tmp_buf += sizeof(T) * weights_size[i];
        }
        cudaMemcpy(scale_list_ptr[0], tmp_buf, sizeof(float) * scale_list_.size_, cudaMemcpyHostToDevice);
        tmp_buf += sizeof(float) * scale_list_.size_;
        cudaMemcpy(scale_list_ptr[1], tmp_buf, sizeof(float) * scale_list_.size_, cudaMemcpyHostToHost);
        tmp_buf += sizeof(float) * scale_list_.size_;
    }

    void ExportWeights(int layer_idx)
    {
        FT_LOG_INFO("Exporting layer %d...", layer_idx);
        FT_LOG_INFO("embed_dim:%d, inter_size:%d", embed_dim_, inter_size_);
        DataType dtype = DataType::TYPE_INVALID;
        if (std::is_same<T, half>::value) {
            dtype = DataType::TYPE_FP16;
        }
        else if (std::is_same<T, float>::value) {
            dtype = DataType::TYPE_FP32;
        }

        std::ostringstream buffer;
        buffer << "./weights/l" << layer_idx;
        DataType wtype = DataType::TYPE_INT8;
        Tensor   w1{
            MEMORY_GPU, wtype, std::vector<size_t>{embed_dim_, embed_dim_}, attention_weights.query_weight.kernel};
        w1.saveNpy(std::string(buffer.str()) + "_q_kern.npy");
        Tensor w2{MEMORY_GPU, dtype, std::vector<size_t>{embed_dim_}, attention_weights.query_weight.bias};
        w2.saveNpy(std::string(buffer.str()) + "_q_bias.npy");
        Tensor w3{MEMORY_GPU, wtype, std::vector<size_t>{embed_dim_, embed_dim_}, attention_weights.key_weight.kernel};
        w3.saveNpy(std::string(buffer.str()) + "_k_kern.npy");
        Tensor w4{MEMORY_GPU, dtype, std::vector<size_t>{embed_dim_}, attention_weights.key_weight.bias};
        w4.saveNpy(std::string(buffer.str()) + "_k_bias.npy");
        Tensor w5{
            MEMORY_GPU, wtype, std::vector<size_t>{embed_dim_, embed_dim_}, attention_weights.value_weight.kernel};
        w5.saveNpy(std::string(buffer.str()) + "_v_kern.npy");
        Tensor w6{MEMORY_GPU, dtype, std::vector<size_t>{embed_dim_}, attention_weights.value_weight.bias};
        w6.saveNpy(std::string(buffer.str()) + "_v_bias.npy");
        Tensor w7{MEMORY_GPU,
                  wtype,
                  std::vector<size_t>{embed_dim_, embed_dim_},
                  attention_weights.attention_output_weight.kernel};
        w7.saveNpy(std::string(buffer.str()) + "_att_o_kern.npy");
        Tensor w8{MEMORY_GPU, dtype, std::vector<size_t>{embed_dim_}, attention_weights.attention_output_weight.bias};
        w8.saveNpy(std::string(buffer.str()) + "_att_o_bias.npy");
        Tensor w9{MEMORY_GPU, dtype, std::vector<size_t>{embed_dim_}, attn_layernorm_weights.gamma};
        w9.saveNpy(std::string(buffer.str()) + "_ln0_scale.npy");
        Tensor w10{MEMORY_GPU, dtype, std::vector<size_t>{embed_dim_}, attn_layernorm_weights.beta};
        w10.saveNpy(std::string(buffer.str()) + "_ln0_bias.npy");
        Tensor w11{
            MEMORY_GPU, wtype, std::vector<size_t>{embed_dim_, inter_size_}, ffn_weights.intermediate_weight.kernel};
        w11.saveNpy(std::string(buffer.str()) + "_ffn_inter_kern.npy");
        Tensor w12{MEMORY_GPU, dtype, std::vector<size_t>{inter_size_}, ffn_weights.intermediate_weight.bias};
        w12.saveNpy(std::string(buffer.str()) + "_ffn_inter_bias.npy");
        Tensor w13{MEMORY_GPU, wtype, std::vector<size_t>{inter_size_, embed_dim_}, ffn_weights.output_weight.kernel};
        w13.saveNpy(std::string(buffer.str()) + "_ffn_o_kern.npy");
        Tensor w14{MEMORY_GPU, dtype, std::vector<size_t>{embed_dim_}, ffn_weights.output_weight.bias};
        w14.saveNpy(std::string(buffer.str()) + "_ffn_o_bias.npy");
        Tensor w15{MEMORY_GPU, dtype, std::vector<size_t>{embed_dim_}, ffn_layernorm_weights.gamma};
        w15.saveNpy(std::string(buffer.str()) + "_ln2_scale.npy");
        Tensor w16{MEMORY_GPU, dtype, std::vector<size_t>{embed_dim_}, ffn_layernorm_weights.beta};
        w16.saveNpy(std::string(buffer.str()) + "_ln2_bias.npy");

        Tensor w17{MEMORY_GPU, DataType::TYPE_FP32, std::vector<size_t>{scale_list_.size_}, scale_list_.d_scale_list_};
        w17.saveNpy(std::string(buffer.str()) + "_d_scalelist.npy");
        Tensor w18{MEMORY_CPU, DataType::TYPE_FP32, std::vector<size_t>{scale_list_.size_}, scale_list_.h_scale_list_};
        w18.saveNpy(std::string(buffer.str()) + "_h_scalelist.npy");
    }

    AttentionINT8Weight<T> attention_weights;
    LayerNormWeight<T>     attn_layernorm_weights;
    FfnINT8Weight<T>       ffn_weights;
    LayerNormWeight<T>     ffn_layernorm_weights;
    ScaleList              scale_list_;

private:
    void setWeightPtr()
    {
        attention_weights.query_weight.kernel            = weights_ptr[0];
        attention_weights.query_weight.bias              = weights_ptr[1];
        attention_weights.key_weight.kernel              = weights_ptr[2];
        attention_weights.key_weight.bias                = weights_ptr[3];
        attention_weights.value_weight.kernel            = weights_ptr[4];
        attention_weights.value_weight.bias              = weights_ptr[5];
        attention_weights.attention_output_weight.kernel = weights_ptr[6];
        attention_weights.attention_output_weight.bias   = weights_ptr[7];
        attn_layernorm_weights.gamma                     = weights_ptr[8];
        attn_layernorm_weights.beta                      = weights_ptr[9];
        ffn_weights.intermediate_weight.kernel           = weights_ptr[10];
        ffn_weights.intermediate_weight.bias             = weights_ptr[11];
        ffn_weights.output_weight.kernel                 = weights_ptr[12];
        ffn_weights.output_weight.bias                   = weights_ptr[13];
        ffn_layernorm_weights.gamma                      = weights_ptr[14];
        ffn_layernorm_weights.beta                       = weights_ptr[15];

        scale_list_.d_scale_list_        = scale_list_ptr[0];
        scale_list_.h_scale_list_        = scale_list_ptr[1];
        attention_weights.scale_list_ptr = &scale_list_;
        ffn_weights.scale_list_ptr       = &scale_list_;

        is_maintain_buffer = true;
    }
    int    embed_dim_;
    int    inter_size_;
    int    layer_idx_;
    bool   is_maintain_buffer = false;
    T*     weights_ptr[WEIGHT_N]{nullptr};
    size_t weights_size[WEIGHT_N];
    bool   is_maintain_sp_buffer = false;
    float* scale_list_ptr[2];
};

#undef WEIGHT_N

}  // namespace fastertransformer
