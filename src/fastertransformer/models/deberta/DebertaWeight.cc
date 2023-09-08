/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "DebertaWeight.h"

namespace fastertransformer {

template<typename T>
DebertaWeight<T>::DebertaWeight(const size_t hidden_units,
                                const size_t inter_size,
                                const size_t max_relative_positions,
                                const size_t relative_position_buckets,
                                const size_t vocab_size,
                                const size_t num_layer,
                                const size_t tensor_para_size,
                                const size_t tensor_para_rank,
                                const size_t pipeline_para_size,
                                const size_t pipeline_para_rank):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    max_relative_positions_(max_relative_positions),
    relative_position_buckets_(relative_position_buckets),
    vocab_size_(vocab_size),
    num_layer_(num_layer),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    pipeline_para_size_(pipeline_para_size),
    pipeline_para_rank_(pipeline_para_rank)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    // [1] word embedding weight [2] word-LN weight [3] word-LN bias [4] relative embedding weight [5] relative-LN
    // weight [6] relative-LN bias
    weights_size[0] = vocab_size_ * hidden_units_;
    weights_size[1] = hidden_units_;
    weights_size[2] = hidden_units_;
    weights_size[3] =
        (relative_position_buckets_ > 0 ? relative_position_buckets_ * 2 : max_relative_positions_ * 2) * hidden_units_;
    weights_size[4] = hidden_units_;
    weights_size[5] = hidden_units_;

    for (int i = 0; i < weights_num_; i++) {
        deviceMalloc(&weights_ptr[i], weights_size[i]);
    }

    setWeightPtr();
    deberta_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; i++) {
        deberta_layer_weights.push_back(
            DebertaLayerWeight<T>(hidden_units_, inter_size_, tensor_para_size_, tensor_para_rank_));
    }
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
DebertaWeight<T>::~DebertaWeight()
{
    if (is_maintain_buffer == true) {
        deberta_layer_weights.clear();
        for (int i = 0; i < weights_num_; i++) {
            deviceFree(weights_ptr[i]);
        }

        word_embedding_table                       = nullptr;
        word_embedding_layernorm_weights.gamma     = nullptr;
        word_embedding_layernorm_weights.beta      = nullptr;
        relative_embedding_table                   = nullptr;
        relative_embedding_layernorm_weights.gamma = nullptr;
        relative_embedding_layernorm_weights.beta  = nullptr;

        is_maintain_buffer = false;
    }
}

template<typename T>
DebertaWeight<T>::DebertaWeight(const DebertaWeight& other):
    DebertaWeight(other.hidden_units_,
                  other.inter_size_,
                  other.max_relative_positions_,
                  other.relative_position_buckets_,
                  other.vocab_size_,
                  other.num_layer_,
                  other.tensor_para_size_,
                  other.tensor_para_rank_,
                  other.pipeline_para_size_,
                  other.pipeline_para_rank_)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    deberta_layer_weights.clear();
    deberta_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; i++) {
        deberta_layer_weights.push_back(other.deberta_layer_weights[i]);
    }
    for (int i = 0; i < weights_num_; i++) {
        deviceMalloc(&weights_ptr[i], weights_size[i]);
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }

    setWeightPtr();
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
DebertaWeight<T>& DebertaWeight<T>::operator=(const DebertaWeight& other)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    hidden_units_              = other.hidden_units_;
    inter_size_                = other.inter_size_;
    max_relative_positions_    = other.max_relative_positions_;
    relative_position_buckets_ = other.relative_position_buckets_;
    vocab_size_                = other.vocab_size_;
    num_layer_                 = other.num_layer_;
    tensor_para_size_          = other.tensor_para_size_;
    tensor_para_rank_          = other.tensor_para_rank_;
    pipeline_para_size_        = other.pipeline_para_size_;
    pipeline_para_rank_        = other.pipeline_para_rank_;

    deberta_layer_weights.clear();
    deberta_layer_weights.reserve(num_layer_);
    for (int i = 0; i < num_layer_; i++) {
        deberta_layer_weights.push_back(other.deberta_layer_weights[i]);
    }
    for (int i = 0; i < weights_num_; i++) {
        deviceMalloc(&weights_ptr[i], weights_size[i]);
        cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
    }

    setWeightPtr();
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);

    return *this;
}

template<typename T>
void DebertaWeight<T>::loadModel(std::string dir_path)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FtCudaDataType model_file_type =
        getModelFileType(dir_path + "/config.ini", "deberta");  // by default FP32 if no .ini exists

    // Note: TensorFlow/PyTorch interface usage is unclear at this time, so weight loading are based on named bin for
    // now
    loadWeightFromBin<T>(weights_ptr[0],
                         {(size_t)weights_size[0]},
                         dir_path + "/model.embeddings.word_embeddings.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[1],
                         {(size_t)weights_size[1]},
                         dir_path + "/model.embeddings.LayerNorm.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[2], {(size_t)weights_size[2]}, dir_path + "/model.embeddings.LayerNorm.bias.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[3],
                         {(size_t)weights_size[3]},
                         dir_path + "/model.encoder.rel_embeddings.weight.bin",
                         model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[4], {(size_t)weights_size[4]}, dir_path + "/model.encoder.LayerNorm.weight.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[5], {(size_t)weights_size[5]}, dir_path + "/model.encoder.LayerNorm.bias.bin", model_file_type);

    for (uint l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l)) {
            deberta_layer_weights[l].loadModel(dir_path + "model.encoder.layer." + std::to_string(l) + ".",
                                               model_file_type);
        }
    }
    FT_LOG_DEBUG(__PRETTY_FUNCTION__, " stop");
}

template<typename T>
bool DebertaWeight<T>::isValidLayerParallelId(int l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_rank_)
           && (l < local_num_layer * (pipeline_para_rank_ + 1));
}

template<typename T>
void DebertaWeight<T>::setWeightPtr()
{
    word_embedding_table                   = weights_ptr[0];
    word_embedding_layernorm_weights.gamma = weights_ptr[1];
    word_embedding_layernorm_weights.beta  = weights_ptr[2];

    relative_embedding_table                   = weights_ptr[3];
    relative_embedding_layernorm_weights.gamma = weights_ptr[4];
    relative_embedding_layernorm_weights.beta  = weights_ptr[5];

    is_maintain_buffer = true;
}

template struct DebertaWeight<float>;
template struct DebertaWeight<half>;
#ifdef ENABLE_BF16
template struct DebertaWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
