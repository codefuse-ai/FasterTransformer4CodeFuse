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

#include "src/fastertransformer/kernels/gen_relative_pos_bias.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/models/t5/T5DecoderLayerWeight.h"
#include "src/fastertransformer/utils/IA3.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
struct T5DecodingWeight {

    T5DecodingWeight() = default;
    T5DecodingWeight(size_t                head_num,
                     size_t                size_per_head,
                     size_t                d_model,
                     size_t                inter_size,
                     size_t                vocab_size,
                     size_t                num_layer,
                     size_t                mem_d_model,
                     size_t                num_bucket_or_max_seq_len,
                     size_t                tensor_para_size,
                     size_t                tensor_para_rank,
                     size_t                pipeline_para_size,
                     size_t                pipeline_para_rank,
                     bool                  t5_with_bias_para            = false,
                     bool                  use_gated_activation_para    = false,
                     PositionEmbeddingType position_embedding_type_para = PositionEmbeddingType::relative,
                     size_t                num_ia3_tasks                = false,
                     size_t                adapter_inter_size           = 0);
    ~T5DecodingWeight();
    T5DecodingWeight(const T5DecodingWeight& other);
    T5DecodingWeight& operator=(const T5DecodingWeight& other);

    std::vector<T5DecoderLayerWeight<T>*> decoder_layer_weights;
    const T*                              pre_decoder_embedding_table             = nullptr;
    const T*                              absolute_or_relative_position_embedding = nullptr;
    LayerNormWeight<T>                    post_decoder_layernorm;
    DenseWeight<T>                        post_decoder_embedding;
    bool                                  t5_with_bias         = false;
    bool                                  use_gated_activation = false;
    // 0 = relative_position_embedding,  1 = absolute_position_embedding
    PositionEmbeddingType position_embedding_type = PositionEmbeddingType::relative;

    void loadModel(std::string dir_path);
    void resizeLayer(const int num_layer);

    void setT5StructureDiff(bool                  t5_with_bias_para,
                            bool                  use_gated_activation_para,
                            PositionEmbeddingType position_embedding_type_para);

    inline size_t getNumIA3Tasks() const
    {
        return ia3_num_tasks_;
    };

private:
    void setWeightPtr();
    void mallocWeights();
    bool isValidLayerParallelId(int l);
    void initialize();

    size_t head_num_;
    size_t size_per_head_;
    size_t d_model_;
    size_t inter_size_;
    size_t vocab_size_;
    size_t num_layer_;
    size_t mem_d_model_;
    // refer to num_buckt if using relative position embedding
    // refer to max_seq_len if using absolute position embedding
    size_t num_bucket_or_max_seq_len_;
    size_t tensor_para_size_;
    size_t tensor_para_rank_;
    size_t pipeline_para_size_;
    size_t pipeline_para_rank_;
    size_t ia3_num_tasks_;
    size_t adapter_inter_size_;
    bool   is_maintain_buffer = false;
    bool   shared_embed_      = false;

    int real_weights_num_;

    const static int weights_num_ = 6;
    T*               weights_ptr[weights_num_];
    size_t           weights_size[weights_num_];
};

}  // namespace fastertransformer
