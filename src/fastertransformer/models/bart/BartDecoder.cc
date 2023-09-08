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

#include "src/fastertransformer/models/bart/BartDecoder.h"

namespace fastertransformer {

template<typename T>
void BartDecoder<T>::initialize()
{
    self_attention_layer_ =
        new TensorParallelDecoderSelfAttentionLayer<T>(max_batch_size_,
                                                       head_num_,
                                                       size_per_head_,
                                                       d_model_,
                                                       q_scaling_,  // adjust according to checkpoint structure
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       false,
                                                       0,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    // (1.0f/ sqrtf((float)size_per_head_))
    cross_attention_layer_ =
        new TensorParallelDecoderCrossAttentionLayer<T>(max_batch_size_,
                                                        head_num_,
                                                        size_per_head_,
                                                        d_model_,
                                                        q_scaling_,  // adjust according to checkpoint structure
                                                        tensor_para_,
                                                        stream_,
                                                        cublas_wrapper_,
                                                        allocator_,
                                                        is_free_buffer_after_forward_,
                                                        custom_all_reduce_comm_,
                                                        enable_custom_all_reduce_);

    bool use_gated_activation = activation_type_ == ActivationType::GeGLU || activation_type_ == ActivationType::ReGLU
                                || activation_type_ == ActivationType::SiGLU;
    if (activation_type_ == ActivationType::Gelu || activation_type_ == ActivationType::GeGLU) {
        ffn_layer_ = new TensorParallelGeluFfnLayer<T>(max_batch_size_,
                                                       1,
                                                       1,
                                                       d_model_,
                                                       0,
                                                       inter_size_,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       false,
                                                       0,
                                                       use_gated_activation,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
    else if (activation_type_ == ActivationType::Relu || activation_type_ == ActivationType::ReGLU) {
        ffn_layer_ = new TensorParallelReluFfnLayer<T>(max_batch_size_,
                                                       1,
                                                       1,
                                                       d_model_,
                                                       0,
                                                       inter_size_,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       false,
                                                       0,
                                                       use_gated_activation,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
    else if (activation_type_ == ActivationType::Silu || activation_type_ == ActivationType::SiGLU) {
        ffn_layer_ = new TensorParallelSiluFfnLayer<T>(max_batch_size_,
                                                       1,
                                                       1,
                                                       d_model_,
                                                       0,
                                                       inter_size_,
                                                       tensor_para_,
                                                       stream_,
                                                       cublas_wrapper_,
                                                       allocator_,
                                                       true,
                                                       is_free_buffer_after_forward_,
                                                       false,
                                                       use_gated_activation,
                                                       custom_all_reduce_comm_,
                                                       enable_custom_all_reduce_);
    }
}

template<typename T>
void BartDecoder<T>::allocateBuffer()
{
    if (is_allocate_buffer_ == false) {
        decoder_normed_input_ = reinterpret_cast<T*>(
            allocator_->reMalloc(decoder_normed_input_, sizeof(T) * max_batch_size_ * d_model_, false));
        self_attn_output_ = reinterpret_cast<T*>(
            allocator_->reMalloc(self_attn_output_, sizeof(T) * max_batch_size_ * d_model_, false));
        normed_self_attn_output_ = reinterpret_cast<T*>(
            allocator_->reMalloc(normed_self_attn_output_, sizeof(T) * max_batch_size_ * d_model_, false));
        cross_attn_output_ = reinterpret_cast<T*>(
            allocator_->reMalloc(cross_attn_output_, sizeof(T) * max_batch_size_ * d_model_, false));
        normed_cross_attn_output_ = reinterpret_cast<T*>(
            allocator_->reMalloc(normed_cross_attn_output_, sizeof(T) * max_batch_size_ * d_model_, false));
        decoder_layer_output_ = reinterpret_cast<T*>(
            allocator_->reMalloc(decoder_layer_output_, sizeof(T) * max_batch_size_ * d_model_, false));
        is_allocate_buffer_ = true;
    }
}

template<typename T>
void BartDecoder<T>::allocateBuffer(size_t batch_size)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    decoder_normed_input_ =
        reinterpret_cast<T*>(allocator_->reMalloc(decoder_normed_input_, sizeof(T) * batch_size * d_model_, false));
    self_attn_output_ =
        reinterpret_cast<T*>(allocator_->reMalloc(self_attn_output_, sizeof(T) * batch_size * d_model_, false));
    normed_self_attn_output_ =
        reinterpret_cast<T*>(allocator_->reMalloc(normed_self_attn_output_, sizeof(T) * batch_size * d_model_, false));
    cross_attn_output_ =
        reinterpret_cast<T*>(allocator_->reMalloc(cross_attn_output_, sizeof(T) * batch_size * d_model_, false));
    normed_cross_attn_output_ =
        reinterpret_cast<T*>(allocator_->reMalloc(normed_cross_attn_output_, sizeof(T) * batch_size * d_model_, false));
    decoder_layer_output_ =
        reinterpret_cast<T*>(allocator_->reMalloc(decoder_layer_output_, sizeof(T) * batch_size * d_model_, false));
    is_allocate_buffer_ = true;
}

template<typename T>
void BartDecoder<T>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_ == true) {
        allocator_->free((void**)(&decoder_normed_input_));
        allocator_->free((void**)(&self_attn_output_));
        allocator_->free((void**)(&normed_self_attn_output_));
        allocator_->free((void**)(&cross_attn_output_));
        allocator_->free((void**)(&normed_cross_attn_output_));
        allocator_->free((void**)(&decoder_layer_output_));
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool BartDecoder<T>::isValidBatchSize(size_t batch_size)
{
    if (batch_size <= max_batch_size_) {
        return true;
    }
    else {
        freeBuffer();
        max_batch_size_ = batch_size * 1.2;
        return true;
    }
}

template<typename T>
BartDecoder<T>::BartDecoder(size_t                              max_batch_size,
                            size_t                              head_num,
                            size_t                              size_per_head,
                            size_t                              inter_size,
                            size_t                              d_model,
                            size_t                              num_layer,
                            float                               layernorm_eps,
                            cudaStream_t                        stream,
                            cublasMMWrapper*                    cublas_wrapper,
                            IAllocator*                         allocator,
                            bool                                is_free_buffer_after_forward,
                            NcclParam                           tensor_para,
                            NcclParam                           pipeline_para,
                            ActivationType                      activation_type,
                            LayerNormType                       layernorm_type,
                            float                               q_scaling,
                            std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                            int                                 enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward),
    max_batch_size_(max_batch_size),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    d_model_(d_model),
    num_layer_(num_layer),
    layernorm_eps_(layernorm_eps),
    hidden_units_(head_num_ * size_per_head),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    activation_type_(activation_type),
    layernorm_type_(layernorm_type),
    q_scaling_(q_scaling),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce)
{
    initialize();
}

template<typename T>
BartDecoder<T>::BartDecoder(BartDecoder<T> const& decoder):
    BaseLayer(decoder.stream_, decoder.cublas_wrapper_, decoder.allocator_, decoder.is_free_buffer_after_forward_),
    max_batch_size_(decoder.max_batch_size_),
    head_num_(decoder.head_num_),
    size_per_head_(decoder.size_per_head_),
    inter_size_(decoder.inter_size_),
    d_model_(decoder.d_model_),
    num_layer_(decoder.num_layer_),
    layernorm_eps_(decoder.layernorm_eps_),
    hidden_units_(decoder.hidden_units_),
    tensor_para_(decoder.tensor_para_),
    pipeline_para_(decoder.pipeline_para_),
    activation_type_(decoder.activation_type_),
    layernorm_type_(decoder.layernorm_type_),
    q_scaling_(decoder.q_scaling_),
    custom_all_reduce_comm_(decoder.custom_all_reduce_comm_),
    enable_custom_all_reduce_(decoder.enable_custom_all_reduce_)
{
    initialize();
}

template<typename T>
BartDecoder<T>::~BartDecoder()
{
    delete self_attention_layer_;
    delete cross_attention_layer_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void BartDecoder<T>::setStream(cudaStream_t stream)
{
    self_attention_layer_->setStream(stream);
    cross_attention_layer_->setStream(stream);
    ffn_layer_->setStream(stream);
    BaseLayer::setStream(stream);
}

template<typename T>
bool BartDecoder<T>::isValidLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool BartDecoder<T>::isFirstLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool BartDecoder<T>::isLastLayerParallelId(uint l)
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return l < num_layer_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
int BartDecoder<T>::getFirstLayerParallelId()
{
    int local_num_layer = (int)(ceil(num_layer_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
void BartDecoder<T>::forward(std::vector<Tensor>*                           output_tensors,
                             const std::vector<Tensor>*                     input_tensors,
                             const std::vector<BartDecoderLayerWeight<T>*>* decoder_layer_weight)
{
    // input tensors:
    //      decoder_input [local_batch_size, d_model_],
    //      encoder_output [local_batch_size, mem_max_seq_len, mem_d_model_],
    //      encoder_sequence_length [local_batch_size],
    //      finished [local_batch_size],
    //      step [1] on cpu
    //      sequence_lengths [local_batch_size]
    //      relative_attention_bias [1, head_num, step, step] or [1, head_num, max_seq_len, max_seq_len]
    //      ite [1] on cpu
    //      cache_indirection [local_batch_size / beam_width, beam_width, max_seq_len]
    //              Here, local_batch_size contains the beam_width, so local_batch_size / beam_width
    //              is real local_batch_size.

    // output tensors:
    //      decoder_output [local_batch_size, d_model_],
    //      key_cache [num_layer / pipeline_para_.world_size_, batch, head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [num_layer / pipeline_para_.world_size_, batch, head_num, max_seq_len, size_per_head]
    //      key_mem_cache [num_layer / pipeline_para_.world_size_, batch_size, mem_max_seq_len, hidden_dimension],
    //      value_mem_cache [num_layer / pipeline_para_.world_size_, batch_size, mem_max_seq_len, hidden_dimension]
    //      attention_output: shape = [num_layer / pipeline_para_.world_size_, batch_size, beam,
    //          head_num / tensor_para_.world_size_, max_seq_len, mem_max_seq_len]
    //          offset = [batch_offset, layer_offset_base] optional, float*

    FT_CHECK(input_tensors->size() == 9);
    FT_CHECK(output_tensors->size() == 5 || output_tensors->size() == 6);
    isValidBatchSize(input_tensors->at(0).shape[0]);
    const size_t local_batch_size = input_tensors->at(0).shape[0];
    allocateBuffer(local_batch_size);

    const size_t   mem_max_seq_len = input_tensors->at(1).shape[1];
    const uint     ite             = input_tensors->at(7).getVal<uint>();
    const DataType data_type       = getTensorType<T>();

    std::vector<size_t> self_k_cache_shape;
    self_k_cache_shape.push_back(local_batch_size);
    for (auto t = output_tensors->at(1).shape.begin() + 2; t != output_tensors->at(1).shape.end(); ++t) {
        self_k_cache_shape.push_back(*t);
    }
    std::vector<size_t> self_v_cache_shape;
    self_v_cache_shape.push_back(local_batch_size);
    for (auto t = output_tensors->at(2).shape.begin() + 2; t != output_tensors->at(2).shape.end(); ++t) {
        self_v_cache_shape.push_back(*t);
    }

    const std::vector<size_t> mem_cache_shape = {
        local_batch_size, output_tensors->at(3).shape[2], output_tensors->at(3).shape[3]};

    const bool output_cross_attention = output_tensors->size() == 6;
    const uint max_seq_len            = output_cross_attention ? output_tensors->at(5).shape[4] : 0;

    for (uint l = 0; l < num_layer_; l++) {
        if (isValidLayerParallelId(l) == false) {
            continue;
        }

        T* decoder_input  = (l == 0) ? input_tensors->at(0).getPtr<T>() : decoder_layer_output_;
        T* decoder_output = (l == num_layer_ - 1) ? output_tensors->at(0).getPtr<T>() : decoder_layer_output_;

        if (isFirstLayerParallelId(l) == true && pipeline_para_.rank_ != 0 && pipeline_para_.world_size_ > 1) {
            // ftNcclRecv(decoder_input, local_batch_size * d_model_, pipeline_para_.rank_ - 1,
            // pipeline_para_, stream_);

            ftNcclRecv(decoder_input + local_batch_size * d_model_ / tensor_para_.world_size_ * tensor_para_.rank_,
                       local_batch_size * d_model_ / tensor_para_.world_size_,
                       pipeline_para_.rank_ - 1,
                       pipeline_para_,
                       stream_);
            if (tensor_para_.world_size_ > 1) {
                ftNcclAllGather(decoder_input,
                                decoder_input,
                                (int)local_batch_size * d_model_ / tensor_para_.world_size_,
                                tensor_para_.rank_,
                                tensor_para_,
                                stream_);
            }
        }

        size_t cache_offset = l - getFirstLayerParallelId();
        for (auto t = output_tensors->at(1).shape.begin() + 1; t != output_tensors->at(1).shape.end(); ++t) {
            cache_offset *= *t;
        };
        size_t ite_cache_offset = ite * local_batch_size;
        for (auto t = output_tensors->at(1).shape.begin() + 2; t != output_tensors->at(1).shape.end(); ++t) {
            ite_cache_offset *= *t;
        }
        cache_offset += ite_cache_offset;

        size_t mem_cache_offset = l - getFirstLayerParallelId();
        for (auto t = output_tensors->at(3).shape.begin() + 1; t != output_tensors->at(3).shape.end(); ++t) {
            mem_cache_offset *= *t;
        };
        ite_cache_offset = ite * local_batch_size;
        for (auto t = output_tensors->at(3).shape.begin() + 2; t != output_tensors->at(3).shape.end(); ++t) {
            ite_cache_offset *= *t;
        }
        mem_cache_offset += ite_cache_offset;

        if (layernorm_type_ == LayerNormType::pre_layernorm) {
            invokeGeneralT5LayerNorm(decoder_normed_input_,
                                     decoder_input,
                                     decoder_layer_weight->at(l)->self_attn_layernorm_weights.gamma,
                                     decoder_layer_weight->at(l)->self_attn_layernorm_weights.beta,
                                     layernorm_eps_,
                                     local_batch_size,
                                     d_model_,
                                     stream_);
        }
        sync_check_cuda_error();
        TensorMap self_attention_input_tensors{
            {"input_query",
             Tensor{MEMORY_GPU,
                    data_type,
                    {local_batch_size, d_model_},
                    layernorm_type_ == LayerNormType::pre_layernorm ? decoder_normed_input_ : decoder_input}},
            {"finished", input_tensors->at(3)},
            {"sequence_lengths", input_tensors->at(5)},
            {"step", input_tensors->at(4)},
        };
        self_attention_input_tensors.insertIfValid("relative_attention_bias", input_tensors->at(6));
        self_attention_input_tensors.insertIfValid("cache_indirection", input_tensors->at(8));

        TensorMap self_attention_output_tensors{
            {"hidden_features", Tensor{MEMORY_GPU, data_type, {local_batch_size, d_model_}, self_attn_output_}},
            {"key_cache",
             Tensor{MEMORY_GPU, data_type, self_k_cache_shape, output_tensors->at(1).getPtrWithOffset(cache_offset)}},
            {"value_cache",
             Tensor{MEMORY_GPU, data_type, self_v_cache_shape, output_tensors->at(2).getPtrWithOffset(cache_offset)}}};
        self_attention_layer_->forward(&self_attention_output_tensors,
                                       &self_attention_input_tensors,
                                       &decoder_layer_weight->at(l)->self_attention_weights);

        if (layernorm_type_ == LayerNormType::pre_layernorm) {
            invokeGeneralAddBiasResidualT5PreLayerNorm(
                self_attn_output_,
                normed_self_attn_output_,
                decoder_input,
                decoder_layer_weight->at(l)->cross_attn_layernorm_weights.gamma,
                decoder_layer_weight->at(l)->cross_attn_layernorm_weights.beta,
                decoder_layer_weight->at(l)->self_attention_weights.attention_output_weight.bias,
                layernorm_eps_,
                local_batch_size,
                d_model_,
                stream_);
        }
        else if (layernorm_type_ == LayerNormType::post_layernorm) {
            invokeGeneralAddBiasResidualT5PreLayerNorm(
                self_attn_output_,
                self_attn_output_,
                decoder_input,
                decoder_layer_weight->at(l)->self_attn_layernorm_weights.gamma,
                decoder_layer_weight->at(l)->self_attn_layernorm_weights.beta,
                decoder_layer_weight->at(l)->self_attention_weights.attention_output_weight.bias,
                layernorm_eps_,
                local_batch_size,
                d_model_,
                stream_);
        }
        sync_check_cuda_error();

        TensorMap cross_attention_input_tensors{
            {"input_query",
             Tensor{MEMORY_GPU,
                    data_type,
                    {local_batch_size, d_model_},
                    layernorm_type_ == LayerNormType::pre_layernorm ? normed_self_attn_output_ : self_attn_output_}},
            {"encoder_output", input_tensors->at(1)},
            {"encoder_sequence_length", input_tensors->at(2)},
            {"finished", input_tensors->at(3)},
            {"step", input_tensors->at(4)}};
        TensorMap cross_attention_output_tensors{
            {"hidden_features", Tensor{MEMORY_GPU, data_type, {local_batch_size, d_model_}, cross_attn_output_}},
            {"key_cache",
             Tensor{MEMORY_GPU, data_type, mem_cache_shape, output_tensors->at(3).getPtrWithOffset(mem_cache_offset)}},
            {"value_cache",
             Tensor{MEMORY_GPU, data_type, mem_cache_shape, output_tensors->at(4).getPtrWithOffset(mem_cache_offset)}}};
        if (output_cross_attention) {
            int          local_layer_id          = l - getFirstLayerParallelId();
            const size_t cross_attentions_offset = local_layer_id * output_tensors->at(5).offsets[1]
                                                   + output_tensors->at(5).offsets[0] * head_num_
                                                         / tensor_para_.world_size_ * max_seq_len * mem_max_seq_len;
            cross_attention_output_tensors.insert(
                "cross_attentions",
                Tensor{MEMORY_GPU,
                       TYPE_FP32,
                       {local_batch_size, head_num_ / tensor_para_.world_size_, max_seq_len, mem_max_seq_len},
                       output_tensors->at(5).getPtrWithOffset<float>(cross_attentions_offset)});
        }
        cross_attention_layer_->forward(&cross_attention_output_tensors,
                                        &cross_attention_input_tensors,
                                        &decoder_layer_weight->at(l)->cross_attention_weights);

        if (layernorm_type_ == LayerNormType::pre_layernorm) {
            invokeGeneralAddBiasResidualT5PreLayerNorm(
                cross_attn_output_,
                normed_cross_attn_output_,
                self_attn_output_,
                decoder_layer_weight->at(l)->layernorm_weights.gamma,
                decoder_layer_weight->at(l)->layernorm_weights.beta,
                decoder_layer_weight->at(l)->cross_attention_weights.attention_output_weight.bias,
                layernorm_eps_,
                local_batch_size,
                d_model_,
                stream_);
        }
        else if (layernorm_type_ == LayerNormType::post_layernorm) {
            invokeGeneralAddBiasResidualT5PreLayerNorm(
                cross_attn_output_,
                cross_attn_output_,
                self_attn_output_,
                decoder_layer_weight->at(l)->cross_attn_layernorm_weights.gamma,
                decoder_layer_weight->at(l)->cross_attn_layernorm_weights.beta,
                decoder_layer_weight->at(l)->cross_attention_weights.attention_output_weight.bias,
                layernorm_eps_,
                local_batch_size,
                d_model_,
                stream_);
        }
        sync_check_cuda_error();

        TensorMap ffn_input_tensors(
            {{"ffn_input",
              Tensor{MEMORY_GPU,
                     data_type,
                     {local_batch_size, d_model_},
                     layernorm_type_ == LayerNormType::pre_layernorm ? normed_cross_attn_output_ :
                                                                       cross_attn_output_}}});

        TensorMap ffn_output_tensors;

        ffn_output_tensors.insert("ffn_output",
                                  Tensor{MEMORY_GPU, data_type, {local_batch_size, d_model_}, decoder_output});
        ffn_layer_->forward(&ffn_output_tensors, &ffn_input_tensors, &decoder_layer_weight->at(l)->ffn_weights);

        if (layernorm_type_ == LayerNormType::pre_layernorm) {
            invokeT5AddBiasResidual(decoder_output,
                                    cross_attn_output_,
                                    decoder_layer_weight->at(l)->ffn_weights.output_weight.bias,
                                    local_batch_size,
                                    d_model_,
                                    stream_);
        }
        else if (layernorm_type_ == LayerNormType::post_layernorm) {
            invokeGeneralAddBiasResidualT5PreLayerNorm(decoder_output,
                                                       decoder_output,
                                                       cross_attn_output_,
                                                       decoder_layer_weight->at(l)->layernorm_weights.gamma,
                                                       decoder_layer_weight->at(l)->layernorm_weights.beta,
                                                       decoder_layer_weight->at(l)->ffn_weights.output_weight.bias,
                                                       layernorm_eps_,
                                                       local_batch_size,
                                                       d_model_,
                                                       stream_);
        }
        sync_check_cuda_error();

        if (isLastLayerParallelId(l) == true && pipeline_para_.rank_ != pipeline_para_.world_size_ - 1
            && pipeline_para_.world_size_ > 1) {
            // ftNcclSend(decoder_output, local_batch_size * d_model_, pipeline_para_.rank_ + 1,
            // pipeline_para_, stream_);

            ftNcclSend(decoder_output + local_batch_size * d_model_ / tensor_para_.world_size_ * tensor_para_.rank_,
                       local_batch_size * d_model_ / tensor_para_.world_size_,
                       pipeline_para_.rank_ + 1,
                       pipeline_para_,
                       stream_);
        }
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
}

template class BartDecoder<float>;
template class BartDecoder<half>;
#ifdef ENABLE_BF16
template class BartDecoder<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
