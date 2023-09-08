/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "src/fastertransformer/models/gptneox/GptNeoX.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#ifdef BUILD_PYBIND
#include "src/fastertransformer/th_op/gptneox/utils/nccl_inherit_utils.h"
#include "src/fastertransformer/th_op/gptneox/utils/pybind_callback_utils.h"
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
#endif
namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

using std::vector;

class IFGptNeoX {
public:
    virtual ~IFGptNeoX() {}
    virtual void forward(th::Tensor&              input_ids,
                         th::Tensor&              input_lengths,
                         th::Tensor&              output_ids,
                         th::Tensor&              sequence_lengths,
                         th::Tensor&              cum_log_probs,
                         const size_t             request_output_len,
                         const size_t             beam_width,
                         th::optional<th::Tensor> top_k_opt,
                         th::optional<th::Tensor> top_p_opt,
                         th::optional<th::Tensor> beam_search_diversity_rate_opt,
                         th::optional<th::Tensor> temperature_opt,
                         th::optional<th::Tensor> len_penalty_opt,
                         th::optional<th::Tensor> repetition_penalty_opt,
                         th::optional<th::Tensor> random_seed_opt,
                         th::optional<th::Tensor> stop_words_list_opt,
                         th::optional<th::Tensor> optional_last_tokens_opt,
#ifdef BUILD_PYBIND
                         th::optional<int64_t> return_cum_log_probs_opt,
                         py::object            callback_opt) = 0;
#else
                         th::optional<int64_t> return_cum_log_probs_opt) = 0;
#endif
};

template<typename T>
class FTGptNeoX: public IFGptNeoX {
public:
    FTGptNeoX(
#ifdef BUILD_PYBIND
        nccl_inherit::HackGroupNCCL* nccl_hack_group,
        const size_t                 rank,
#endif
        const size_t             head_num,
        const size_t             size_per_head,
        const size_t             inter_size,
        const size_t             layer_num,
        const size_t             vocab_size,
        const size_t             rotary_embedding_dim,
        const int                start_id,
        const int                end_id,
        const int64_t            tensor_para_size,
        const int64_t            pipeline_para_size,
        const int64_t            int8_mode,
        const size_t             max_seq_len,
        const bool               use_gptj_residual,
        const vector<th::Tensor> weights,
        const vector<th::Tensor> int8_weights,
        const vector<th::Tensor> scale):
#ifdef BUILD_PYBIND
        nccl_hack_group_(nccl_hack_group),
        rank_(rank),
#endif
        head_num_(head_num),
        size_per_head_(size_per_head),
        inter_size_(inter_size),
        layer_num_(layer_num),
        vocab_size_(vocab_size),
        rotary_embedding_dim_(rotary_embedding_dim),
        start_id_(start_id),
        end_id_(end_id),
        use_gptj_residual_(use_gptj_residual),
        weights_(weights),
        int8_weights_(int8_weights),
        scale_(scale),
        tensor_para_size_(tensor_para_size),
        pipeline_para_size_(pipeline_para_size),
        int8_mode_(int8_mode)
    {
        ft::check_cuda_error(cublasLtCreate(&cublasltHandle_));
        cublas_algo_map_      = new ft::cublasAlgoMap(GEMM_CONFIG, "");
        cublas_wrapper_mutex_ = new std::mutex();

#ifdef BUILD_PYBIND
        nccl_inherit::ftNcclInitialize(
            tensor_para_, pipeline_para_, tensor_para_size, pipeline_para_size, rank, nccl_hack_group);
#else
        ftNcclInitialize(tensor_para_, pipeline_para_, tensor_para_size, pipeline_para_size);
#endif

        gpt_weights_.resizeLayer(layer_num_);
        for (int i = 0; i < (int)layer_num_; i++) {
            gpt_weights_.decoder_layer_weights[i]->pre_layernorm_weights.beta =
                get_ptr<T>(weights_[i + 0 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->pre_layernorm_weights.gamma =
                get_ptr<T>(weights_[i + 1 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.kernel =
                get_ptr<T>(weights_[i + 2 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.bias =
                get_ptr<T>(weights_[i + 3 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.kernel =
                get_ptr<T>(weights_[i + 4 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.bias =
                get_ptr<T>(weights_[i + 5 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.kernel =
                get_ptr<T>(weights_[i + 6 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.bias =
                get_ptr<T>(weights_[i + 7 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.kernel =
                get_ptr<T>(weights_[i + 8 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.bias =
                get_ptr<T>(weights_[i + 9 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->post_attention_layernorm_weights.beta =
                get_ptr<T>(weights_[i + 10 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->post_attention_layernorm_weights.gamma =
                get_ptr<T>(weights_[i + 11 * layer_num_]);
	
            if (int8_mode_ != 0) {
                gpt_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.int8_kernel =
                    get_ptr<int8_t>(int8_weights_[i + 0 * layer_num_]);
                gpt_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.int8_kernel =
                    get_ptr<int8_t>(int8_weights_[i + 1 * layer_num_]);
                gpt_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.int8_kernel =
                    get_ptr<int8_t>(int8_weights_[i + 2 * layer_num_]);
                gpt_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.int8_kernel =
                    get_ptr<int8_t>(int8_weights_[i + 3 * layer_num_]);
                if (int8_mode == 1) {
                    gpt_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.weight_only_quant_scale =
                        get_ptr<T>(scale_[i + 0 * layer_num_]);
                    gpt_weights_.decoder_layer_weights[i]
                        ->self_attention_weights.attention_output_weight.weight_only_quant_scale =
                        get_ptr<T>(scale_[i + 1 * layer_num_]);
                    gpt_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.weight_only_quant_scale =
                        get_ptr<T>(scale_[i + 2 * layer_num_]);
                    gpt_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.weight_only_quant_scale =
                        get_ptr<T>(scale_[i + 3 * layer_num_]);
                }
            }
        }

        gpt_weights_.pre_decoder_embedding_table   = get_ptr<T>(weights_[12 * layer_num_ + 0]);
        gpt_weights_.post_decoder_layernorm.gamma  = get_ptr<T>(weights_[12 * layer_num_ + 1]);
        gpt_weights_.post_decoder_layernorm.beta   = get_ptr<T>(weights_[12 * layer_num_ + 2]);
        gpt_weights_.post_decoder_embedding.kernel = get_ptr<T>(weights_[12 * layer_num_ + 3]);

        gpt_weights_.setMaxSeqLen(max_seq_len);

        ft::check_cuda_error(cudaGetDeviceProperties(&prop_, 0));

        auto           stream       = at::cuda::getCurrentCUDAStream().stream();
        cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(cublasHandle, stream);
        allocator_      = new ft::Allocator<ft::AllocatorType::TH>();
        cublas_wrapper_ = new ft::cublasMMWrapper(
            cublasHandle, cublasltHandle_, stream, cublas_algo_map_, cublas_wrapper_mutex_, allocator_);

        if (std::is_same<T, half>::value) {
            cublas_wrapper_->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
        }
        else if (std::is_same<T, float>::value) {
            cublas_wrapper_->setFP32GemmConfig();
        }

        ft::AttentionType attention_type = ft::getAttentionType<T>(size_per_head_,
                                                                   ft::getSMVersion(),
                                                                   true,   // remove_padding
                                                                   0,      // gpt supports any-seq-length fmha
                                                                   true,   // is_fuse
                                                                   false,  // with_relative_position_bias
                                                                   true);  // causal_mask

        gpt_ = new ft::GptNeoX<T>(head_num_,
                                  size_per_head_,
                                  inter_size_,
                                  layer_num_,
                                  vocab_size_,
                                  rotary_embedding_dim_,
                                  start_id_,
                                  end_id_,
                                  end_id_ + 1,  // p/prompt tuning virtual token start id
                                  ft::PromptLearningType::no_prompt,
                                  use_gptj_residual_,
                                  0.0f,  // beam_search_diversity_rate,
                                  1,     // top_k,
                                  0.0,   // top_p,
                                  0,     // random_seed,
                                  1.0f,  // temperature,
                                  1.0f,  // len_penalty,
                                  1.0f,  // repetition_penalty,
                                  tensor_para_,
                                  pipeline_para_,
                                  stream,
                                  cublas_wrapper_,
                                  allocator_,
                                  false,           // is_free_buffer_after_forward
                                  &prop_,          // cuda_device_prop
                                  attention_type,  // attention_type
                                  nullptr,         // custom_all_reduce_comm
                                  0,               // enable_custom_all_reduce
                                  int8_mode_);
    }

    ~FTGptNeoX() override
    {
        delete gpt_;
        delete cublas_wrapper_;
        delete allocator_;

        ft::ftNcclParamDestroy(tensor_para_);
        ft::ftNcclParamDestroy(pipeline_para_);
        cublasLtDestroy(cublasltHandle_);
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    void forward(th::Tensor&              input_ids,
                 th::Tensor&              input_lengths,
                 th::Tensor&              output_ids,
                 th::Tensor&              sequence_lengths,
                 th::Tensor&              cum_log_probs,
                 const size_t             request_output_len,
                 const size_t             beam_width,
                 th::optional<th::Tensor> top_k_opt,
                 th::optional<th::Tensor> top_p_opt,
                 th::optional<th::Tensor> beam_search_diversity_rate_opt,
                 th::optional<th::Tensor> temperature_opt,
                 th::optional<th::Tensor> len_penalty_opt,
                 th::optional<th::Tensor> repetition_penalty_opt,
                 th::optional<th::Tensor> random_seed_opt,
                 th::optional<th::Tensor> stop_words_list_opt,
                 th::optional<th::Tensor> optional_last_tokens_opt,
#ifdef BUILD_PYBIND
                 th::optional<int64_t> return_cum_log_probs_opt,
                 py::object            callback_opt) override
#else
                 th::optional<int64_t> return_cum_log_probs_opt) override
#endif
    {
        int return_cum_log_probs = return_cum_log_probs_opt.has_value() ? (int)return_cum_log_probs_opt.value() : 0;
        const size_t request_batch_size = (size_t)input_ids.size(0);
        const size_t max_input_length   = (size_t)input_ids.size(1);
        const int    total_output_len   = (int)(max_input_length + request_output_len);

#ifdef BUILD_PYBIND
        pybind_callback::__ctx__ ctx;
        if (!callback_opt.is_none()) {
            auto callback = callback_opt.cast<std::function<void(py::dict)>>();
            ctx = pybind_callback::__ctx__{callback, input_lengths, (int)request_batch_size, (int)beam_width, end_id_};
            gpt_->registerCallback(&pybind_callback::callback, &ctx);
        }
        else {
            gpt_->unRegisterCallback();
        }
#endif

        std::vector<uint32_t> output_seq_len(request_batch_size, total_output_len);

        std::unordered_map<std::string, ft::Tensor> input_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"input_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, max_input_length},
                        get_ptr<int>(input_ids)}},
            {"input_lengths",
             ft::Tensor{
                 ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size}, get_ptr<int>(input_lengths)}},
            {"output_seq_len",
             ft::Tensor{
                 ft::MEMORY_CPU, ft::TYPE_UINT32, std::vector<size_t>{request_batch_size}, output_seq_len.data()}}};
        if (stop_words_list_opt.has_value()) {
            auto stop_words_list = stop_words_list_opt.value();
            // [batch_size, 2, stop_words_length]
            input_tensors.insert(
                {"stop_words_list",
                 ft::Tensor{ft::MEMORY_GPU,
                            ft::TYPE_INT32,
                            std::vector<size_t>{request_batch_size, 2, (size_t)stop_words_list.size(2)},
                            get_ptr<int>(stop_words_list)}});
        }
        if (optional_last_tokens_opt.has_value()) {
            auto optional_last_tokens = optional_last_tokens_opt.value();
            input_tensors.insert(
                {"optional_last_tokens",
                 ft::Tensor{ft::MEMORY_GPU,
                            ft::TYPE_INT32,
                            std::vector<size_t>{request_batch_size, (size_t)optional_last_tokens.size(1)},
                            get_ptr<int>(optional_last_tokens)}});
        }
        if (beam_width > 1 && beam_search_diversity_rate_opt.has_value()) {
            input_tensors.insert(
                {"beam_search_diversity_rate",
                 convert_tensor<float>(beam_search_diversity_rate_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (top_p_opt.has_value()) {
            input_tensors.insert(
                {"runtime_top_p", convert_tensor<float>(top_p_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (top_k_opt.has_value()) {
            input_tensors.insert(
                {"runtime_top_k", convert_tensor<uint>(top_k_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (temperature_opt.has_value()) {
            input_tensors.insert(
                {"temperature", convert_tensor<float>(temperature_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (len_penalty_opt.has_value()) {
            input_tensors.insert(
                {"len_penalty", convert_tensor<float>(len_penalty_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (repetition_penalty_opt.has_value()) {
            input_tensors.insert({"repetition_penalty",
                                  convert_tensor<float>(repetition_penalty_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (random_seed_opt.has_value()) {
            input_tensors.insert(
                {"random_seed",
                 convert_tensor<unsigned long long int>(random_seed_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }

        std::unordered_map<std::string, ft::Tensor> output_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"output_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, beam_width, (size_t)total_output_len},
                        get_ptr<int>(output_ids)}},
            {"sequence_length",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, beam_width},
                        get_ptr<int>(sequence_lengths)}}};

        if (return_cum_log_probs > 0) {
            output_tensors.insert({"cum_log_probs",
                                   ft::Tensor{ft::MEMORY_GPU,
                                              ft::TYPE_FP32,
                                              std::vector<size_t>{request_batch_size, beam_width},
                                              get_ptr<float>(cum_log_probs)}});
        }

        try {
            gpt_->forward(&output_tensors, &input_tensors, &gpt_weights_);
        }
        catch (std::runtime_error& error) {
            std::cout << error.what();
            exit(-1);
        }
        catch (...) {
            std::cout << "Runtime error";
            exit(-1);
        }
    }

private:
#ifdef BUILD_PYBIND
    int                          rank_ = 0;
    nccl_inherit::HackGroupNCCL* nccl_hack_group_;
#endif
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t inter_size_;
    const size_t layer_num_;
    const size_t vocab_size_;
    const size_t rotary_embedding_dim_;
    const int    start_id_;
    const int    end_id_;
    const bool   use_gptj_residual_;

    const int64_t int8_mode_ = 0;

    // const ft::gptVariantParams gpt_variant_params_;

    std::vector<th::Tensor> int8_weights_;
    std::vector<th::Tensor> scale_;
    std::vector<th::Tensor> weights_;
    cublasLtHandle_t        cublasltHandle_;
    std::mutex*             cublas_wrapper_mutex_;
    ft::cublasAlgoMap*      cublas_algo_map_;
    struct cudaDeviceProp   prop_;
    ft::GptNeoXWeight<T>    gpt_weights_;

    ft::GptNeoX<T>*                       gpt_;
    ft::cublasMMWrapper*                  cublas_wrapper_;
    ft::Allocator<ft::AllocatorType::TH>* allocator_;

    ft::NcclParam tensor_para_;
    ft::NcclParam pipeline_para_;

    int64_t tensor_para_size_;
    int64_t pipeline_para_size_;
};

class GptNeoXOp: public th::jit::CustomClassHolder {
public:
    GptNeoXOp(
#ifdef BUILD_PYBIND
        const c10d::ProcessGroup& nccl_process_group,
        const int64_t             rank,
#endif
        const int64_t            head_num,
        const int64_t            size_per_head,
        const int64_t            inter_size,
        const int64_t            layer_num,
        const int64_t            vocab_size,
        const int64_t            rotary_embedding_dim,
        const int64_t            start_id,
        const int64_t            end_id,
        const int64_t            tensor_para_size,
        const int64_t            pipeline_para_size,
        const int64_t            int8_mode,
        const int64_t            max_seq_len,
        const bool               use_gptj_residual,
        const vector<th::Tensor> weights,
        const vector<th::Tensor> int8_weights,
        const vector<th::Tensor> scale);

    ~GptNeoXOp();

    vector<th::Tensor> forward(th::Tensor               input_ids,
                               th::Tensor               input_lengths,
                               const int64_t            output_len,
                               th::optional<int64_t>    beam_width_opt,
                               th::optional<th::Tensor> top_k_opt,
                               th::optional<th::Tensor> top_p_opt,
                               th::optional<th::Tensor> beam_search_diversity_rate_opt,
                               th::optional<th::Tensor> temperature_opt,
                               th::optional<th::Tensor> len_penalty_opt,
                               th::optional<th::Tensor> repetition_penalty_opt,
                               th::optional<th::Tensor> random_seed_opt,
                               th::optional<th::Tensor> stop_words_list_opt,
                               th::optional<th::Tensor> optional_last_tokens_opt,
#ifdef BUILD_PYBIND
                               th::optional<int64_t> return_cum_log_probs_opt,
                               py::object            callback_opt);
#else
                               th::optional<int64_t> return_cum_log_probs_opt);
#endif

private:
    const at::ScalarType    st_;
    IFGptNeoX*              ftgpt;
    std::vector<th::Tensor> weights;
#ifdef BUILD_PYBIND
    nccl_inherit::HackGroupNCCL* nccl_hack_group_;
#endif
};

}  // namespace torch_ext
