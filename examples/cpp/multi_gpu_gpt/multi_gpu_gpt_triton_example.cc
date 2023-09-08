/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <memory>
#include <thread>

#include "3rdparty/INIReader.h"
#include "examples/cpp/multi_gpu_gpt/gpt_example_utils.h"
#include "src/fastertransformer/triton_backend/multi_gpu_gpt/ParallelGptTritonModel.h"
#include "src/fastertransformer/triton_backend/multi_gpu_gpt/ParallelGptTritonModelInstance.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

namespace ft = fastertransformer;

std::vector<std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>>
broadcastRequest(const std::vector<int>&      v_start_ids,
                 const std::vector<int>&      v_start_lengths,
                 const int                    node_id,
                 const int                    gpu_count,
                 const int                    beam_width,
                 const int                    request_output_len,
                 const float                  beam_search_diversity_rate,
                 const uint                   runtime_top_k,
                 const float                  runtime_top_p,
                 const float                  temperature,
                 const float                  len_penalty,
                 const float                  repetition_penalty,
                 const float                  presence_penalty,
                 const int                    min_length,
                 const unsigned long long int random_seed,
                 const bool                   is_return_log_probs,
                 const bool                   is_return_context_cum_log_probs,
                 const bool                   is_return_context_embeddings,
                 std::vector<void*>*          pointer_record)
{
    // broadcast the request to all nodes, and copy "gpu_count" copies on different gpu
    int size_1 = v_start_ids.size();
    int size_2 = v_start_lengths.size();
    ft::mpi::bcast(&size_1, 1, ft::mpi::MPI_TYPE_INT, 0, ft::mpi::COMM_WORLD);
    ft::mpi::bcast(&size_2, 1, ft::mpi::MPI_TYPE_INT, 0, ft::mpi::COMM_WORLD);

    std::vector<int> v_input_ids(size_1);
    std::vector<int> v_input_lengths(size_2);

    if (node_id == 0) {
        memcpy(v_input_ids.data(), v_start_ids.data(), size_1 * sizeof(int));
        memcpy(v_input_lengths.data(), v_start_lengths.data(), size_2 * sizeof(int));
    }
    ft::mpi::barrier();

    int request_batch_size = size_2;
    int max_input_len      = size_1 / size_2;

    ft::mpi::bcast(v_input_ids.data(), size_1, ft::mpi::MPI_TYPE_INT, 0, ft::mpi::COMM_WORLD);
    ft::mpi::bcast(v_input_lengths.data(), size_2, ft::mpi::MPI_TYPE_INT, 0, ft::mpi::COMM_WORLD);

    std::vector<std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>> request_list;
    for (int device_id = 0; device_id < gpu_count; device_id++) {
        ft::check_cuda_error(cudaSetDevice(device_id));

        int* d_input_ids;
        int* d_input_lengths;

        if (max_input_len == 0) {
            // unconditional case, no input ids, so do nothing.
            d_input_ids     = nullptr;
            d_input_lengths = nullptr;
            max_input_len   = 0;
        }
        else {
            // conditional case.
            ft::deviceMalloc(&d_input_ids, size_1, false);
            ft::deviceMalloc(&d_input_lengths, size_2, false);
            ft::cudaH2Dcpy(d_input_ids, v_input_ids.data(), size_1);
            ft::cudaH2Dcpy(d_input_lengths, v_input_lengths.data(), size_2);
        }

        uint32_t* request_output_len_ptr = (uint32_t*)malloc(request_batch_size * sizeof(uint32_t));
        for (int i = 0; i < request_batch_size; i++) {
            request_output_len_ptr[i] = request_output_len;
        }

        request_list.push_back(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>(
            new std::unordered_map<std::string, triton::Tensor>{
                {"input_ids",
                 triton::Tensor{triton::MEMORY_GPU,
                                triton::TYPE_INT32,
                                std::vector<size_t>{(size_t)request_batch_size, (size_t)max_input_len},
                                d_input_ids}},
                {"input_lengths",
                 triton::Tensor{triton::MEMORY_GPU,
                                triton::TYPE_INT32,
                                std::vector<size_t>{(size_t)request_batch_size},
                                d_input_lengths}},
                {"request_output_len",
                 triton::Tensor{triton::MEMORY_CPU,
                                triton::TYPE_INT32,
                                std::vector<size_t>{(size_t)request_batch_size},
                                request_output_len_ptr}}}));
        int* beam_width_ptr = new int(beam_width);
        pointer_record->push_back(beam_width_ptr);
        request_list[device_id]->insert(
            {"beam_width",
             triton::Tensor{triton::MEMORY_CPU, triton::TYPE_INT32, std::vector<size_t>{1}, beam_width_ptr}});
        if (beam_width > 1) {
            float* beam_search_diversity_rate_ptr = new float(beam_search_diversity_rate);
            pointer_record->push_back(beam_search_diversity_rate_ptr);
            request_list[device_id]->insert(
                {"beam_search_diversity_rate",
                 triton::Tensor{
                     triton::MEMORY_CPU, triton::TYPE_FP32, std::vector<size_t>{1}, beam_search_diversity_rate_ptr}});
        }
        else {
            if (runtime_top_p != 0.0f) {
                float* runtime_top_p_ptr = new float(runtime_top_p);
                pointer_record->push_back(runtime_top_p_ptr);
                request_list[device_id]->insert(
                    {"runtime_top_p",
                     triton::Tensor{triton::MEMORY_CPU, triton::TYPE_FP32, std::vector<size_t>{1}, runtime_top_p_ptr}});
            }
            if (runtime_top_k != 0) {
                uint* runtime_top_k_ptr = new uint(runtime_top_k);
                pointer_record->push_back(runtime_top_k_ptr);
                request_list[device_id]->insert(
                    {"runtime_top_k",
                     triton::Tensor{
                         triton::MEMORY_CPU, triton::TYPE_UINT32, std::vector<size_t>{1}, runtime_top_k_ptr}});
            }
        }
        float* temperature_ptr = new float(temperature);
        pointer_record->push_back(temperature_ptr);
        request_list[device_id]->insert(
            {"temperature",
             triton::Tensor{triton::MEMORY_CPU, triton::TYPE_FP32, std::vector<size_t>{1}, temperature_ptr}});
        float* len_penalty_ptr = new float(len_penalty);
        pointer_record->push_back(len_penalty_ptr);
        request_list[device_id]->insert(
            {"len_penalty",
             triton::Tensor{triton::MEMORY_CPU, triton::TYPE_FP32, std::vector<size_t>{1}, len_penalty_ptr}});
        float* repetition_penalty_ptr = new float(repetition_penalty);
        pointer_record->push_back(repetition_penalty_ptr);
        request_list[device_id]->insert(
            {"repetition_penalty",
             triton::Tensor{triton::MEMORY_CPU, triton::TYPE_FP32, std::vector<size_t>{1}, repetition_penalty_ptr}});
        if (repetition_penalty != 1.0f) {
            float* repetition_penalty_ptr = new float(repetition_penalty);
            pointer_record->push_back(repetition_penalty_ptr);
            request_list[device_id]->insert(
                {"repetition_penalty",
                 triton::Tensor{
                     triton::MEMORY_CPU, triton::TYPE_FP32, std::vector<size_t>{1}, repetition_penalty_ptr}});
        }
        if (presence_penalty != 0.0f) {
            float* presence_penalty_ptr = new float(presence_penalty);
            pointer_record->push_back(presence_penalty_ptr);
            request_list[device_id]->insert(
                {"presence_penalty",
                 triton::Tensor{triton::MEMORY_CPU, triton::TYPE_FP32, std::vector<size_t>{1}, presence_penalty_ptr}});
        }
        int* min_length_ptr = new int(min_length);
        pointer_record->push_back(min_length_ptr);
        request_list[device_id]->insert(
            {"min_length",
             triton::Tensor{triton::MEMORY_CPU, triton::TYPE_INT32, std::vector<size_t>{1}, min_length_ptr}});
        unsigned long long int* random_seed_ptr = new unsigned long long int(random_seed);
        pointer_record->push_back(random_seed_ptr);
        request_list[device_id]->insert(
            {"random_seed",
             triton::Tensor{triton::MEMORY_CPU, triton::TYPE_UINT64, std::vector<size_t>{1}, random_seed_ptr}});
        bool* is_return_log_probs_ptr = new bool(is_return_log_probs);
        pointer_record->push_back(is_return_log_probs_ptr);
        request_list[device_id]->insert(
            {"is_return_log_probs",
             triton::Tensor{triton::MEMORY_CPU, triton::TYPE_BOOL, std::vector<size_t>{1}, is_return_log_probs_ptr}});
        bool* is_return_context_cum_log_probs_ptr = new bool(is_return_context_cum_log_probs);
        pointer_record->push_back(is_return_context_cum_log_probs_ptr);
        request_list[device_id]->insert(
            {"is_return_context_cum_log_probs",
             triton::Tensor{
                 triton::MEMORY_CPU, triton::TYPE_BOOL, std::vector<size_t>{1}, is_return_context_cum_log_probs_ptr}});
        bool* is_return_context_embeddings_ptr = new bool(is_return_context_embeddings);
        pointer_record->push_back(is_return_context_embeddings_ptr);
        request_list[device_id]->insert(
            {"is_return_context_embeddings",
             triton::Tensor{
                 triton::MEMORY_CPU, triton::TYPE_BOOL, std::vector<size_t>{1}, is_return_context_embeddings_ptr}});

        pointer_record->push_back(d_input_ids);
        pointer_record->push_back(d_input_lengths);
        pointer_record->push_back(request_output_len_ptr);
    }

    return request_list;
}

std::vector<std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>>
prepareRequest(std::string ini_name, const int node_id, const int gpu_count, std::vector<void*>* pointer_record)
{
    INIReader reader = INIReader(ini_name);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << ini_name << "'\n";
        ft::FT_CHECK(false);
    }

    const size_t request_batch_size = reader.GetInteger("request", "request_batch_size");
    const size_t beam_width         = reader.GetInteger("ft_instance_hyperparameter", "beam_width");
    const size_t request_output_len = reader.GetInteger("request", "request_output_len");
    const int    top_k              = reader.GetInteger("ft_instance_hyperparameter", "top_k");
    const float  top_p              = reader.GetFloat("ft_instance_hyperparameter", "top_p");
    const float  temperature        = reader.GetFloat("ft_instance_hyperparameter", "temperature");
    const float  repetition_penalty = reader.GetFloat("ft_instance_hyperparameter", "repetition_penalty", 1.0f);
    const float  presence_penalty   = reader.GetFloat("ft_instance_hyperparameter", "presence_penalty", 0.0f);
    const int    min_length         = reader.GetInteger("ft_instance_hyperparameter", "min_length", 0);
    const float  len_penalty        = reader.GetFloat("ft_instance_hyperparameter", "len_penalty");
    const float  beam_search_diversity_rate =
        reader.GetFloat("ft_instance_hyperparameter", "beam_search_diversity_rate");
    const unsigned long long int random_seed         = 0;
    const bool                   is_return_log_probs = reader.GetBoolean("request", "return_log_probs", false);
    // Whether to include input contexts in computing the cumulative log probabilities.
    const bool is_return_context_cum_log_probs = reader.GetBoolean("request", "context_log_probs", false);
    const bool is_return_context_embeddings    = reader.GetBoolean("request", "context_embeddings", false);
    if (!is_return_log_probs && !is_return_context_cum_log_probs) {
        FT_LOG_WARNING("context_log_probs will be ignored since return_log_probs is disabled.");
    }

    const int        end_id = 50256;
    std::vector<int> v_start_ids;
    std::vector<int> v_start_lengths;

    size_t max_input_len = 0;
    ft::read_start_ids(request_batch_size,
                       &v_start_lengths,
                       &v_start_ids,
                       max_input_len,
                       end_id,
                       1,
                       "../examples/cpp/multi_gpu_gpt/start_ids.csv");

    auto request_list = broadcastRequest(v_start_ids,
                                         v_start_lengths,
                                         node_id,
                                         gpu_count,
                                         beam_width,
                                         request_output_len,
                                         beam_search_diversity_rate,
                                         top_k,
                                         top_p,
                                         temperature,
                                         len_penalty,
                                         repetition_penalty,
                                         presence_penalty,
                                         min_length,
                                         random_seed,
                                         is_return_log_probs,
                                         is_return_context_cum_log_probs,
                                         is_return_context_embeddings,
                                         pointer_record);
    return request_list;
}

int threadCreateModelInstances(std::shared_ptr<AbstractTransformerModel>                         model,
                               std::vector<std::unique_ptr<AbstractTransformerModelInstance>>*   model_instances,
                               const int                                                         device_id,
                               const int                                                         rank,
                               std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_comms,
                               std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comm = nullptr)
{
    FT_LOG_INFO("rank = %d", rank);
    ft::check_cuda_error(cudaSetDevice(device_id));
    cudaStream_t stream;
    ft::check_cuda_error(cudaStreamCreate(&stream));
    model->createSharedWeights(device_id, rank);
    auto model_instance = model->createModelInstance(device_id, rank, stream, nccl_comms, custom_all_reduce_comm);
    model_instances->at(device_id) = std::move(model_instance);
    FT_LOG_INFO("model instance %d is created", device_id);
    ft::print_mem_usage();
    return 0;
}

int threadForward(std::unique_ptr<AbstractTransformerModelInstance>*                model_instance,
                  std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>  request,
                  std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>* output_tensors,
                  const int                                                         device_id)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    *output_tensors = (*model_instance)->forward(request);
    return 0;
}

int main(int argc, char* argv[])
{
    /*
        Prepare the nccl ids, node id, device id and world size
        by MPI or triton
    */

    ft::mpi::initialize(&argc, &argv);
    int node_id  = ft::mpi::getCommWorldRank();
    int node_num = ft::mpi::getCommWorldSize();

    // Note: Only supports that all nodes have same gpu count
    const int   gpu_count  = ft::getDeviceCount();
    const int   world_size = node_num * gpu_count;
    std::string ini_name   = argc >= 2 ? std::string(argv[1]) : "../examples/cpp/multi_gpu_gpt/gpt_config.ini";

    // step 1: Create model
    std::shared_ptr<AbstractTransformerModel> model              = AbstractTransformerModel::createGptModel(ini_name);
    int                                       tensor_para_size   = model->getTensorParaSize();
    int                                       pipeline_para_size = model->getPipelineParaSize();
    FT_CHECK_WITH_INFO(world_size == (tensor_para_size * pipeline_para_size),
                       ft::fmtstr("World Size (%d) != Tensor Parallel Size (%d) * Pipeline Parallel Size (%d).",
                                  world_size,
                                  tensor_para_size,
                                  pipeline_para_size));

    std::cout << model->toString();

    // step 2: Initialize the NCCL
    std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_comms = model->createNcclParams(node_id);
    cudaDeviceSynchronize();

    // Optional Step: create custom all reduce comm
    std::vector<std::shared_ptr<ft::AbstractCustomComm>> custom_all_reduce_comms;
    model->createCustomComms(&custom_all_reduce_comms, world_size);

    // step 3: Create model instances
    std::vector<std::unique_ptr<AbstractTransformerModelInstance>> model_instances((size_t)gpu_count);
    std::vector<std::thread>                                       threads;

    threads.clear();

    for (int device_id = 0; device_id < gpu_count; device_id++) {
        const int rank = node_id * gpu_count + device_id;
        threads.push_back(std::thread(threadCreateModelInstances,
                                      model,
                                      &model_instances,
                                      device_id,
                                      rank,
                                      nccl_comms,
                                      custom_all_reduce_comms[rank]));
    }
    for (auto& t : threads) {
        t.join();
    }

    // step 4: prepare request
    std::vector<void*> pointer_record;  // Used to prevent the pointers are release after leaving functions
    std::vector<std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>> request_list =
        prepareRequest(ini_name, node_id, gpu_count, &pointer_record);
    FT_LOG_INFO("request is created");

    // step 5: Forward
    std::vector<std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>> output_tensors_lists(
        (size_t)gpu_count);
    for (int i = 0; i < 2; i++) {
        threads.clear();
        for (int device_id = 0; device_id < gpu_count; device_id++) {
            threads.push_back(std::thread(threadForward,
                                          &model_instances[device_id],
                                          request_list[device_id],
                                          &output_tensors_lists[device_id],
                                          device_id));
        }
        for (auto& t : threads) {
            t.join();
        }
    }
    FT_LOG_INFO("forward is completed.");

    const int* d_output_ids = (const int*)output_tensors_lists[0].get()->at("output_ids").data;
    const int  batch_size   = output_tensors_lists[0].get()->at("output_ids").shape[0];
    const int  beam_width   = output_tensors_lists[0].get()->at("output_ids").shape[1];
    const int  seq_len      = output_tensors_lists[0].get()->at("output_ids").shape[2];
    // step 6: check results
    if (node_id == 0) {

        std::string fName   = "out";
        auto        outFile = std::ofstream(fName, std::ios::out);
        if (!outFile.is_open()) {
            FT_LOG_WARNING("Cannot write results into output file %s", fName.c_str());
        }
        else {
            size_t outCount = batch_size * beam_width * seq_len;
            int*   hBuf     = new int[outCount];
            ft::cudaD2Hcpy(hBuf, d_output_ids, outCount);

            {
                std::cout << "Writing " << outCount << " elements\n";
                int zeroCount = 0;
                for (size_t i = 0; i < outCount; i++) {
                    if (hBuf[i] == int(0)) {
                        zeroCount++;
                    }
                    outFile << hBuf[i] << " ";
                    if ((i + 1) % (seq_len) == 0) {
                        outFile << std::endl;
                    }

                    if (i < 10) {
                        printf("%5d ", hBuf[i]);
                    }
                    if ((i + 1) % (seq_len) == 0 && i < 10) {
                        std::cout << std::endl;
                    }
                }
                std::cout << std::endl << "zeroCount = " << zeroCount << std::endl;
            }
            delete[] hBuf;
        }
    }

    // test time
    struct timeval start, end;
    ft::mpi::barrier();
    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);

    const int ite = 1;
    for (int i = 0; i < ite; i++) {
        threads.clear();
        for (int device_id = 0; device_id < gpu_count; device_id++) {
            threads.push_back(std::thread(threadForward,
                                          &model_instances[device_id],
                                          request_list[device_id],
                                          &output_tensors_lists[device_id],
                                          device_id));
        }
        for (auto& t : threads) {
            t.join();
        }
    }

    cudaDeviceSynchronize();
    ft::mpi::barrier();

    gettimeofday(&end, NULL);

    FT_LOG_INFO("batch_size %d beam_width %d seq_len %d"
                " FT-CPP-GPT-Triton-time %.2f ms",
                batch_size,
                beam_width,
                seq_len,
                ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / ite);

    ft::mpi::finalize();
    return 0;
}
