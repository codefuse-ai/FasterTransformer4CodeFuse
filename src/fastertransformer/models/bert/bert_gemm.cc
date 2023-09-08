/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/utils/gemm_test/encoder_gemm_func.h"
#include "src/fastertransformer/utils/gemm_test/encoder_igemm_func.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace ft = fastertransformer;

int main(int argc, char* argv[])
{
    if (argc < 7 || argc > 9) {
        FT_LOG_ERROR("./bin/bert_gemm batch_size \\ \n"
                     "                seq_len \\ \n"
                     "                head_number \\ \n"
                     "                size_per_head \\ \n"
                     "                data_type \\ \n"
                     "                int8_mode \\ \n"
                     "                tensor_para_size \\\n"
                     "                is_append (append new config into exist gemm_config.ini or not)");
        FT_LOG_ERROR("e.g. ./bin/bert_gemm 1 32 12 64 0 0 1 1");
        return 0;
    }

    const int                batch_size    = atoi(argv[1]);
    const int                seq_len       = atoi(argv[2]);
    const int                head_num      = atoi(argv[3]);
    const int                size_per_head = atoi(argv[4]);
    const ft::CublasDataType data_type     = static_cast<ft::CublasDataType>(atoi(argv[5]));  // 0 FP32, 1 FP16, 2 BF 16
    const int                int8_mode     = atoi(argv[6]);
    const int                tensor_para_size = argc < 8 ? 1 : atoi(argv[7]);
    const bool               is_append        = argc < 9 ? false : (bool)(atoi(argv[8]));

    const int inter_size = 4 * head_num * size_per_head;
    FT_CHECK_WITH_INFO(
        head_num % tensor_para_size == 0,
        ft::fmtstr("[ERROR] head_num (%d) %% tensor_para_size (%d) != 0", head_num, tensor_para_size));
    FT_LOG_INFO("arguments:");
    FT_LOG_INFO("  batch_size: %d", batch_size);
    FT_LOG_INFO("  head_num: %d", head_num);
    FT_LOG_INFO("  size_per_head: %d", size_per_head);
    FT_LOG_INFO("  data_type: %d", data_type);
    FT_LOG_INFO("  int8_mode: %d", int8_mode);
    FT_LOG_INFO("  tensor_para_size: %d", tensor_para_size);
    FT_LOG_INFO("  is_append: %d", (int)is_append);
    std::cout << std::endl;

    void*  gemm_test_buf;
    size_t buf_size_in_byte =
        ft::calGemmTestBufSizeInByte(batch_size, seq_len, head_num, size_per_head, inter_size, 0, int8_mode, data_type);
    size_t total, free;
    ft::check_cuda_error(cudaMemGetInfo(&free, &total));
    if (free < buf_size_in_byte + 10 * 1024 * 1024) {
        FT_LOG_ERROR(" There is no enough device memory for gemm test!\n"
                     " %ld Bytes is needed, but only %ld Bytes is free.",
                     buf_size_in_byte,
                     free);
        gemm_test_buf = NULL;
        return -1;
    }
    else {
        ft::deviceMalloc(reinterpret_cast<char**>(&gemm_test_buf), buf_size_in_byte, false);
    }

    if (int8_mode != 0) {
        ft::generate_encoder_igemm_config(batch_size, seq_len, head_num, size_per_head, gemm_test_buf, is_append);
    }
    else if (data_type == ft::FLOAT_DATATYPE) {
        ft::generate_encoder_gemm_config<float>(
            batch_size, seq_len, head_num, size_per_head, gemm_test_buf, is_append, tensor_para_size);
    }
    else if (data_type == ft::HALF_DATATYPE) {
        ft::generate_encoder_gemm_config<half>(
            batch_size, seq_len, head_num, size_per_head, gemm_test_buf, is_append, tensor_para_size);
    }
#ifdef ENABLE_BF16
    else if (data_type == ft::BFLOAT16_DATATYPE) {
        ft::generate_encoder_gemm_config<__nv_bfloat16>(
            batch_size, seq_len, head_num, size_per_head, gemm_test_buf, is_append, tensor_para_size);
    }
#endif
    else {
        FT_LOG_ERROR("data type only supports fp32(0), fp16(1), bf16(2). \n");
        return -1;
    }

    ft::check_cuda_error(cudaFree(gemm_test_buf));
    return 0;
}
