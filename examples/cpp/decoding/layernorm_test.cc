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

#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include <cuda_profiler_api.h>
#include <sys/time.h>

using namespace fastertransformer;

template<typename T>
int checkNonZero(T* A, int size)
{
    T* h_A = (T*)malloc(sizeof(T) * size);
    cudaMemcpy(h_A, A, sizeof(T) * size, cudaMemcpyDeviceToHost);
    int noneZeroNum = 0;
    for (int ii = 0; ii < size; ii++) {
        if (fabs(float(h_A[ii]) - 0.0f) > 0.0001f) {
            noneZeroNum += 1;
        }
    }
    free(h_A);
    return noneZeroNum;
}

template<typename TA, typename TB>
void checkMat(TA* A, TB* B, int size, const char* mark, float threshold)
{
    float max_diff = -10000.0f;
    float max_diff_a, max_diff_b;
    TA*   matA       = (TA*)malloc(sizeof(TA) * size);
    TB*   matB       = (TB*)malloc(sizeof(TB) * size);
    int   not_passed = 0;
    cudaMemcpy(matA, A, sizeof(TA) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(matB, B, sizeof(TB) * size, cudaMemcpyDeviceToHost);
    float A_nonZero_ratio = float(checkNonZero(A, size)) / float(size);
    float B_nonZero_ratio = float(checkNonZero(B, size)) / float(size);
    if (A_nonZero_ratio < 0.1 || B_nonZero_ratio < 0.1) {
        printf("[%s] nonZero ratio [%f] [%f]\n", mark, A_nonZero_ratio, B_nonZero_ratio);
    }
    for (int jjj = 0; jjj < size; jjj++) {
        float diff = fabs(float(matA[jjj]) - float(matB[jjj]));
        if (diff > max_diff) {
            max_diff   = diff;
            max_diff_a = float(matA[jjj]);
            max_diff_b = float(matB[jjj]);
        }
        if (fabs(float(matA[jjj]) - float(matB[jjj])) > threshold) {
            not_passed += 1;
            if (not_passed < 1000) {
                printf("%d %f %f %f\n", jjj, float(matA[jjj]), float(matB[jjj]), float(matA[jjj]) - float(matB[jjj]));
            }
        }
    }
    FT_LOG_INFO("[%s] max diff : %f ; a : %f ; b : %f", mark, max_diff, max_diff_a, max_diff_b);
    if (not_passed != 0) {
        FT_LOG_ERROR("[%s] different elements : %d ", mark, not_passed);
        FT_CHECK(false);
    }
    else {
        FT_LOG_INFO("[%s] check pass!", mark);
    }
    free(matA);
    free(matB);
}

template<typename T>
void layernorm_test(const int m, const int n);

template<typename T>
void add_bias_residual_layernorm_test(const int m, const int n);

int main(int argc, char** argv)
{
    if (argc != 4) {
        printf("[ERROR] layernorm_test max_m max_n data_type\n");
        printf("e.g., ./bin/layernorm_test 1 1024 1\n");
        return 0;
    }

    int                  max_m     = atoi(argv[1]);
    int                  max_n     = atoi(argv[2]);
    const FtCudaDataType data_type = static_cast<FtCudaDataType>(atoi(argv[3]));  // 0 FP32, 1 FP16, 2 BF 16

    for (int m = 1; m <= max_m; m *= 2) {
        for (int n = 128; n <= max_n; n *= 2) {
            if (data_type == FP16) {
                add_bias_residual_layernorm_test<half>(m, n);
            }
#ifdef ENABLE_BF16
            else if (data_type == BF16) {
                add_bias_residual_layernorm_test<__nv_bfloat16>(m, n);
            }
#endif
            else if (data_type == FP32) {
                add_bias_residual_layernorm_test<float>(m, n);
            }
            else {
                FT_LOG_ERROR("data_type should be fp32, fp16 or bf16!");
                exit(-1);
            }
        }
    }
    return 0;
}

template<typename T>
void layernorm_test(const int m, const int n)
{
    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));
    printf("Device %s\n", prop.name);

    const float layernorm_eps = 1e-4f;
    T *         input, *output_opt, *output_baseline, *gamma, *beta;
    deviceMalloc(&input, m * n);
    deviceMalloc(&output_baseline, m * n);
    deviceMalloc(&output_opt, m * n);
    deviceMalloc(&gamma, n);
    deviceMalloc(&beta, n);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    const int ite = 5000;

    // warmup
    for (int i = 0; i < 1000; i++) {
        invokeGeneralLayerNorm<T>(output_baseline, input, gamma, beta, layernorm_eps, m, n, (float*)nullptr, 0, stream);
        invokeGeneralLayerNorm<T>(
            output_opt, input, gamma, beta, layernorm_eps, m, n, (float*)nullptr, 0, stream, true);
    }

    struct timeval start, end;
    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);
    for (int i = 0; i < ite; i++) {
        invokeGeneralLayerNorm<T>(output_baseline, input, gamma, beta, layernorm_eps, m, n, (float*)nullptr, 0, stream);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    float baseline_time = ((end.tv_sec - start.tv_sec) * 1000000. + (end.tv_usec - start.tv_usec) * 1.) / ite;

    struct timeval start_2, end_2;
    cudaDeviceSynchronize();
    gettimeofday(&start_2, NULL);
    for (int i = 0; i < ite; i++) {
        invokeGeneralLayerNorm<T>(
            output_opt, input, gamma, beta, layernorm_eps, m, n, (float*)nullptr, 0, stream, true);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end_2, NULL);
    float opt_time = ((end_2.tv_sec - start_2.tv_sec) * 1000000. + (end_2.tv_usec - start_2.tv_usec) * 1.) / ite;

    print_abs_mean(output_baseline, m * n, stream, "output_baseline");
    print_abs_mean(output_opt, m * n, stream, "output_opt");

    printf("[INFO] baseline time: %f us\n", baseline_time);
    printf("[INFO] opt time: %f us\n", opt_time);
    printf("[INFO] m %d, n %d, speedup: %f \n", m, n, baseline_time / opt_time);
    return;
}

template<typename T>
void add_bias_residual_layernorm_test(const int m, const int n)
{
    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));
    printf("Device %s\n", prop.name);

    const float layernorm_eps = 1e-4f;
    int         opt_version   = 2;
    T *         input, *output_opt, *output_baseline, *gamma, *beta, *bias;
    T *         normed_output_opt, *normed_output_baseline;
    deviceMalloc(&input, m * n);
    deviceMalloc(&output_baseline, m * n);
    deviceMalloc(&output_opt, m * n);
    cudaD2Dcpy(output_opt, output_baseline, m * n);
    deviceMalloc(&normed_output_opt, m * n);
    deviceMalloc(&normed_output_baseline, m * n);
    deviceMalloc(&gamma, n);
    deviceMalloc(&beta, n);
    deviceMalloc(&bias, n);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    const int warmup_ite = 1000;  // 1000;
    const int ite        = 5000;  // 5000;

    // verify correctness
    invokeGeneralAddBiasResidualPreLayerNorm(output_baseline,
                                             normed_output_baseline,
                                             output_baseline,
                                             input,
                                             gamma,
                                             beta,
                                             bias,
                                             layernorm_eps,
                                             m,
                                             n,
                                             (float*)nullptr,
                                             (float*)nullptr,
                                             (float*)nullptr,
                                             (float*)nullptr,
                                             0,
                                             stream,
                                             0);

    invokeGeneralAddBiasResidualPreLayerNorm(output_opt,
                                             normed_output_opt,
                                             output_opt,
                                             input,
                                             gamma,
                                             beta,
                                             bias,
                                             layernorm_eps,
                                             m,
                                             n,
                                             (float*)nullptr,
                                             (float*)nullptr,
                                             (float*)nullptr,
                                             (float*)nullptr,
                                             0,
                                             stream,
                                             opt_version);
    float threshold = 0.0f;
    if (std::is_same<T, float>::value) {
        threshold = 1e-6f;
    }
    else if (std::is_same<T, half>::value) {
        threshold = 1e-3;
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        threshold = 5e-2;
    }
#endif
    else {
        FT_LOG_ERROR("data_type should be fp32, fp16 or bf16!");
        exit(-1);
    }
    checkMat(output_baseline, output_opt, m * n, "output_baseline vs output_opt", threshold);
    checkMat(
        normed_output_baseline, normed_output_opt, m * n, "normed_output_baseline vs normed_output_opt", threshold);

    // warmup
    for (int i = 0; i < warmup_ite; i++) {
        invokeGeneralAddBiasResidualPreLayerNorm(output_baseline,
                                                 normed_output_baseline,
                                                 output_baseline,
                                                 input,
                                                 gamma,
                                                 beta,
                                                 bias,
                                                 layernorm_eps,
                                                 m,
                                                 n,
                                                 (float*)nullptr,
                                                 (float*)nullptr,
                                                 (float*)nullptr,
                                                 (float*)nullptr,
                                                 0,
                                                 stream,
                                                 0);
        invokeGeneralAddBiasResidualPreLayerNorm(output_opt,
                                                 normed_output_opt,
                                                 output_opt,
                                                 input,
                                                 gamma,
                                                 beta,
                                                 bias,
                                                 layernorm_eps,
                                                 m,
                                                 n,
                                                 (float*)nullptr,
                                                 (float*)nullptr,
                                                 (float*)nullptr,
                                                 (float*)nullptr,
                                                 0,
                                                 stream,
                                                 opt_version);
    }

    struct timeval start, end;
    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);
    for (int i = 0; i < ite; i++) {
        invokeGeneralAddBiasResidualPreLayerNorm(output_baseline,
                                                 normed_output_baseline,
                                                 output_baseline,
                                                 input,
                                                 gamma,
                                                 beta,
                                                 bias,
                                                 layernorm_eps,
                                                 m,
                                                 n,
                                                 (float*)nullptr,
                                                 (float*)nullptr,
                                                 (float*)nullptr,
                                                 (float*)nullptr,
                                                 0,
                                                 stream,
                                                 0);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    float baseline_time = ((end.tv_sec - start.tv_sec) * 1000000. + (end.tv_usec - start.tv_usec) * 1.) / ite;

    struct timeval start_2, end_2;
    cudaDeviceSynchronize();
    gettimeofday(&start_2, NULL);
    for (int i = 0; i < ite; i++) {
        invokeGeneralAddBiasResidualPreLayerNorm(output_opt,
                                                 normed_output_opt,
                                                 output_opt,
                                                 input,
                                                 gamma,
                                                 beta,
                                                 bias,
                                                 layernorm_eps,
                                                 m,
                                                 n,
                                                 (float*)nullptr,
                                                 (float*)nullptr,
                                                 (float*)nullptr,
                                                 (float*)nullptr,
                                                 0,
                                                 stream,
                                                 opt_version);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end_2, NULL);
    float opt_time = ((end_2.tv_sec - start_2.tv_sec) * 1000000. + (end_2.tv_usec - start_2.tv_usec) * 1.) / ite;
    sync_check_cuda_error();

    printf("[INFO] baseline time: %f us\n", baseline_time);
    printf("[INFO] opt time: %f us\n", opt_time);
    printf("[INFO] m %3d, n %5d, speedup: %f \n", m, n, baseline_time / opt_time);
    return;
}
