# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import argparse
import re
import numpy as np
import torch

ACTIVATION_AMAX_NUM = 72
INT8O_GEMM_NUM = 8
TRT_FUSED_MHA_AMAX_NUM = 3
SCALE_RESERVE_NUM = 21

def checkpoint_quantization(init_dict, ths_path='../../../build/lib/libth_transformer.so', verbose=True):
    print("Quantizing checkpoint ...")
    torch.classes.load_library(ths_path)
    weight_quantize = torch.ops.fastertransformer.vit_weight_quantize

    def init_graph():
        layer_num = 0
        regex = re.compile('layer.\d+')
        amaxTotalNum = 0
        for name, tensor_value in init_dict.items():
            if "ffn.fc1" in name and amaxTotalNum == 0:
                amaxTotalNum = ACTIVATION_AMAX_NUM + 9 * tensor_value.size(1) + INT8O_GEMM_NUM + TRT_FUSED_MHA_AMAX_NUM + SCALE_RESERVE_NUM
                if verbose:
                    print("amaxTotalNum", amaxTotalNum)
                    print("Hidden size:", tensor_value.size(1))
            tmp = regex.findall(name)
            if len(tmp) < 1:
                continue
            num_tmp = int(tmp[0].replace("layer.", ""))
            if layer_num < num_tmp:
                layer_num = num_tmp
        layer_num = layer_num + 1
        #add new var for amax
        for i in range(layer_num):
            init_dict["transformer.encoder.layer.{}.amaxList".format(i)] = torch.zeros((amaxTotalNum,), dtype=torch.float32)
        return layer_num, amaxTotalNum
    layer_num, amaxTotalNum = init_graph()

    kernel_name_list = ["attn.query",
                        "attn.key",
                        "attn.value",
                        "attn.out",
                        "ffn.fc1",
                        "ffn.fc2"]

    amax_name_list =   ["attn.query._input_quantizer",
                        "attn.query._aftergemm_quantizer",
                        "attn.matmul_q_input_quantizer",
                        "attn.key._aftergemm_quantizer",
                        "attn.matmul_k_input_quantizer",
                        "attn.value._aftergemm_quantizer",
                        "attn.matmul_v_input_quantizer",
                        "attn.softmax_input_quantizer",
                        "attn.matmul_a_input_quantizer",
                        "attn.out._input_quantizer",
                        "attn.out._aftergemm_quantizer",
                        "ffn.fc1._input_quantizer",
                        "ffn.fc1._aftergemm_quantizer",
                        "ffn.fc2._input_quantizer",
                        "ffn.fc2._aftergemm_quantizer",
                        "special_F2Bias_scale",
                        ]

    int8O_gemm_weight_amax_list = [0 for i in range(INT8O_GEMM_NUM)]
    int8O_gemm_weight_list = ["attn.query", 
                              "attn.key", 
                              "attn.value", 
                              "attn.matmul_k_input_quantizer",
                              "attn.matmul_v_input_quantizer", 
                              "attn.out", 
                              "ffn.fc1", 
                              "ffn.fc2"]

    int8O_gemm_input_amax_list = [0 for i in range(INT8O_GEMM_NUM)]
    int8O_gemm_input_list = ["attn.query._input_quantizer",
                             "attn.key._input_quantizer",
                             "attn.value._input_quantizer", 
                             "attn.matmul_q_input_quantizer", 
                             "attn.matmul_a_input_quantizer",
                             "attn.out._input_quantizer",
                             "ffn.fc1._input_quantizer", 
                             "ffn.fc2._input_quantizer"]
    
    int8O_gemm_output_amax_list = [0 for i in range(INT8O_GEMM_NUM)]
    int8O_gemm_output_list = ["attn.query._aftergemm_quantizer",
                              "attn.key._aftergemm_quantizer",
                              "attn.value._aftergemm_quantizer",
                              "attn.softmax_input_quantizer", 
                              "attn.out._input_quantizer",
                              "attn.out._aftergemm_quantizer",
                              "ffn.fc1._aftergemm_quantizer", 
                              "ffn.fc2._aftergemm_quantizer"]

    same_value_tuple_list = [("attn.query._input_quantizer",
                              "attn.key._input_quantizer",
                              "attn.value._input_quantizer")]

    factor = 1000000.0
    for i in range(layer_num):
        amaxList = np.zeros([amaxTotalNum]).astype(np.float32)
        amax_id = 0
        # verify some quantizers have same value. input_quantizer is per-tensor quantization
        for same_value_tuple in same_value_tuple_list:
            tmp_v = init_dict["transformer.encoder.layer.{}.{}._amax".format(i, same_value_tuple[0])].numpy()
            for same_value_name in same_value_tuple:
                tmp_v_2 = init_dict["transformer.encoder.layer.{}.{}._amax".format(i, same_value_name)].numpy()
                assert(np.allclose(tmp_v, tmp_v_2))

        for amax_name in amax_name_list:
            if amax_name == "special_F2Bias_scale":
                if i != layer_num - 1:
                    quant_max = init_dict["transformer.encoder.layer.{}.{}._amax".format(i+1, amax_name_list[0])].item()
                    amax = abs(quant_max)
                else:
                    #not used, placeholder
                    amax = 1.0
                amaxList[amax_id] = amax
                amax_id += 1
                amaxList[amax_id] = amax/127.0
                amax_id += 1
                amaxList[amax_id] = amax/127.0/127.0
                amax_id += 1
                amaxList[amax_id] = 127.0/amax
                amax_id += 1
                continue

            quant_max = init_dict["transformer.encoder.layer.{}.{}._amax".format(i, amax_name)].item()
            amax = abs(quant_max)#round(abs(quant_max)*factor)/factor
            if amax_name in int8O_gemm_input_list:
                int8O_gemm_input_amax_list[int8O_gemm_input_list.index(amax_name)] = amax
                if amax_name == "attn.query._input_quantizer":
                    int8O_gemm_input_amax_list[int8O_gemm_input_list.index("attn.key._input_quantizer")] = amax
                    int8O_gemm_input_amax_list[int8O_gemm_input_list.index("attn.value._input_quantizer")] = amax
            if amax_name in int8O_gemm_output_list:
                int8O_gemm_output_amax_list[int8O_gemm_output_list.index(amax_name)] = amax
            if amax_name in int8O_gemm_weight_list:
                int8O_gemm_weight_amax_list[int8O_gemm_weight_list.index(amax_name)] = amax      
            amaxList[amax_id] = amax
            amax_id += 1
            amaxList[amax_id] = amax/127.0
            amax_id += 1
            amaxList[amax_id] = amax/127.0/127.0
            amax_id += 1
            amaxList[amax_id] = 127.0/amax
            amax_id += 1
            if verbose:
                print(i, amax_name)
                print('quant_max:', quant_max)
                print('amax:', amax)
        if verbose:
            print("done process layer_{} activation amax".format(i))

        #kernel amax starts from ACTIVATION_AMAX_NUM
        assert amax_id == 64
        amax_id = ACTIVATION_AMAX_NUM
        for kernel_id, kernel_name in enumerate(kernel_name_list):
            kernel = init_dict["transformer.encoder.layer.{}.{}.weight".format(i, kernel_name)].transpose(-1, -2).contiguous()
            quant_max2 = init_dict["transformer.encoder.layer.{}.{}._weight_quantizer._amax".format(i, kernel_name)]
            amax2 = abs(quant_max2)
            if (amax2.dim() == 0):
                quant_max_processed = torch.full((kernel.size(1),), amax2.item(), dtype=amax2.dtype, device=amax2.device)
            else:
                quant_max_processed = amax2.view(-1)
            kernel_processed = weight_quantize(kernel.cuda(), quant_max_processed.cuda())
            init_dict["transformer.encoder.layer.{}.{}.weight".format(i, kernel_name)] = kernel_processed
            if kernel_name in int8O_gemm_weight_list:
                int8O_gemm_weight_amax_list[int8O_gemm_weight_list.index(kernel_name)] = quant_max_processed[0]
            for e in quant_max_processed:
                amaxList[amax_id] = e
                amax_id += 1
            # if verbose:
            #     print(i, kernel_name)
            #     print('kernel:', kernel)
            #     print('quant_max2:', quant_max2)
            #     print('quant_max_processed_:', quant_max_processed)
            
        #for int8O gemm deQuant
        for j in range(INT8O_GEMM_NUM):
            amaxList[amax_id] = (int8O_gemm_input_amax_list[j]*int8O_gemm_weight_amax_list[j])/(127.0*int8O_gemm_output_amax_list[j])
            amax_id += 1

        #for trt fused MHA amax 
        #### QKV_addBias_amax
        amaxList[amax_id] = np.maximum(np.maximum(amaxList[8],amaxList[16]), amaxList[24])
        amax_id += 1
        #### softmax amax
        amaxList[amax_id] = amaxList[32]
        amax_id += 1
        #### bmm2 amax
        amaxList[amax_id] = amaxList[36]
        amax_id += 1

        init_dict["transformer.encoder.layer.{}.amaxList".format(i)] = torch.tensor(amaxList, dtype=torch.float32)
        if verbose:
            print("done process layer_{} kernel weight".format(i))

    print("Quantizing checkpoint done.")
    return init_dict


if __name__ == '__main__':
    model_dict = torch.load('checkpoint/ViT-B_16_calib.pth', map_location='cpu')
    checkpoint_quantization(model_dict, '../../../build/lib/libth_transformer.so', verbose=True)