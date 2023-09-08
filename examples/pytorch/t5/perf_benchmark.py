# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

"""
perf_benchmark.py

Unlike translate_example.py, this script focuses on benchmarking the performance of the FasterTransformers T5
implementation such that it can be compared with other frameworks apples-to-apples. The changes include:

- Use random input data and disable accuracy checking.
- Use fixed input/output sequence lengths and disable early_stopping.
- Add better controls on the number of warm-ups and the number of iterations to run the inference for.

"""

import argparse
import configparser
import os
import sys
import math
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(dir_path + "/../../../3rdparty/transformers/src/")

from transformers import T5ForConditionalGeneration # transformers-4.10.0-py3
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")

from examples.pytorch.t5.utils.ft_encoder import FTT5EncoderWeight, FTT5Encoder
from examples.pytorch.t5.utils.ft_decoding import FTT5DecodingWeight, FTT5Decoding, FTT5

class TranslationResult(object):
    def __init__(self, name, frame_work):
        self.name = name
        self.frame_work = frame_work # FT or HF
        self.file_name = name + ".txt"

        self.token_list = []
        self.batch_ids_list = []
        self.batch_seq_len_list = []
        self.batch_num = 0
        self.execution_time = 0.0  # seconds
        self.token_num = 0

class InputTokens(object):
    def __init__(self, batch_size, input_seq_len, bos_token, eos_token, vocab_size):
        # Set the last token of each sequence to eos and replace the bos/eos tokens in the middle of the sequences to
        # some other tokens.
        normal_token_list = list(range(vocab_size))
        if bos_token in normal_token_list:
            normal_token_list.remove(bos_token)
        if eos_token in normal_token_list:
            normal_token_list.remove(eos_token)
        self.input_ids = torch.randint(0, len(normal_token_list), (batch_size, input_seq_len))
        for batch_idx in range(batch_size):
            for token_idx in range(input_seq_len):
                if token_idx == input_seq_len - 1:
                    self.input_ids[batch_idx][token_idx] = eos_token
                else:
                    self.input_ids[batch_idx][token_idx] = normal_token_list[self.input_ids[batch_idx][token_idx]]
        # Set attention masks to all ones.
        self.attention_mask = torch.ones((batch_size, input_seq_len), dtype=torch.int64)

def translate(args_dict):
    torch.set_printoptions(precision=6)
    batch_size = args_dict['batch_size']
    beam_size = args_dict['beam_width']
    output_seq_len = args_dict['seq_len']
    input_seq_len = args_dict['input_seq_len'] if args_dict['input_seq_len'] > 0 else output_seq_len
    time_args = args_dict["test_time"]
    beam_search_diversity_rate = args_dict['beam_search_diversity_rate']
    topk = args_dict['sampling_topk']
    topp = args_dict['sampling_topp']
    tensor_para_size = args_dict['tensor_para_size']
    pipeline_para_size = args_dict['pipeline_para_size']
    warmup_iterations = args_dict['warmup_iterations']
    infer_iterations = args_dict['iterations']
    infer_duration = args_dict['duration']
    seed = args_dict['seed']
    skip_gemm = args_dict['skip_gemm']
    torch.manual_seed(seed)

    ## huggingface without bias and use relative position embedding
    ## relative position embedding -> 0, absolute position embedding -> 1
    t5_with_bias = False
    use_gated_activation = False
    t5_with_moe = False
    position_embedding_type = 0
    weight_data_type = np.float32
    ## only huggingface model path supported
    model_path = args_dict['model_path'] if args_dict['model_path'] != None else args_dict['model']
    ckpt_path = args_dict['ckpt_path']
    model_type = args_dict['model_type']
    ## read checkpoint config if exists
    ckpt_config = configparser.ConfigParser()
    activation_type = "relu"
    if (model_type in ["Megatron", "Megatron-DeepSpeed"]):
        ckpt_config_path = os.path.join(ckpt_path, 'config.ini')
        if os.path.isfile(ckpt_config_path):
            ckpt_config.read(ckpt_config_path)
            ## update structure config
            t5_with_bias = ckpt_config.getboolean('structure', 't5_with_bias')
            position_embedding_type = 0 if ckpt_config.get('structure', 'position_embedding_type') == 'relative' else 1
            use_gated_activation = ckpt_config.getboolean('structure', 'use_gated_activation')
            t5_with_moe= ckpt_config.getint('structure', 't5_with_moe') == 1
            weight_data_type = {"fp16": np.float16, "fp32": np.float32}[ckpt_config.get("encoder", "weight_data_type")]
            activation_type = "gated-gelu" if use_gated_activation else "gelu" # change to gelu, which is default setting of Megatron T5
            moe_layers_in_encoder = []
            moe_layers_in_decoder = []
            if (ckpt_config.get('structure', 'moe_layers_in_encoder') != '[]'):
                moe_layers_in_encoder = [int(n) for n in ckpt_config.get('structure', 'moe_layers_in_encoder')[1:-1].replace(" ", "").split(',')]
            if (ckpt_config.get('structure', 'moe_layers_in_decoder') != '[]'):
                moe_layers_in_decoder = [int(n) for n in ckpt_config.get('structure', 'moe_layers_in_encoder')[1:-1].replace(" ", "").split(',')]

        else:
            raise Exception("config file does exist with the ckpt !")

    if model_type == "Megatron" and args_dict['ckpt_path'] == None:
        raise Exception("Megatron T5 model needs to specify checkpoint path !")

    print("\n=============== Argument ===============")
    for key in args_dict:
        print("{}: {}".format(key, args_dict[key]))
    print("========================================")

    lib_path = args_dict['lib_path']

    t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    if dist.is_mpi_available():
        try:
            dist.init_process_group(backend='mpi')
            rank = dist.get_rank()
        except:
            rank = dist.get_rank()
    else:
        rank = 0
    
    if time_args.find("0") != -1 or time_args.find("2") != -1:
        t5_model = t5_model.to(rank)
        if args_dict['data_type'] == 'fp16':
            t5_model = t5_model.half()

    encoder_config = t5_model.encoder.config
    decoder_config = t5_model.decoder.config
    encoder_config.update({"num_experts": 0})
    decoder_config.update({"num_experts": 0})
    encoder_config.update({"moe_layer_index": []})
    decoder_config.update({"moe_layer_index": []})

    q_scaling = 1.0 / (math.sqrt(encoder_config.d_kv))
    if (model_type in ["Megatron", "Megatron-DeepSpeed"]):
        ## update configs when using Megatron model structure
        q_scaling = 1.0

        encoder_config.d_model = ckpt_config.getint('encoder', 'd_model')
        encoder_config.vocab_size = ckpt_config.getint('encoder', 'vocab_size')
        encoder_config.num_heads = ckpt_config.getint('encoder', 'num_heads')
        encoder_config.d_kv = ckpt_config.getint('encoder', 'd_kv')
        encoder_config.d_ff = ckpt_config.getint('encoder', 'd_ff')
        encoder_config.num_layers = ckpt_config.getint('encoder', 'num_layers')
        encoder_config.relative_attention_num_buckets = ckpt_config.getint('encoder', 'relative_attention_num_buckets_or_max_pos_seq_len')
        if model_type == "Megatron-DeepSpeed":
            encoder_config.num_experts = ckpt_config.getint('encoder', 'num_experts')
            encoder_config.moe_layer_index = moe_layers_in_encoder

        decoder_config.d_model = ckpt_config.getint('decoder', 'd_model')
        decoder_config.vocab_size = ckpt_config.getint('decoder', 'vocab_size')
        decoder_config.num_heads = ckpt_config.getint('decoder', 'num_heads')
        decoder_config.d_kv = ckpt_config.getint('decoder', 'd_kv')
        decoder_config.d_ff = ckpt_config.getint('decoder', 'd_ff')
        decoder_config.num_layers = ckpt_config.getint('decoder', 'num_layers')
        decoder_config.relative_attention_num_buckets = ckpt_config.getint('decoder', 'relative_attention_num_buckets_or_max_pos_seq_len')
        if model_type == "Megatron-DeepSpeed":
            decoder_config.num_experts = ckpt_config.getint('decoder', 'num_experts')
            decoder_config.moe_layer_index = moe_layers_in_decoder
        decoder_config.decoder_start_token_id = ckpt_config.getint('decoder', 'decoder_start_token_id')
        decoder_config.eos_token_id = ckpt_config.getint('decoder', 'eos_token_id')

    print(f"{model_type} encoder_config: {encoder_config}")
    print(f"{model_type} decoder_config: {decoder_config}")

    if os.path.isfile("gemm_config.in") and rank == 0:
        cmd = f"rm gemm_config.in"
        print(f"Run {cmd}")
        os.system(cmd)
    translation_result_list = []
    if time_args.find("0") != -1:
        translation_result_list.append(TranslationResult("hf-beamsearch-warmup", "HF"))
        translation_result_list.append(TranslationResult("hf-beamsearch", "HF"))
    if time_args.find("1") != -1:
        translation_result_list.append(TranslationResult("ft-beamsearch-warmup", "FT"))
        translation_result_list.append(TranslationResult("ft-beamsearch", "FT"))
        if rank == 0 and not skip_gemm:
            is_fp16 = 1 if args_dict['data_type'] == 'fp16' else 0
            cmd = f"./bin/t5_gemm {math.ceil(batch_size / pipeline_para_size)} {beam_size} {128} " \
                f"{encoder_config.d_model} {encoder_config.num_heads} {encoder_config.d_kv} {encoder_config.d_ff} " \
                f"{decoder_config.d_model} {decoder_config.num_heads} {decoder_config.d_kv} {decoder_config.d_ff} " \
                f"{decoder_config.vocab_size} {is_fp16} {tensor_para_size} 1 > .tmp_gemm.log"
            print(f"Run gemm test: {cmd}")
            os.system(cmd)
    if time_args.find("2") != -1:
        translation_result_list.append(TranslationResult("hf-sampling-warmup", "HF"))
        translation_result_list.append(TranslationResult("hf-sampling", "HF"))
    if time_args.find("3") != -1:
        translation_result_list.append(TranslationResult("ft-sampling-warmup", "FT"))
        translation_result_list.append(TranslationResult("ft-sampling", "FT"))
        if rank == 0 and not skip_gemm:
            is_fp16 = 1 if args_dict['data_type'] == 'fp16' else 0
            cmd = f"./bin/t5_gemm {math.ceil(batch_size / pipeline_para_size)} {1} {128} " \
                f"{encoder_config.d_model} {encoder_config.num_heads} {encoder_config.d_kv} {encoder_config.d_ff} " \
                f"{decoder_config.d_model} {decoder_config.num_heads} {decoder_config.d_kv} {decoder_config.d_ff} " \
                f"{decoder_config.vocab_size} {is_fp16} {tensor_para_size} 1 1 > .tmp_gemm.log"
            print(f"Run gemm test: {cmd}")
            os.system(cmd)

    if time_args.find("1") != -1 or time_args.find("3") != -1:
        ft_encoder_weight = FTT5EncoderWeight(
            encoder_config,
            tensor_para_size,
            pipeline_para_size,
            t5_with_bias=t5_with_bias,
            use_gated_activation=use_gated_activation,
            t5_with_moe=t5_with_moe,
            position_embedding_type=position_embedding_type,
            weight_data_type=weight_data_type
        )
        ft_decoding_weight = FTT5DecodingWeight(
            decoder_config,
            tensor_para_size,
            pipeline_para_size,
            t5_with_bias=t5_with_bias,
            use_gated_activation=use_gated_activation,
            t5_with_moe=t5_with_moe,
            position_embedding_type=position_embedding_type,
            weight_data_type=weight_data_type,
        )

        if args_dict["ckpt_path"] is not None:
            ft_encoder_weight.load_from_bin(args_dict["ckpt_path"], model_type=model_type)
            ft_decoding_weight.load_from_bin(args_dict["ckpt_path"], model_type=model_type)
        else:
            ft_encoder_weight.load_from_model(t5_model)
            ft_decoding_weight.load_from_model(t5_model)
        
        if args_dict['data_type'] == 'fp16':
            t5_model = t5_model.half()
            ft_encoder_weight.to_half()
            ft_decoding_weight.to_half()

        # This script assumes fixed sequence length, so using remove_padding will not benefit.
        remove_padding = False
        ft_encoder = FTT5Encoder(ft_encoder_weight.w, lib_path, encoder_config.num_heads,
                                encoder_config.d_kv, encoder_config.d_ff,
                                encoder_config.d_model, remove_padding, encoder_config.num_layers,
                                encoder_config.relative_attention_num_buckets, encoder_config.num_experts, encoder_config.moe_layer_index,
                                128, False, q_scaling, tensor_para_size, pipeline_para_size, t5_with_bias,
                                position_embedding_type, 0, activation_type)
        ft_decoding = FTT5Decoding(ft_decoding_weight.w, lib_path,
                                decoder_config.num_heads, decoder_config.d_kv,
                                decoder_config.d_ff, encoder_config.d_model,
                                decoder_config.d_model, decoder_config.num_layers,
                                decoder_config.decoder_start_token_id,
                                # Set eos token id to -1 to effectively disable early stopping.
                                # decoder_config.eos_token_id,
                                -1,
                                decoder_config.vocab_size,
                                q_scaling,
                                decoder_config.relative_attention_num_buckets, decoder_config.num_experts, decoder_config.moe_layer_index, max_distance=128,
                                tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size,
                                t5_with_bias=t5_with_bias, moe_k=0, activation_type=activation_type,
                                position_embedding_type = position_embedding_type)

        ft_t5 = FTT5(ft_encoder, ft_decoding)

    input_token = InputTokens(batch_size, input_seq_len, decoder_config.decoder_start_token_id, decoder_config.eos_token_id, decoder_config.vocab_size)

    for i in range(len(translation_result_list)):
        sys.stdout.flush()
        is_warmup = (translation_result_list[i].name.find("warmup") != -1)
        min_duration = infer_duration if not is_warmup else 0
        min_iterations = infer_iterations if not is_warmup else warmup_iterations
        iter_idx = 0
        
        start_time = datetime.now()
        while iter_idx < min_iterations or (datetime.now() - start_time).total_seconds() < min_duration:
            iter_idx += 1

            if translation_result_list[i].frame_work == "HF":
                if translation_result_list[i].name.find("beamsearch") != -1:
                    hf_outputs = t5_model.generate(input_token.input_ids.to("cuda"),
                                                   min_length=output_seq_len + 1,
                                                   max_length=output_seq_len + 1, # "+1" because HF counts <bos> as well.
                                                   early_stopping=False,
                                                   num_beams=beam_size)
                elif translation_result_list[i].name.find("sampling") != -1:
                    hf_outputs = t5_model.generate(input_token.input_ids.to("cuda"),
                                                   min_length=output_seq_len + 1,
                                                   max_length=output_seq_len + 1, # "+1" because HF counts <bos> as well.
                                                   early_stopping=False,
                                                   do_sample=True,
                                                   top_k=topk if topk > 0 else None,
                                                   top_p=topp if topp > 0.0 else None)
                translation_result_list[i].batch_ids_list.append(hf_outputs)
                translation_result_list[i].batch_seq_len_list.append(np.ones(input_seq_len) * output_seq_len)
            elif translation_result_list[i].frame_work == "FT":
                tmp_beam_size = beam_size
                if translation_result_list[i].name.find("sampling") != -1:
                    tmp_beam_size = 1
                ft_decoding_outputs, ft_decoding_seq_lens = ft_t5(input_token,
                                                                  None,
                                                                  tmp_beam_size,
                                                                  output_seq_len,
                                                                  topk,
                                                                  topp,
                                                                  beam_search_diversity_rate=beam_search_diversity_rate)
                translation_result_list[i].batch_ids_list.append(ft_decoding_outputs)
                translation_result_list[i].batch_seq_len_list.append(ft_decoding_seq_lens)

            translation_result_list[i].batch_num += 1

        stop_time = datetime.now()
        translation_result_list[i].execution_time = (stop_time - start_time).total_seconds()
        if translation_result_list[i].name.find("warmup") != -1:
            continue
        
        for batch_token, batch_seq_len in zip(translation_result_list[i].batch_ids_list, translation_result_list[i].batch_seq_len_list):
            for j in range(len(batch_token)):
                if translation_result_list[i].frame_work == "HF":
                    translation_result_list[i].token_list.append(batch_token[j][1:])
                    translation_result_list[i].token_num += sum(batch_token[j][1:] != 0)
                elif translation_result_list[i].frame_work == "FT":
                    translation_result_list[i].token_list.append(batch_token[j][0][:batch_seq_len[j][0]])
                    translation_result_list[i].token_num += batch_seq_len[j][0]

    if rank == 0:
        for t in translation_result_list:
            if t.name.find("warmup") != -1: 
                continue
            print(f"[INFO] {t.name} translates {t.batch_num} batches taking {t.execution_time:.2f} sec to translate "
                f"{t.token_num} tokens ({(t.execution_time / t.batch_num * 1000):.4f} ms per batch), "
                f"{(t.token_num / t.execution_time):.0f} tokens/sec.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-batch', '--batch_size', type=int, default=1, metavar='NUMBER',
                        help='batch size (default: 1)')
    parser.add_argument('-beam', '--beam_width', type=int, default=4, metavar='NUMBER',
                        help='beam width (default: 4)')
    parser.add_argument('-s', '--seq_len', type=int, default=256, metavar='NUMBER',
                        help='fixed output sequence length, excluding bos but including eos (default: 256)')
    parser.add_argument('-inseq', '--input_seq_len', type=int, default=0, metavar='NUMBER',
                        help='fixed input sequence length, including eos (default: same as fixed output sequence length)')
    parser.add_argument('-time', '--test_time', type=str, default='', metavar='STRING',
                        help='''
                            Test the time of which one (default: '' (not test anyone) ); 
                            '': not test anyone 
                            '0': test hf_beamsearch  
                            '1': test ft_beamsearch 
                            '2': test hf_sampling 
                            '3': test ft_sampling 
                            'e.g., if you want to test tf_beamsearch and ft_sampling, 
                            then you need to use -time '03' ''')
    parser.add_argument('-diversity_rate', '--beam_search_diversity_rate', type=float, default=0.0, metavar='NUMBER',
                        help='deviersity rate of beam search. default is 0. When diversity rate = 0, it is equivalent to the naive beam search.')
    parser.add_argument('-topk', '--sampling_topk', type=int, default=1, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('-topp', '--sampling_topp', type=float, default=0.0, metavar='NUMBER',
                        help='Probability (p) value of top p sampling in decoding. Default is 0.0. ')
    parser.add_argument('-d', '--data_type', type=str, default="fp32", metavar='STRING',
                        help='data type for inference (default: fp32)', choices=['fp32', 'fp16'])
    parser.add_argument('-ld', '--load_data_type', type=str, default="fp32", metavar='STRING',
                        help='data type for loading weights (default: fp32)', choices=['fp32', 'fp16'])
    parser.add_argument('-lib_path', '--lib_path', type=str, default="lib/libth_transformer.so", metavar='STRING',
                        help='the path of FasterTransformer pytorch t5 op library.')
    parser.add_argument('-model_path', '--model_path', type=str, default=None, metavar='STRING',
                        help='T5 model path.')
    parser.add_argument('-model', '--model', type=str, default="t5-small", metavar='STRING',
                        help='T5 model size. Only used when --model_path=None', choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"])
    parser.add_argument('-tensor_para_size', '--tensor_para_size', type=int, default=1, metavar='NUMBER',
                        help='size of tensor parallelism (default: 1)')
    parser.add_argument('-pipeline_para_size', '--pipeline_para_size', type=int, default=1, metavar='NUMBER',
                        help='size of pipeline parallelism (default: 1)')
    # assume checkpoint config is also in the same path
    parser.add_argument('--ckpt_path', type=str, help='path to the checkpoint file.')
    parser.add_argument('--model_type', type=str, default="Huggingface", choices=["Huggingface", "Megatron", "Megatron-DeepSpeed"],
                        help='Megatron T5 uses bias and supports both absulte and relative positional embedding;'
                        'Huggingface T4 adopts the paper\'s implementation and has no bias')
    # flags for performance benchmarking
    parser.add_argument('-warmup_iter', '--warmup_iterations', type=int, default=1, metavar='NUMBER',
                        help='Number of warm-up iterations for each implementation.')
    parser.add_argument('-iter', '--iterations', type=int, default=10, metavar='NUMBER',
                        help='Minimal number of inference iterations for each implementation.')
    parser.add_argument('-duration', '--duration', type=int, default=3, metavar='NUMBER',
                        help='Minimal duration in seconds for inference iterations for each implementation.')
    parser.add_argument('-seed', '--seed', type=int, default=0, metavar='NUMBER',
                        help='Random seed used to generate random input values.')
    parser.add_argument('-skip_gemm', '--skip_gemm', action="store_true",
                        help='Skip the gemm autotuning by not calling the ./bin/t5_gemm binary.')
    args = parser.parse_args()

    translate(vars(args))