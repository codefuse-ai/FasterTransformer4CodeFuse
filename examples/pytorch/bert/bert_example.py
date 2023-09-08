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

from __future__ import print_function
import threading

import os
import argparse
import timeit
import torch
import torch.distributed as dist
import torch.cuda.nvtx as nvtx
import time
import sys
import numpy as np
import random
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.utils import print_memory_usage
from examples.pytorch.bert.utils.encoder import HuggingFaceEncoder
from examples.pytorch.bert.utils.encoder import CustomEncoder
from examples.pytorch.bert.utils.encoder import EncoderWeights


def sequence_mask(lengths, max_len=None, is_2d=True):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    mask = (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
    if is_2d:
        return mask
    else:
        mask = mask.view(-1, 1, 1, max_len)
        m2 = mask.transpose(2, 3)
        return mask * m2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', type=int,
                        help='batch size')
    parser.add_argument('layer_num', type=int,
                        help='number of layers')
    parser.add_argument('seq_len', type=int,
                        help='sequence length')
    parser.add_argument('head_num', type=int,
                        help='head number')
    parser.add_argument('head_size', type=int,
                        help='size per head')
    parser.add_argument('--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--int8_mode', type=int, default=0, metavar='NUMBER',
                        help='int8 mode (default: 0)', choices=[0, 1, 2, 3])
    parser.add_argument('--time', action='store_true',
                        help='test the time or not.')
    parser.add_argument('--avg_seq_len', type=int, default=-1, metavar='NUMBER',
                        help='average sequence length (default: -1)')
    parser.add_argument('--sparse', action='store_true',
                        help='Whether use sparse feature (only support SM 8.0 and 8.6, and SPARSITY_SUPPORT need be ON).')
    parser.add_argument('--weight_path', type=str,
                        default=None,
                        help='path containing the pretrained weights')
    parser.add_argument('--ths_path', type=str, default='./lib/libth_transformer.so',
                        help='path of the pyt_fastertransformer dynamic lib file')
    parser.add_argument('-thread_num', '--thread_num', type=int, default=1, metavar='int',
                        help='Testing multithread if thread_num > 1.')
    parser.add_argument('-tensor_para_size', '--tensor_para_size', type=int, default=1, metavar='int',
                        help='Size of tensor parallelism.')
    parser.add_argument('-pipeline_para_size', '--pipeline_para_size', type=int, default=1, metavar='int',
                        help='Size of pipeline parallelism.')
    parser.add_argument('--error-threshold', type=float,
                        help='Threshold of error')

    args = parser.parse_args()
    bert_example(vars(args))


def bert_example(args):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    if dist.is_mpi_available():
        try:
            dist.init_process_group(backend='mpi')
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
    else:
        rank = 0

    batch_size = args['batch_size']
    seq_len = args['seq_len']
    if args['weight_path'] is not None:
        if 'large' in args['weight_path']:
            layer_num = 24
            head_num = 16
            head_size = 64
        elif 'base' in args['weight_path']:
            layer_num = 12
            head_num = 12
            head_size = 64
        else:
            layer_num = args['layer_num']
            head_num = args['head_num']
            head_size = args['head_size']
    else:
        layer_num = args['layer_num']
        head_num = args['head_num']
        head_size = args['head_size']
    hidden_dim = head_num * head_size

    if args['int8_mode'] == 1:
        per_channel = True
    elif args['int8_mode'] == 2 or args['int8_mode'] == 3:
        per_channel = False
    elif args['int8_mode'] != 0:
        raise ValueError("wrong int8_mode argument")

    if rank == 0:
        print("\n=============== Argument ===============")
        for key in args:
            print("{}: {}".format(key, args[key]))
        print("========================================\n")

    inp = torch.empty(batch_size, seq_len, hidden_dim).cuda()
    torch.nn.init.normal_(inp, -0.02, 0.02)
    if args['avg_seq_len'] > 0:
        mem_seq_lens = torch.ones((batch_size,)) * args['avg_seq_len']
        mem_seq_lens = mem_seq_lens.to(torch.int32).cuda()
    elif args['avg_seq_len'] == -1:
        mem_seq_lens = torch.randint(1, seq_len + 1, (batch_size,), dtype=torch.int32).cuda()
    else:
        raise ValueError("wrong avg_seq_len")

    mask = sequence_mask(mem_seq_lens, args['seq_len'], False).to(torch.float)
    # mask = torch.randint(0, 2, (batch_size, seq_len, seq_len), dtype=torch.float32).cuda()
    if args['data_type'] == 'fp16' or args['int8_mode'] != 0:
        inp = inp.half()
        mask = mask.half()
    elif args['data_type'] == 'bf16':
        inp = inp.bfloat16()
        mask = mask.bfloat16()

    pretrained_weights = torch.load(args['weight_path']) if (args['weight_path'] is not None) else None
    weights = EncoderWeights(layer_num, hidden_dim, pretrained_weights, args['sparse'])
    ft_weights = EncoderWeights(layer_num, hidden_dim, weights.weights, args['sparse'],
                                args["tensor_para_size"], args["pipeline_para_size"])
    world_size = dist.get_world_size() if dist.is_mpi_available() else 1
    assert world_size == args["tensor_para_size"] * \
        args["pipeline_para_size"], f"[ERROR] world_size ({world_size}) != tensor_para_size ({args['tensor_para_size']}) * pipeline_para_size ({args['pipeline_para_size']})"
    ft_weights._generated_weights = True  # for int8 handling
    if rank == 0:
        hf_encoder = HuggingFaceEncoder(layer_num, head_num, head_size, weights)
        hf_encoder.cuda()
        if args['data_type'] == 'fp16' or args['int8_mode'] != 0:
            hf_encoder.half()
        elif args['data_type'] == 'bf16':
            hf_encoder.bfloat16()
        hf_encoder.eval()
        hf_encoder = torch.jit.trace(hf_encoder, (inp, mask))

    if args['int8_mode'] != 0:
        ft_weights.to_int8(args['sparse'], args['ths_path'])
    elif args['data_type'] == 'fp16':
        ft_weights.to_half()
    elif args['data_type'] == 'bf16':
        ft_weights.to_bfloat16()
    ft_weights.to_cuda()
    custom_encoder = CustomEncoder(layer_num, head_num, head_size, ft_weights,
                                   int8_mode=args['int8_mode'],
                                   remove_padding=False,
                                   sparse=args['sparse'],
                                   path=args['ths_path'],
                                   tensor_para_size=args["tensor_para_size"],
                                   pipeline_para_size=args["pipeline_para_size"])
    custom_encoder = torch.jit.script(custom_encoder)

    eff_custom_encoder = CustomEncoder(layer_num, head_num, head_size, ft_weights,
                                       int8_mode=args['int8_mode'],
                                       remove_padding=True,
                                       sparse=args['sparse'],
                                       path=args['ths_path'],
                                       tensor_para_size=args["tensor_para_size"],
                                       pipeline_para_size=args["pipeline_para_size"])
    eff_custom_encoder = torch.jit.script(eff_custom_encoder)

    with torch.no_grad():
        output_mask = sequence_mask(mem_seq_lens, args['seq_len']).to(mask.dtype).unsqueeze(-1)
        if rank == 0:
            hf_output = hf_encoder(inp, mask)[0] * output_mask
            print(hf_output)
            print(hf_output.size())

        ft_inp = inp.to(f"cuda:{rank}")
        ft_mask = mask.to(f"cuda:{rank}")
        ft_mem_seq_lens = mem_seq_lens.to(f"cuda:{rank}")
        ft_output_mask = output_mask.to(f"cuda:{rank}")
        ft_output = custom_encoder(ft_inp, ft_mask, ft_mem_seq_lens)[0] * ft_output_mask
        if rank == 0:
            print(ft_output)
            print(ft_output.size())

        eff_ft_output = eff_custom_encoder(ft_inp, ft_mask, ft_mem_seq_lens)[0] * ft_output_mask
        if rank == 0:
            print(eff_ft_output)
            print(eff_ft_output.size())

        if rank == 0:
            FT_diff = torch.abs(hf_output - ft_output).float()  # Prevent error under bfloat16
            print('FT Mean diff: {}'.format(torch.mean(FT_diff)))
            print('FT Max diff:  {}'.format(torch.max(FT_diff)))
            print('FT Min diff:  {}'.format(torch.min(FT_diff)))

        if rank == 0:
            EFF_diff = torch.abs(hf_output - eff_ft_output).float()  # Prevent error under bfloat16
            print('EFF-FT Mean diff: {}'.format(torch.mean(EFF_diff)))
            print('EFF-FT Max diff:  {}'.format(torch.max(EFF_diff)))
            print('EFF-FT Min diff:  {}'.format(torch.min(EFF_diff)))

        if args['time']:
            iterations = 100

            if rank == 0:
                for i in range(iterations):
                    output = hf_encoder(inp, mask)
                t10 = timeit.default_timer()
                # nvtx.range_push("hf")
                for i in range(iterations):
                    # nvtx.range_push("hf"+str(i))
                    output = hf_encoder(inp, mask)
                    # nvtx.range_pop()
                # nvtx.range_pop()
                t1 = timeit.default_timer() - t10
                # time.sleep(60)

            for i in range(iterations):
                output = custom_encoder(ft_inp, ft_mask, ft_mem_seq_lens)
            t20 = timeit.default_timer()
            # nvtx.range_push("ext")
            for i in range(iterations):
                # nvtx.range_push("ext"+str(i))
                output = custom_encoder(ft_inp, ft_mask, ft_mem_seq_lens)
                # nvtx.range_pop()
            # nvtx.range_pop()
            t2 = timeit.default_timer() - t20
            # time.sleep(60)

            for i in range(iterations):
                output = eff_custom_encoder(ft_inp, ft_mask, ft_mem_seq_lens)
            t30 = timeit.default_timer()
            # nvtx.range_push("eff_ext")
            for i in range(iterations):
                # nvtx.range_push("eff_ext"+str(i))
                output = eff_custom_encoder(ft_inp, ft_mask, ft_mem_seq_lens)
                # nvtx.range_pop()
            # nvtx.range_pop()
            t3 = timeit.default_timer() - t30
            # time.sleep(60)

            if rank == 0:
                print("[INFO] HuggingFaceEnocder time costs:    {:.2f} ms".format(t1 * 1000 / iterations))
                print("[INFO] FasterTransformer time costs:     {:.2f} ms".format(t2 * 1000 / iterations))
                print("[INFO] EFF-FasterTransformer time costs: {:.2f} ms".format(t3 * 1000 / iterations))

        if args['thread_num'] > 1:
            # Multi-threading demonstration
            assert world_size == 1, "[ERROR] multi thread does not support MGMN"
            thread_list = []
            thread_num = args['thread_num']
            iterations = 100

            def run():
                t40 = timeit.default_timer()
                for i in range(iterations):
                    ft_output = custom_encoder(ft_inp, ft_mask, ft_mem_seq_lens)[0] * ft_output_mask
                t4 = timeit.default_timer() - t40
                if rank == 0:
                    diff = torch.abs(hf_output - ft_output)
                    print('FT Mean diff: {}'.format(torch.mean(diff)))
                    print('FT Max diff:  {}'.format(torch.max(diff)))
                    print('FT Min diff:  {}'.format(torch.min(diff)))
                    print("[INFO] batch_size {} max_seq_len {} {} layer FT-OP-time {:6.2f} ms with {} threads".format(batch_size,
                                                                                                                      seq_len, layer_num, t4, thread_num))

            for i in range(thread_num):
                thread_list.append(threading.Thread(target=run, name="RunFT"))
            for t in thread_list:
                t.start()
            for t in thread_list:
                t.join()

    torch.cuda.empty_cache()
    sys.stdout.flush()
    if rank == 0:
        if (args["error_threshold"] != None):
            assert max(torch.mean(FT_diff), torch.mean(EFF_diff)) < args["error_threshold"], "[ERROR] TEST FAIL!"
            print("[INFO] TEST PASS!")

        return max(torch.mean(FT_diff), torch.mean(EFF_diff))
    else:
        return 0


if __name__ == '__main__':
    main()
