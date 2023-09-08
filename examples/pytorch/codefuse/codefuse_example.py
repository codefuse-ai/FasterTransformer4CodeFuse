# -*- encoding: utf-8 -*-
import logging
import os
import sys
import json
import torch
import time
from typing import List
import numpy as np
from configparser import ConfigParser
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoTokenizer
import argparse
import traceback
import random


#  _   _ _____ ___ _     ____    ____   ____ ____  ___ ____ _____ 
# | | | |_   _|_ _| |   / ___|  / ___| / ___|  _ \|_ _|  _ \_   _|
# | | | | | |  | || |   \___ \  \___ \| |   | |_) || || |_) || |  
# | |_| | | |  | || |___ ___) |  ___) | |___|  _ < | ||  __/ | |  
#  \___/  |_| |___|_____|____/  |____/ \____|_| \_\___|_|    |_|  

def to_word_list_format(words_list: List[List[str]], tokenizer):

    flat_ids = []
    offsets = []
    for words in words_list:
        item_flat_ids = []
        item_offsets = []

        for word in words:
            ids = tokenizer.encode(word)

            if len(ids) == 0:
                continue

            item_flat_ids += ids
            item_offsets.append(len(ids))

        flat_ids.append(np.array(item_flat_ids))
        offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    result = np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))
    return torch.from_numpy(np.ascontiguousarray(result))


def is_chinese_char(cp):
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True
    return False

def is_garbage(cp):
    if is_chinese_char(cp):
        return False
    # ASCII
    if cp < 128: 
        return False
    # ，。？！、；：“”‘’（）《》【】{}[]<>|-=_+*&^%$#@￥~·`…
    if cp in [65292, 12290, 65311, 65281, 12289, 65307, 65306, 8220, 8221, 8216, 8217, 65288, 
                65289, 12298, 12299, 12304, 12305, 123, 125, 91, 93, 60, 62, 124, 45, 61, 95, 
                43, 42, 38, 94, 37, 36, 35, 64, 65509, 126, 183, 96, 8230]:
        return False
    return True

class token_stream_2_str_stream_convertor:
    def __init__(self, end_id, tokenizer, local_rank) -> None:
        self.token_cache = []
        self.print_len = 0

        self.end_id = end_id
        self.tokenizer = tokenizer
        self.has_stop = False
        self.local_rank = local_rank
    
    def send_str(self, str_to_send):
        if self.local_rank == 0:
            print(str_to_send, end="", flush=True)

    def send_finish(self):
        if self.local_rank == 0:
            print("\n\nend\n\n", end="", flush=True)
    
    def append_token(self, token):
        if self.has_stop:
            return

        # ref huggingface TextStreamer
        if token != self.end_id:
            self.token_cache.append(token)
        text = self.tokenizer.decode(self.token_cache)
        if token == self.end_id:
            printable_text = text[self.print_len :] if len(text) > 0 else ""
            if len(printable_text) > 0 and is_garbage(ord(printable_text[-1])):
                printable_text = printable_text[: -1]
            self.token_cache = []
            self.print_len = 0
        elif text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        elif len(text) > 0 and is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.send_str(printable_text)

        if token == self.end_id:
            self.has_stop = True
            self.send_finish()

class TrieNode():
    def __init__(self):
        self.children = {}
        self.last = False

class Trie():
    def __init__(self, vocab):
        self.root = TrieNode()
        self.vocab = vocab
        self.formTrie((self.vocab.keys()))

    def formTrie(self, keys):
        for key in keys:
            self.insert(key)

    def insert(self, key):
        node = self.root
        for a in key:
            if not node.children.get(a):
                node.children[a] = TrieNode()
            node = node.children[a]
        node.last = True

    def suggestionsRec(self, node, word, results):
        if node.last:
            results.append((word, self.vocab[word]))
        for a, n in node.children.items():
            self.suggestionsRec(n, word + a, results)

    def printAutoSuggestions(self, key, results):
        node = self.root
        for a in key:
            if not node.children.get(a):
                return 0
            node = node.children[a]

        if not node.children:
            return -1

        self.suggestionsRec(node, key, results)
        return 1

#  __  __  ___  ____  _____ _       ____   ____ ____  ___ ____ _____ 
# |  \/  |/ _ \|  _ \| ____| |     / ___| / ___|  _ \|_ _|  _ \_   _|
# | |\/| | | | | | | |  _| | |     \___ \| |   | |_) || || |_) || |  
# | |  | | |_| | |_| | |___| |___   ___) | |___|  _ < | ||  __/ | |  
# |_|  |_|\___/|____/|_____|_____| |____/ \____|_| \_\___|_|    |_|  

str_type_map = {"fp32": torch.float32, "fp16": torch.float16}

class GptNeoXWeights(object):
    def __init__(self, 
                 head_num, size_per_head, layer_num, vocab_size, 
                 max_seq_len, tensor_para_size, pipeline_para_size, use_gptj_residual, 
                 int8_mode=0, inference_data_type: str = "fp16",
                 weights_data_type: np.dtype = np.float32, enable_int8_weights=False, use_pybind11=False):
        assert(head_num % tensor_para_size == 0)
        if int8_mode == 1:
            torch_infer_dtype = str_type_map[inference_data_type]
            assert torch_infer_dtype == torch.float16 or torch_infer_dtype == torch.bfloat16, "Weight only quant only supported for infer type fp16 or bf16."
            if not use_pybind11:
                quant = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix
                self.weight_transpose_calibrate_quantize = lambda x: quant(x, torch.int8)
            else:
                import libth_common
                self.weight_transpose_calibrate_quantize = libth_common.symmetric_quantize_last_axis_of_batched_matrix_int8
        else:
            assert int8_mode == 0, "Invalid int8 mode for GPT. Must be 0 or 1"

        self.head_num = head_num
        self.size_per_head = size_per_head
        self.layer_num = layer_num
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.layers_per_device = layer_num // pipeline_para_size

        self.use_gptj_residual = use_gptj_residual

        local_head_num = head_num // tensor_para_size
        global_head_num = head_num
        local_hidden_units = local_head_num * size_per_head
        global_hidden_units = global_head_num * size_per_head
        local_inter_size = local_hidden_units * 4

        self.local_head_num = local_head_num
        self.global_head_num = global_head_num
        self.local_hidden_units = local_hidden_units
        self.global_hidden_units = global_hidden_units
        self.local_inter_size = local_inter_size

        self.int8_mode = int8_mode
        self.enable_int8_weights = enable_int8_weights
        
        self.use_pybind11 = use_pybind11

        if isinstance(weights_data_type, str):
            try:
                weights_data_type = {
                    "fp16": np.float16,
                    "fp32": np.float32,
                    "float16": np.float16,
                    "float32": np.float32,
                }[weights_data_type]
            except KeyError:
                raise ValueError(f"Don't know how to interpret weights_data_type: {weights_data_type}")

        assert weights_data_type in [np.float32, np.float16]
        self.weights_data_type = weights_data_type
        self.inference_data_type = str_type_map[inference_data_type]

        self.w = []
        self.int8_w = []
        self.scale = []
        # Transformer blocks
        self.w.extend([torch.zeros(global_hidden_units, dtype=self.inference_data_type)] * layer_num)                           # pre_layernorm_weights.beta
        self.w.extend([torch.zeros(global_hidden_units, dtype=self.inference_data_type)] * layer_num)                           # pre_layernorm_weights.gamma
        self.w.extend([torch.zeros(global_hidden_units, local_hidden_units * 3, dtype=self.inference_data_type)] * layer_num)   # self_attention_weights.query_weight.kernel
        self.w.extend([torch.zeros(local_hidden_units * 3, dtype=self.inference_data_type)] * layer_num)                        # self_attention_weights.query_weight.bias
        self.w.extend([torch.zeros(local_hidden_units, global_hidden_units, dtype=self.inference_data_type)] * layer_num)       # self_attention_weights.attention_output_weight.kernel
        self.w.extend([torch.zeros(global_hidden_units, dtype=self.inference_data_type) if not use_gptj_residual else torch.empty(0)] * layer_num)
                                                                                                # self_attention_weights.attention_output_weight.bias
        
        self.w.extend([torch.zeros(global_hidden_units, local_inter_size, dtype=self.inference_data_type)] * layer_num)         # ffn_weights.intermediate_weight.kernel
        self.w.extend([torch.zeros(local_inter_size, dtype=self.inference_data_type)] * layer_num)                              # ffn_weights.intermediate_weight.bias
        self.w.extend([torch.zeros(local_inter_size, global_hidden_units, dtype=self.inference_data_type)] * layer_num)         # ffn_weights.output_weight.kernel
        self.w.extend([torch.zeros(global_hidden_units, dtype=self.inference_data_type)] * layer_num)                           # ffn_weights.output_weight.bias
        
        self.w.extend([torch.zeros(global_hidden_units, dtype=self.inference_data_type)] * layer_num)                           # post_attention_layernorm_weights.beta
        self.w.extend([torch.zeros(global_hidden_units, dtype=self.inference_data_type)] * layer_num)                           # post_attention_layernorm_weights.gamma

        # After Transformer blocks
        self.w.append(torch.zeros(vocab_size, global_hidden_units, dtype=self.inference_data_type))                             # pre_decoder_embedding_table
        self.w.append(torch.zeros(global_hidden_units, dtype=self.inference_data_type))                                         # post_decoder_layernorm.beta
        self.w.append(torch.zeros(global_hidden_units, dtype=self.inference_data_type))                                         # post_decoder_layernorm.gamma
        self.w.append(torch.zeros(vocab_size, global_hidden_units, dtype=self.inference_data_type))                             # post_decoder_embedding.kernel

        # Initialization
        # self._map(lambda w: torch.nn.init.normal_(w, mean=0., std=0.01))

        if (self.int8_mode != 0):
            self.int8_w.extend([torch.zeros(global_hidden_units, local_hidden_units *
                               3, dtype=torch.int8)] * layer_num)   # self_int8_kernel
            self.scale.extend([torch.zeros(local_hidden_units * 3, dtype=torch.float)] * layer_num)   # self_scale
            self.int8_w.extend([torch.zeros(local_hidden_units, global_hidden_units, dtype=torch.int8)]
                               * layer_num)   # self_output_int8_kernel
            self.scale.extend([torch.zeros(global_hidden_units, dtype=torch.float)] * layer_num)   # self_output_scale
            self.int8_w.extend([torch.zeros(global_hidden_units, local_inter_size,
                               dtype=torch.int8)] * layer_num)   # ffn_int8_kernel1
            self.scale.extend([torch.zeros(local_inter_size, dtype=torch.float)] * layer_num)   # ffn_scale1
            self.int8_w.extend([torch.zeros(local_inter_size, global_hidden_units,
                               dtype=torch.int8)] * layer_num)   # ffn_int8_kernel2
            self.scale.extend([torch.zeros(global_hidden_units, dtype=torch.float)] * layer_num)   # ffn_scale2

            if enable_int8_weights:
                for i in range(layer_num):
                    self.w[2 * layer_num + i] = torch.empty(0).to(self.inference_data_type)
                    self.w[4 * layer_num + i] = torch.empty(0).to(self.inference_data_type)
                    self.w[6 * layer_num + i] = torch.empty(0).to(self.inference_data_type)
                    self.w[8 * layer_num + i] = torch.empty(0).to(self.inference_data_type)

    def __getitem__(self, idx):
        return self.w[idx]

    def __setitem__(self, idx, val):
        self.w[idx] = val

    def __len__(self):
        return len(self.w)

    def _map(self, func):
        for i in range(len(self.w)):
            if isinstance(self.w[i], list):
                for j in range(len(self.w[i])):
                    self.w[i][j] = func(self.w[i][j])
            else:
                self.w[i] = func(self.w[i])

    def _map_int8(self, func):
        for i in range(len(self.int8_w)):
            if isinstance(self.int8_w[i], list):
                for j in range(len(self.int8_w[i])):
                    self.int8_w[i][j] = func(self.int8_w[i][j])

            else:
                self.int8_w[i] = func(self.int8_w[i])
        for i in range(len(self.scale)):
            if isinstance(self.scale[i], list):
                for j in range(len(self.scale[i])):
                    self.scale[i][j] = func(self.scale[i][j])

            else:
                self.scale[i] = func(self.scale[i])

    def _map_int8_scales(self, func):
        for i in range(len(self.scale)):
            if isinstance(self.scale[i], list):
                for j in range(len(self.scale[i])):
                    self.scale[i][j] = func(self.scale[i][j])

            else:
                self.scale[i] = func(self.scale[i])

    def load(self, ckpt_path, tensor_para_rank, pipeline_para_rank):
        
        if not os.path.exists(ckpt_path):
            return False
        w = []

        type_map = {np.float32: torch.float32, np.float16: torch.float16}
        # Load
        def is_load(i):
            return i >= self.layers_per_device * pipeline_para_rank and i < self.layers_per_device * (pipeline_para_rank + 1)

        file_names = ["input_layernorm.bias", 
                      "input_layernorm.weight", 
                      "attention.query_key_value.weight.%d" % tensor_para_rank if not self.enable_int8_weights else None,
                      "attention.query_key_value.bias.%d" % tensor_para_rank,
                      "attention.dense.weight.%d" % tensor_para_rank if not self.enable_int8_weights else None,
                      "attention.dense.bias" if not self.use_gptj_residual else None,
                      "mlp.dense_h_to_4h.weight.%d" % tensor_para_rank if not self.enable_int8_weights else None,
                      "mlp.dense_h_to_4h.bias.%d" % tensor_para_rank,
                      "mlp.dense_4h_to_h.weight.%d" % tensor_para_rank if not self.enable_int8_weights else None,
                      "mlp.attention.bias.sum" if self.use_gptj_residual else "mlp.dense_4h_to_h.bias",
                      "post_attention_layernorm.bias",
                      "post_attention_layernorm.weight"]

        for file_name in file_names:
            for i in range(self.layer_num):
                if file_name is not None and is_load(i):
                    w.append(torch.from_numpy(np.fromfile(
                                "%s/model.layers.%d.%s.bin" % (ckpt_path, i, file_name),
                                dtype=self.weights_data_type)).to(self.inference_data_type))
                else:
                    w.append(torch.empty(0).to(self.inference_data_type))

        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.wte.bin", dtype=self.weights_data_type)).to(self.inference_data_type))
        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.final_layernorm.weight.bin", dtype=self.weights_data_type)).to(self.inference_data_type))
        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.final_layernorm.bias.bin", dtype=self.weights_data_type)).to(self.inference_data_type))
        w.append(torch.from_numpy(np.fromfile(ckpt_path + "/model.lm_head.weight.bin", dtype=self.weights_data_type)).to(self.inference_data_type))

        try:
            for i in range(len(w)):
                if w[i].nelement() > 0:
                    self.w[i] = w[i].reshape(self.w[i].shape)
                else:
                    self.w[i] = w[i]

        except RuntimeError:
            raise RuntimeError(
                f"head_num, size_per_head, vocab_size, and max_seq_len must be the same as the ones during training "
                f"(idx: {i} expected shape: {self.w[i].shape} got shape: {w[i].shape})."
            )

        # transpose calibrate quantize the kernel
        layer_num = self.layer_num
        if self.int8_mode != 0:
            for i in range(layer_num):
                if not self.enable_int8_weights:
                    self.int8_w[i + 0 * layer_num], self.scale[i + 0 *
                                                            layer_num] = self.weight_transpose_calibrate_quantize(self.w[2 * layer_num + i])
                    self.int8_w[i + 1 * layer_num], self.scale[i + 1 *
                                                            layer_num] = self.weight_transpose_calibrate_quantize(self.w[4 * layer_num + i])
                    self.int8_w[i + 2 * layer_num], self.scale[i + 2 *
                                                            layer_num] = self.weight_transpose_calibrate_quantize(self.w[6 * layer_num + i])
                    self.int8_w[i + 3 * layer_num], self.scale[i + 3 *
                                                            layer_num] = self.weight_transpose_calibrate_quantize(self.w[8 * layer_num + i])

                    # We clear the original weights since they are no longer needed
                    if self.int8_mode == 1:
                        self.w[2 * layer_num + i] = torch.empty(0).to(self.inference_data_type)
                        self.w[4 * layer_num + i] = torch.empty(0).to(self.inference_data_type)
                        self.w[6 * layer_num + i] = torch.empty(0).to(self.inference_data_type)
                        self.w[8 * layer_num + i] = torch.empty(0).to(self.inference_data_type)
                else:
                    quanted_weight_file_names = [
                        "attention.query_key_value.weight.%d" % tensor_para_rank,
                        "attention.dense.weight.%d" % tensor_para_rank,
                        "mlp.dense_h_to_4h.weight.%d" % tensor_para_rank,
                        "mlp.dense_4h_to_h.weight.%d" % tensor_para_rank,
                    ]
                    for quanted_weight_idx, file_name in enumerate(quanted_weight_file_names):
                        self.int8_w[i + quanted_weight_idx * layer_num] = torch.from_numpy(np.fromfile(
                            "%s/model.layers.%d.%s.q.bin" % (ckpt_path, i, file_name), dtype=np.int8)).to(torch.int8)
                        self.scale[i + quanted_weight_idx * layer_num] = torch.from_numpy(np.fromfile(
                            "%s/model.layers.%d.%s.s.bin" % (ckpt_path, i, file_name), dtype=self.weights_data_type)).to(self.inference_data_type)
        return True


class GptNeoX(nn.Module):
    def __init__(self,
                 head_num, size_per_head,
                 vocab_size, rotary_embedding_dim, 
                 start_id, end_id, layer_num,
                 max_seq_len,
                 tensor_para_size, pipeline_para_size,
                 use_gptj_residual,
                 lib_path,
                 int8_mode=0, inference_data_type: str = "fp16",
                 weights_data_type: np.dtype = np.float32,
                 enable_int8_weights=False,
                 use_pybind11=False):
        super().__init__()
        self.head_num = head_num
        self.size_per_head = size_per_head
        self.inter_size = 4 * self.head_num * self.size_per_head
        self.vocab_size = vocab_size
        self.rotary_embedding_dim = rotary_embedding_dim
        self.start_id = start_id
        self.end_id = end_id
        self.max_seq_len = max_seq_len
        self.layer_num = layer_num
        self.use_gptj_residual = use_gptj_residual
        self.int8_mode = int8_mode

        self.enable_int8_weights = enable_int8_weights

        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.build_model = False
        self.weights_data_type = weights_data_type
        self.inference_data_type = inference_data_type
        
        self.use_pybind11 = use_pybind11

        assert torch.cuda.is_available(), "CUDA is required for this model."

        assert head_num % tensor_para_size == 0, "head_num must be a multiple of tensor_para_size."
        assert layer_num % pipeline_para_size == 0, "layer_num must be a multiple of pipeline_para_size."

        if not self.use_pybind11:
            # Load the C++ model into Pytorch model.
            torch.classes.load_library(os.path.abspath(lib_path))
            self.GptNeoXOp = torch.classes.FasterTransformer.GptNeoXOp
        else:
            sys.path.append(os.path.abspath(lib_path))
            import libth_gptneox
            self.GptNeoXOp = libth_gptneox.GptNeoXOp
        
        # Prepare weights
        self.weights = GptNeoXWeights(head_num, size_per_head, layer_num, vocab_size,
                                      max_seq_len, tensor_para_size, pipeline_para_size, use_gptj_residual, int8_mode=int8_mode,
                                      weights_data_type=weights_data_type, inference_data_type=inference_data_type,
                                      use_pybind11=use_pybind11, enable_int8_weights=enable_int8_weights)
        
        # Prepare for tensor/pipeline parallel
        try:
            if not self.use_pybind11:
                dist.init_process_group(backend='mpi')
            else:
                try:
                    dist.init_process_group(backend='nccl')
                except:
                    # nccl init got exception when world size is 1
                    # but we need a comm object even it's useless
                    dist.init_process_group(backend='mpi')
        except:
            print("[INFO] WARNING: Have initialized the process group")
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.device_count = torch.cuda.device_count()
        self.device = self.rank % self.device_count
        torch.cuda.set_device(self.device)

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        # print(tensor_para_size * pipeline_para_size)
        assert self.world_size == tensor_para_size * pipeline_para_size, "tensor_para_size * pipeline_para_size must be equal to world_size."

        self.tensor_para_rank = self.rank % self.tensor_para_size
        self.pipeline_para_rank = self.rank // self.tensor_para_size

        # Create and copy model to the device.
        # self.cuda()

    def load(self, ckpt_path):
        is_load = self.weights.load(ckpt_path, tensor_para_rank=self.tensor_para_rank,
                                    pipeline_para_rank=self.pipeline_para_rank)
        self.cuda()
        return is_load

    def half(self):
        self.weights._map(lambda w: w.half())
        self.cuda()

    def cuda(self):
        self.weights._map(lambda w: w.cuda(self.device))

        if self.int8_mode != 0:
            self.weights._map_int8(lambda w: w.cuda(self.device))

        if self.build_model:
            del self.model
            self.build_model = False
        
        if self.use_pybind11:
            try:
                comm = dist.distributed_c10d._get_default_group()
            except Exception as _:
                comm = dist.distributed_c10d._default_pg
            if comm is None:
                raise RuntimeError("Unsupported PyTorch version")
            self.model = self.GptNeoXOp(comm, self.rank, self.head_num, self.size_per_head, self.inter_size,
                                        self.layer_num, self.vocab_size, self.rotary_embedding_dim, 
                                        self.start_id, self.end_id, self.tensor_para_size, self.pipeline_para_size, self.int8_mode,
                                        self.max_seq_len, self.use_gptj_residual, self.weights.w, self.weights.int8_w, self.weights.scale)
        else:
            self.model = self.GptNeoXOp(self.head_num, self.size_per_head, self.inter_size,
                                        self.layer_num, self.vocab_size, self.rotary_embedding_dim, 
                                        self.start_id, self.end_id, self.tensor_para_size, self.pipeline_para_size, self.int8_mode,
                                        self.max_seq_len, self.use_gptj_residual, self.weights.w, self.weights.int8_w, self.weights.scale)

        self.build_model = True

    def forward(self,
                start_ids: torch.Tensor,
                start_lengths: torch.Tensor,
                output_len,
                beam_width = 1,
                top_k: torch.Tensor = None,
                top_p: torch.Tensor = None,
                beam_search_diversity_rate: torch.Tensor = None,
                temperature: torch.Tensor = None,
                len_penalty: torch.Tensor = None,
                repetition_penalty: torch.Tensor = None,
                random_seed: torch.Tensor = None,
                stop_words_list: torch.Tensor = None,
                optional_last_tokens: torch.Tensor = None,
                return_output_length = False,
                return_cum_log_probs = 0,
                callback = None):
        if not self.build_model:
            self.cuda()
        input_len = start_ids.size(1)
        assert input_len > 0, "input len must be larger than zero. For an unconditional case, use start_id as the first token."

        # Inputs to device
        input_ids = start_ids.cuda(self.device)
        input_lengths = start_lengths.cuda(self.device)
        if stop_words_list is not None:
            stop_words_list = stop_words_list.cuda(self.device)
        if optional_last_tokens is not None:
            optional_last_tokens = optional_last_tokens.cuda(self.device)
        # outputs: output_ids, output_lengths, output_cum_log_probs (optional)
        outputs = self.model.forward(input_ids,
                                     input_lengths,
                                     output_len,
                                     beam_width, # optional, can be None
                                     top_k, # optional, can be None
                                     top_p, # optional, can be None
                                     beam_search_diversity_rate, # optional, can be None
                                     temperature, # optional, can be None
                                     len_penalty, # optional, can be None
                                     repetition_penalty, # optional, can be None
                                     random_seed, # optional, can be None
                                     stop_words_list, # optional, can be None
                                     optional_last_tokens, # optional, can be None
                                     return_cum_log_probs, # optional, can be None
                                     callback) # optional, can be None

        if return_cum_log_probs == 0:
            output_ids, output_lengths = outputs
        else:
            output_ids, output_lengths, output_cum_log_probs = outputs
        if return_output_length:
            if return_cum_log_probs > 0:
                return output_ids, output_lengths, output_cum_log_probs
            else:
                return output_ids, output_lengths
        else:
            return output_ids

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

#  __  __  ___  ____  _____ _       _   _ ____    _    ____ _____ 
# |  \/  |/ _ \|  _ \| ____| |     | | | / ___|  / \  / ___| ____|
# | |\/| | | | | | | |  _| | |     | | | \___ \ / _ \| |  _|  _|  
# | |  | | |_| | |_| | |___| |___  | |_| |___) / ___ \ |_| | |___ 
# |_|  |_|\___/|____/|_____|_____|  \___/|____/_/   \_\____|_____|

def init_model_and_tokenizer(lib_path, ckpt_path, tokenizer_file_path, tensor_parallel, int8_mode=0, enable_int8_weights=False, trie_needed=False, end_id=None):

    config = ConfigParser()
    config.read(os.path.join(ckpt_path, "config.ini"))
    head_num = int(config.get('gptneox', 'head_num'))
    size_per_head = int(config.get('gptneox', 'size_per_head'))
    vocab_size = int(config.get('gptneox', 'vocab_size'))
    layer_num = int(config.get('gptneox', 'num_layer'))
    rotary_embedding = int(config.get('gptneox', 'rotary_embedding'))
    start_id = int(config.get('gptneox', 'start_id'))
    end_id = int(config.get('gptneox', 'end_id')) if end_id is None else end_id
    use_gptj_residual = (config.get('gptneox', 'use_gptj_residual') == "1")
    weight_data_type = config.get('gptneox', 'weight_data_type')

    rank = dist.get_rank() if dist.is_initialized() else 0
    device_count = dist.get_world_size() if dist.is_initialized() else 1
    device = rank % device_count
    torch.cuda.set_device(device)
    device = torch.cuda.current_device()

    # sentencepiece needed
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_file_path)

    # Prepare model.
    gpt = GptNeoX(head_num, size_per_head, vocab_size, rotary_embedding,
                  start_id, end_id, layer_num, 
                  max_seq_len=1024, # deprecated
                  tensor_para_size=tensor_parallel, 
                  pipeline_para_size=1, 
                  use_gptj_residual=use_gptj_residual, 
                  lib_path=lib_path, 
                  int8_mode=int8_mode,
                  inference_data_type="fp16", 
                  weights_data_type=weight_data_type, 
                  use_pybind11=True,
                  enable_int8_weights=enable_int8_weights)
    if not gpt.load(ckpt_path=ckpt_path):
        print("[WARNING] Checkpoint file not found. Model loading is skipped.")
    
    trie = Trie(tokenizer.get_vocab())
    
    if not trie_needed:
        return gpt, tokenizer
    else:
        return gpt, tokenizer, trie 


def generate(gpt: GptNeoX, tokenizer, texts: List[str], output_len, beam_width,
             top_k=None, top_p=None, beam_search_diversity_rate=None,
             temperature=None, len_penalty=None, repetition_penalty=None, random_seed=None, input_ids_list=None,
             callback=None, stop_words_list: List[List[str]]=None, 
             last_token_list: List[str]=None, trie: Trie=None):

    def __c(_para, __type, __tensor_init):
        if _para is None:
            return None
        elif isinstance(_para, __type):
            return __tensor_init([_para])
        elif isinstance(_para, list):
            return __tensor_init(_para)
        else:
            raise RuntimeError("type don't match with %s" % str(__type))

    def _ci(_para):
        return __c(_para, int, torch.IntTensor)

    def _cl(_para):
        return __c(_para, int, torch.LongTensor)

    def _cf(_para):
        return __c(_para, float, torch.FloatTensor)

    assert texts is not None or input_ids_list is not None

    if texts is not None:
        input_ids_list = [torch.IntTensor(tokenizer.encode(text)) for text in texts]
    else:
        input_ids_list = [torch.IntTensor(_li) for _li in input_ids_list]

    input_lengths_list = [ids.size(-1) for ids in input_ids_list]

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=gpt.end_id)
    input_lengths = torch.IntTensor(input_lengths_list)
    
    if stop_words_list is not None:
        stop_words_list = to_word_list_format(stop_words_list, tokenizer)

    if last_token_list is not None:
        assert trie is not None, "trie is None, can't select last token"
        optional_last_tokens = []
        for last_token in last_token_list:
            option_token_ids = []
            trie.printAutoSuggestions(last_token, option_token_ids)
            option_token_ids = [t_i for t_s, t_i in option_token_ids]
            if len(option_token_ids) == 0:
                option_token_ids.append(gpt.end_id)
            optional_last_tokens.append(torch.IntTensor(option_token_ids))
        optional_last_tokens = pad_sequence(optional_last_tokens, batch_first=True, padding_value=-1)
    else:
        optional_last_tokens = None

    top_k                      = _ci(top_k)
    top_p                      = _cf(top_p)
    beam_search_diversity_rate = _cf(beam_search_diversity_rate)
    temperature                = _cf(temperature)
    len_penalty                = _cf(len_penalty)
    repetition_penalty         = _cf(repetition_penalty)
    random_seed                = _cl(random_seed)

    start_time = time.time()
    with torch.no_grad():
        tokens_batch, _, output_cum_log_probs = gpt(
            start_ids=input_ids,
            start_lengths=input_lengths,
            output_len=output_len,
            beam_width=beam_width,
            top_k=top_k,
            top_p=top_p,
            beam_search_diversity_rate=beam_search_diversity_rate,
            temperature=temperature,
            len_penalty=len_penalty,
            repetition_penalty=repetition_penalty,
            random_seed=random_seed,
            stop_words_list=stop_words_list,
            optional_last_tokens=optional_last_tokens,
            return_output_length=True,
            return_cum_log_probs=1,
            callback=callback)
    latency = time.time() - start_time
    tokens_batch = tokens_batch.detach().cpu().tolist()
    output_cum_log_probs = output_cum_log_probs.detach().cpu().tolist()

    outputs = []
    output_lengths = []
    for beam_ids, input_length in zip(tokens_batch, input_lengths_list):
        beam_outputs = []
        beam_output_lengths = []
        for ids in beam_ids:
            ids_remove_padding = []
            for __id in ids[input_length:]:
                if __id == gpt.end_id:
                    break
                ids_remove_padding.append(__id)
            output = tokenizer.decode(ids_remove_padding)
            if len(output) > 0 and is_garbage(ord(output[-1])):
                output = output[: -1]
            beam_outputs.append(output)
            beam_output_lengths.append(len(ids_remove_padding))
        outputs.append(beam_outputs)
        output_lengths.append(beam_output_lengths)

    return outputs, output_lengths, output_cum_log_probs, latency


#  __  __    _    ___ _   _   _   _    _    _   _ ____  _     _____ ____  
# |  \/  |  / \  |_ _| \ | | | | | |  / \  | \ | |  _ \| |   | ____|  _ \ 
# | |\/| | / _ \  | ||  \| | | |_| | / _ \ |  \| | | | | |   |  _| | |_) |
# | |  | |/ ___ \ | || |\  | |  _  |/ ___ \| |\  | |_| | |___| |___|  _ < 
# |_|  |_/_/   \_\___|_| \_| |_| |_/_/   \_\_| \_|____/|_____|_____|_| \_\

def get_data_package(request_dict, default_random_seed):
    def gather_batched_parameter(key, default_value=None):
        if default_value is None and all(key not in prompt_dict for prompt_dict in prompt_dicts):
            return None
        if default_value is None and any(key not in prompt_dict for prompt_dict in prompt_dicts):
            raise RuntimeError("default_value is None while %s is also None." % key)
        return [prompt_dict[key] if key in prompt_dict else default_value for prompt_dict in prompt_dicts]

    output_len                  = request_dict["out_seq_length"]
    beam_width                  = request_dict["beam_width"] if "beam_width" in request_dict else 1

    prompt_dicts = request_dict["prompts"]
    prompt_strs = []
    for prompt_dict in prompt_dicts:        
        assert isinstance(prompt_dict["prompt"], str)
        prompt_strs.append(prompt_dict["prompt"])

    stop_words_list            = gather_batched_parameter("stop_words")
    last_token_list            = gather_batched_parameter("last_token")
    # https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/text_generation#transformers.GenerationConfig
    top_k                      = gather_batched_parameter("top_k"                     , default_value=50)
    # https://github.com/NVIDIA/FasterTransformer/blob/main/examples/pytorch/gptneox/gptneox_example.py
    top_p                      = gather_batched_parameter("top_p"                     , default_value=0.)
    beam_search_diversity_rate = gather_batched_parameter("beam_search_diversity_rate", default_value=0.)
    temperature                = gather_batched_parameter("temperature"               , default_value=1.)
    len_penalty                = gather_batched_parameter("len_penalty"               , default_value=0.)
    repetition_penalty         = gather_batched_parameter("repetition_penalty"        , default_value=1.)
    random_seed                = gather_batched_parameter("random_seed"               , default_value=default_random_seed)

    data_package = {"texts": prompt_strs, "output_len": output_len, "beam_width": beam_width, 
                "top_k": top_k, "top_p": top_p, "beam_search_diversity_rate": beam_search_diversity_rate,
                "temperature": temperature, "len_penalty": len_penalty, "repetition_penalty": repetition_penalty,
                "random_seed": random_seed, "stop_words_list": stop_words_list, "last_token_list": last_token_list}
    return data_package

class CodeFuseHandler(object):
    def __init__(self, lib_path, ckpt_path, tokenizer_path, int8_mode, enable_int8_weights, world_size=1, local_rank=0, end_id=None):
        self.local_rank = local_rank
        self.world_size = world_size

        logging.info("start init rank: %d" % local_rank)
        try:
            model, tokenizer, trie = init_model_and_tokenizer(lib_path=lib_path, 
                                                              ckpt_path=ckpt_path, 
                                                              tokenizer_file_path=tokenizer_path, 
                                                              tensor_parallel=world_size, 
                                                              int8_mode=int8_mode,
                                                              enable_int8_weights=enable_int8_weights,
                                                              trie_needed=True,
                                                              end_id=end_id)
            self.model = model
            self.tokenizer = tokenizer
            self.trie = trie
            generate(model, tokenizer, ["demo"], 2, 1)
        except BaseException as err:
            logging.exception(err)

    def predict(self, request_dict, trace_id):
        logging.info("%s request: %s" % (trace_id, json.dumps(request_dict, ensure_ascii=False)))

        try:
            default_random_seed = random.randint(0, 1048576)
            if self.world_size > 1:
                default_random_seed_tensor = torch.IntTensor([default_random_seed]).to("cuda")
                dist.broadcast(default_random_seed_tensor, src=0)
                default_random_seed = default_random_seed_tensor.cpu().tolist()[0]

            use_callback = "stream" in request_dict and request_dict["stream"]

            data_package = get_data_package(request_dict, default_random_seed)

            batch_size = len(data_package["texts"])
            beam_width = data_package["beam_width"]
            random_seed = data_package["random_seed"]

            use_callback = use_callback and self.local_rank == 0

            if use_callback:
                try:
                    convertors = [[token_stream_2_str_stream_convertor(self.model.end_id, self.tokenizer, self.local_rank)
                                for _j in range(beam_width)] for _i in range(batch_size)]
                    def callback(message_dict: dict):
                        try:
                            last_tokens = message_dict["last_tokens"]
                            for batch_idx in range(batch_size):
                                for beam_idx in range(beam_width):
                                    convertor = convertors[batch_idx][beam_idx]
                                    last_token = last_tokens[batch_idx][beam_idx]
                                    convertor.append_token(last_token)
                        except BaseException as err:
                            logging.error("callback error: %s" % str(err))

                except BaseException as err:
                    use_callback = False
                    logging.error("call back init error: %s" % str(err))
            result, lengths, cum_log_probs, latency = generate(self.model, self.tokenizer, 
                                                               trie=self.trie,
                                                               callback=callback if use_callback else None, 
                                                               **data_package)

            if use_callback:
                try:
                    for batch_idx in range(batch_size):
                        for beam_idx in range(beam_width):
                            convertor = convertors[batch_idx][beam_idx]
                            convertor.append_token(self.model.end_id)
                except BaseException as err:
                    logging.error("append end_id error: %s" % str(err))

            response = {"latency":          latency, 
                        "random_seed":      random_seed, 
                        "generated_code":   result, 
                        "length":           lengths, 
                        "cum_log_prob":     cum_log_probs}
            response_str = json.dumps(response, ensure_ascii=False)
            logging.info("%s response: %s" % (trace_id, response_str))
            resultMap = {"res": response_str}
            resultCode = 0       # 0表示成功，其它为失败
            errorMessage = "ok"  # errorMessage为predict函数对外透出的信息

            return (resultCode, errorMessage, resultMap)
        except BaseException as _:
            resultMap = {"res": ""}
            resultCode = 1
            errorMessage = traceback.format_exc()

            return (resultCode, errorMessage, resultMap)


# 用于调试UserHandler类的功能
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--lib_path', type=str, default="../build/lib")
    parser.add_argument('--ckpt_path', type=str, default="../models/codefuse/fastertransformer/1-gpu")
    parser.add_argument('--tokenizer_path', type=str, default="../models/codefuse/transformers")
    parser.add_argument('--int8_mode', type=int, default=0)
    parser.add_argument('--enable_int8_weights', type=int, default=0)
    parser.add_argument('--input_file', type=str, default="../input_demo.jsonl")
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    world_size = args.world_size
    local_rank = args.local_rank
    lib_path = args.lib_path
    ckpt_path = args.ckpt_path
    tokenizer_path = args.tokenizer_path
    int8_mode = args.int8_mode
    enable_int8_weights = args.enable_int8_weights
    input_file = args.input_file

    logging_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if world_size == 1:
        local_rank = 0
    else:
        dist.init_process_group(backend='nccl')
        if local_rank is None:
            local_rank = dist.get_rank()

    with open(input_file) as i_f:
        request_dicts = [json.loads(l.strip()) for l in i_f.readlines()]

    codefuse_handler = CodeFuseHandler(lib_path=lib_path, 
                                       ckpt_path=ckpt_path, 
                                       tokenizer_path=tokenizer_path, 
                                       int8_mode=int8_mode, 
                                       enable_int8_weights=enable_int8_weights, 
                                       world_size=world_size, 
                                       local_rank=local_rank)

    print("local_rank:", local_rank)

    for request_dict in request_dicts:
        resultCode, errorMessage, resultMap = codefuse_handler.predict(request_dict, "test")

        if local_rank == 0:
            if resultCode == 0:
                print("#" * 50)
                print("=" * 50)
                print("- request\n%s" % json.dumps(request_dict, ensure_ascii=False, indent=4))
                for batch_idx, beam_result in enumerate(json.loads(resultMap["res"])["generated_code"]):
                    for beam_idx, result in enumerate(beam_result):
                        print(("=" if beam_idx == 0 else "-") * 50)
                        print("- batch_idx %d" % batch_idx)
                        print("- beam_idx %d" % beam_idx)
                        print("- result\n%s" % result)
                print("=" * 50)
                print("- latency %f" % json.loads(resultMap["res"])["latency"])
                print("- random_seed %s" % str(json.loads(resultMap["res"])["random_seed"]))
                print("=" * 50)
            else:
                print(errorMessage)
                raise RuntimeError()