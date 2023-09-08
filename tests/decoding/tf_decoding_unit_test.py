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
import unittest
import argparse
import os
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import sys
import time
from multiprocessing import Process, Value
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../..")
from examples.tensorflow.decoding.translate_example import translate

class TestDecoding(unittest.TestCase):
    
    common_args_dict = {'batch_size' : 128,
                        'max_seq_len': 128,
                        'encoder_head_number': 8,
                        'encoder_size_per_head': 64,
                        'decoder_head_number': 8,
                        'decoder_size_per_head': 64,
                        'encoder_num_layer': 6,
                        'decoder_num_layer': 6,
                        'beam_search_diversity_rate': 0.0,
                        'sampling_topk': 1,
                        'sampling_topp': 0.0,
                        'source_vocabulary': "../examples/tensorflow/decoding/utils/translation/wmtende.vocab",
                        'target_vocabulary': "../examples/tensorflow/decoding/utils/translation/wmtende.vocab",
                        'source': "../examples/tensorflow/decoding/utils/translation/test.en",
                        'target': "../examples/tensorflow/decoding/utils/translation/test.de",
                        "remove_padding": "True",
                        "max_iteration": 10,
                        }

    def check_result(self, beam_width, datatype, test_time, topk=4, topp=0.0, batch_size=-1,
                     decoder_bleu_score_threshold=None, decoding_bleu_score_threshold=None):
        p = Process(target=self.run_translate, args=(beam_width, datatype, test_time, topk, topp,
                                                     batch_size, decoder_bleu_score_threshold, decoding_bleu_score_threshold))
        p.start()
        p.join()
    
    def run_translate(self, beam_width, datatype, test_time, topk=4, topp=0.0, batch_size=-1,
                      decoder_bleu_score_threshold=None, decoding_bleu_score_threshold=None):
        args_dict = copy.deepcopy(self.common_args_dict)
        args_dict['beam_width'] = beam_width
        args_dict['data_type'] = datatype
        args_dict['test_time'] = test_time
        args_dict['sampling_topk'] = topk
        args_dict['sampling_topp'] = topp
        args_dict['model_dir'] = "../translation/ckpt"
        if batch_size != -1:
            args_dict['batch_size'] = batch_size

        tf.reset_default_graph()

        translation_result_list = translate(args_dict)
        #  translation_result_list[0] is warmup, skip it
        op_decoder_bleu_score = translation_result_list[1].bleu_score.score
        op_decoding_bleu_score = translation_result_list[2].bleu_score.score
        if decoder_bleu_score_threshold != None:
            self.assertTrue(op_decoder_bleu_score >= decoder_bleu_score_threshold)
        if decoding_bleu_score_threshold != None:
            self.assertTrue(op_decoding_bleu_score >= decoding_bleu_score_threshold)
        sys.stdout.flush()

    def test_decoding_beamsearch_fp32(self):
        os.system("./bin/decoding_gemm 32 4 8 64 2048 32001 128 512 0 > .tmp.gemm.log && cat gemm_config.in")
        self.check_result(4, 'fp32', '12', batch_size=32, decoder_bleu_score_threshold=37.0, decoding_bleu_score_threshold=37.0)
        
    def test_decoding_beamsearch_fp16(self):
        os.system("./bin/decoding_gemm 32 4 8 64 2048 32001 128 512 1 > .tmp.gemm.log && cat gemm_config.in")
        self.check_result(4, 'fp16', '12', batch_size=32, decoder_bleu_score_threshold=37.0, decoding_bleu_score_threshold=37.0)
        
    def test_decoding_beamsearch_fp32_2(self):
        os.system("./bin/decoding_gemm 16 32 8 64 2048 32001 128 512 0 > .tmp.gemm.log && cat gemm_config.in")
        self.check_result(32, 'fp32', '12', batch_size=16, decoder_bleu_score_threshold=35.0, decoding_bleu_score_threshold=35.0)
        
    def test_decoding_beamsearch_fp16_2(self):
        os.system("./bin/decoding_gemm 16 32 8 64 2048 32001 128 512 1 > .tmp.gemm.log && cat gemm_config.in")
        self.check_result(32, 'fp16', '12', batch_size=16, decoder_bleu_score_threshold=35.0, decoding_bleu_score_threshold=35.0)
    
    def test_decoding_topk_sampling_fp32(self):
        os.system("./bin/decoding_gemm 128 1 8 64 2048 32001 128 512 0 > .tmp.gemm.log && cat gemm_config.in")
        self.check_result(1, 'fp32', '45', 4, 0.0, decoder_bleu_score_threshold=25.0, decoding_bleu_score_threshold=25.0)
        
    def test_decoding_topk_sampling_fp16(self):
        os.system("./bin/decoding_gemm 128 1 8 64 2048 32001 128 512 1 > .tmp.gemm.log && cat gemm_config.in")
        self.check_result(1, 'fp16', '45', 4, 0.0, decoder_bleu_score_threshold=25.0, decoding_bleu_score_threshold=25.0)

    def test_decoding_topk_sampling_fp32_2(self):
        os.system("./bin/decoding_gemm 128 1 8 64 2048 32001 128 512 0 > .tmp.gemm.log && cat gemm_config.in")
        self.check_result(1, 'fp32', '45', 64, 0.0, decoder_bleu_score_threshold=19.0, decoding_bleu_score_threshold=17.0)

    def test_decoding_topk_sampling_fp16_2(self):
        os.system("./bin/decoding_gemm 128 1 8 64 2048 32001 128 512 1 > .tmp.gemm.log && cat gemm_config.in")
        self.check_result(1, 'fp16', '45', 64, 0.0, decoder_bleu_score_threshold=19.0, decoding_bleu_score_threshold=17.0)

    def test_decoding_topp_sampling_fp32(self):
        os.system("./bin/decoding_gemm 128 1 8 64 2048 32001 128 512 0 > .tmp.gemm.log && cat gemm_config.in")
        self.check_result(1, 'fp32', '45', 0, 0.5, decoder_bleu_score_threshold=30.0, decoding_bleu_score_threshold=29.0)
        
    def test_decoding_topp_sampling_fp16(self):
        os.system("./bin/decoding_gemm 128 1 8 64 2048 32001 128 512 1 > .tmp.gemm.log && cat gemm_config.in")
        self.check_result(1, 'fp16', '45', 0, 0.5, decoder_bleu_score_threshold=30.0, decoding_bleu_score_threshold=29.0)

    def test_decoding_topp_sampling_fp32_2(self):
        os.system("./bin/decoding_gemm 128 1 8 64 2048 32001 128 512 0 > .tmp.gemm.log && cat gemm_config.in")
        self.check_result(1, 'fp32', '45', 0, 0.9, decoder_bleu_score_threshold=16.0, decoding_bleu_score_threshold=14.5)

    def test_decoding_topp_sampling_fp16_2(self):
        os.system("./bin/decoding_gemm 128 1 8 64 2048 32001 128 512 1 > .tmp.gemm.log && cat gemm_config.in")
        self.check_result(1, 'fp16', '45', 0, 0.9, decoder_bleu_score_threshold=16.0, decoding_bleu_score_threshold=14.5)

if __name__ == "__main__":
    unittest.main()
