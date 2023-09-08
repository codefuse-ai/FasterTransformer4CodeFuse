import argparse
import os
import sys
import shutil
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from configparser import ConfigParser
import numpy as np
from tqdm import tqdm
import torch

def _quant(in_dir, out_dir, li, fn, rk, shape):

    file_name       = "model.layers.%d.%s.%d.bin"   % (li, fn, rk) 
    int8_file_name  = "model.layers.%d.%s.%d.q.bin" % (li, fn, rk) 
    scale_file_name = "model.layers.%d.%s.%d.s.bin" % (li, fn, rk)

    w = torch.from_numpy(np.fromfile(os.path.join(in_dir, file_name), dtype=weight_data_type)).to(inference_data_type).reshape(shape)
    int8_w, scale = weight_transpose_calibrate_quantize(w)
    int8_w.numpy().astype(np.int8).tofile(os.path.join(out_dir, int8_file_name))
    scale.numpy().astype(weight_data_type).tofile(os.path.join(out_dir, scale_file_name))
    os.remove(os.path.join(out_dir, file_name))
    print("%s quanted" % file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--lib_path', type=str, required=True)
    parser.add_argument('--tensor_para_size', type=int, required=True)
    parser.add_argument('--process', type=int, default=20)
    parser.add_argument('--use_gptj_residual', action='store_true')
    parser.add_argument('--inference_data_type', '--data_type', type=str, choices=['fp32', 'fp16'], default='fp16')
    args = parser.parse_args()
    
    in_dir              = args.in_dir
    out_dir             = args.out_dir
    lib_path            = args.lib_path
    tensor_para_size    = args.tensor_para_size
    process             = args.process
    use_gptj_residual   = args.use_gptj_residual
    inference_data_type = args.inference_data_type

    assert os.path.exists(lib_path)
    sys.path.append(os.path.abspath(os.path.dirname(lib_path)))
    import libth_common
    weight_transpose_calibrate_quantize = libth_common.symmetric_quantize_last_axis_of_batched_matrix_int8
    
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for file_name in tqdm(os.listdir(in_dir)):
        shutil.copy(os.path.join(in_dir, file_name), os.path.join(out_dir, file_name))

    config = ConfigParser()
    config.read(os.path.join(in_dir, "config.ini"))
    head_num = int(config.get('gptneox', 'head_num'))
    size_per_head = int(config.get('gptneox', 'size_per_head'))
    vocab_size = int(config.get('gptneox', 'vocab_size'))
    layer_num = int(config.get('gptneox', 'num_layer'))
    rotary_embedding = int(config.get('gptneox', 'rotary_embedding'))
    start_id = int(config.get('gptneox', 'start_id'))
    end_id = int(config.get('gptneox', 'end_id'))
    use_gptj_residual = (config.get('gptneox', 'use_gptj_residual') == "1")
    weight_data_type = config.get('gptneox', 'weight_data_type')
    
    local_head_num = head_num // tensor_para_size
    global_head_num = head_num
    local_hidden_units = local_head_num * size_per_head
    global_hidden_units = global_head_num * size_per_head
    local_inter_size = local_hidden_units * 4
    
    weight_data_type = {
        "fp16": np.float16,
        "fp32": np.float32,
        "float16": np.float16,
        "float32": np.float32,
    }[weight_data_type]
    
    inference_data_type = {
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[inference_data_type]

    quant_weight_file_names_2_shapes = {
        "attention.query_key_value.weight": (global_hidden_units, local_hidden_units * 3),
        "attention.dense.weight": (local_hidden_units, global_hidden_units),
        "mlp.dense_h_to_4h.weight": (global_hidden_units, local_inter_size),
        "mlp.dense_4h_to_h.weight": (local_inter_size, global_hidden_units)
    }

    pool_for_quant_files = multiprocessing.Pool(process)

    pool_for_quant_files.starmap(
        _quant, 
        [(in_dir, out_dir, li, fn, rk, quant_weight_file_names_2_shapes[fn]) 
         for rk in range(tensor_para_size) for fn in quant_weight_file_names_2_shapes.keys() for li in range(layer_num)]
    )

    pool_for_quant_files.close()
    pool_for_quant_files.join()