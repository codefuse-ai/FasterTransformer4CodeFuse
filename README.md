# FasterTransformer4CodeFuse

<div align="center">
<p><img src="./assets/LOGO.png" width="100%" /></p>
<p>
    <a href="https://github.com/codefuse-ai/FasterTransformer4CodeFuse">
        <img alt="stars" src="https://img.shields.io/github/stars/codefuse-ai/FasterTransformer4CodeFuse?style=social" />
    </a>
    <a href="https://github.com/codefuse-ai/FasterTransformer4CodeFuse">
        <img alt="forks" src="https://img.shields.io/github/forks/codefuse-ai/FasterTransformer4CodeFuse?style=social" />
    </a>
    <a href="https://opensource.org/licenses/MIT">
      <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
    </a>
    <a href="https://github.com/codefuse-ai/FasterTransformer4CodeFuse/issues">
      <img alt="Open Issues" src="https://img.shields.io/github/issues-raw/codefuse-ai/FasterTransformer4CodeFuse" />
    </a>
    <br/>
</p>

  
| [**简体中文**](README_CN.md) |
</div>

## Introduce

Provide high-performance model inference, mainly supporting the CodeFuse model from Ant Group.

Compared to the [original FT](https://github.com/NVIDIA/FasterTransformer), this repo has these features:

- [x] Int8 quantization of CodeFuse model
- [x] Prompt does not require a complete word at the end
- [x] Python API
- [x] Streaming Output with Python API
- [x] Faster model load speed
- [x] Some bugfix

## Performance
> Batch size: 1
<table>
    <tr>
        <td colspan="3">Model</td>
        <td colspan="4">CodeFuse 13B</td>
    </tr>
    <tr>
        <td colspan="3">Measurements</td>
        <td colspan="4">Latency (ms)</td>
    </tr>
    <tr>
        <td colspan="3">GPU</td>
        <td colspan="2">Single A100</td>
        <td colspan="2">2 * A100 Tensor Parallelism</td>
    </tr>
    <tr>
        <td colspan="3">Data Type</td>
        <td>fp16</td>
        <td>int8</td>
        <td>fp16</td>
        <td>int8</td>
    </tr>
    <tr>
        <td rowspan="4">Input/Output Length</td>
        <td>16</td>
        <td>8</td>
        <td>160</td>
        <td>195</td>
        <td>238</td>
        <td>84</td>
    </tr>
    <tr>
        <td>64</td>
        <td>32</td>
        <td>608</td>
        <td>369</td>
        <td>373</td>
        <td>295</td>
    </tr>
    <tr>
        <td>256</td>
        <td>128</td>
        <td>2650</td>
        <td>1530</td>
        <td>1492</td>
        <td>1130</td>
    </tr>
    <tr>
        <td>1024</td>
        <td>512</td>
        <td>10776</td>
        <td>7054</td>
        <td>6786</td>
        <td>5415</td>
    </tr>
    <tr>
        <td colspan="3">Tokens Per Sec</td>
        <td>48</td>
        <td>75</td>
        <td>77</td>
        <td>98</td>
    </tr>
</table>

## Get Start

We run in the container environment: `nvcr.io/nvidia/pytorch:22.09-py3`。

#### 1. Install requirementss

```
pip install --no-cache-dir pybind11==2.6.2 transformers accelerate sentencepiece

echo "export pybind11_DIR=/opt/conda/lib/python3.8/site-packages/pybind11/share/cmake/pybind11/" >> ~/.bashrc
export pybind11_DIR=/opt/conda/lib/python3.8/site-packages/pybind11/share/cmake/pybind11/
```

#### 2. Build

```
mkdir build ; cd build
export TORCH_PYTHON_LIBRARIES=/opt/conda/lib/python3.8/site-packages/torch/lib/libtorch_python.so
cmake -DCMAKE_BUILD_TYPE=Release -DSM="80;75" -DBUILD_PYT=ON -DSPARSITY_SUPPORT=OFF -DMEASURE_BUILD_TIME=ON \
      -DBUILD_CUTLASS_MIXED_GEMM=ON -DBUILD_MULTI_GPU=ON -DBUILD_TRT=OFF \
      -DENABLE_FP8=OFF -DBUILD_PYBIND=ON -DTORCH_PYTHON_LIBRARIES=${TORCH_PYTHON_LIBRARIES} ..
make -j"$(grep -c ^processor /proc/cpuinfo)"
```

#### 3. Run

You can use `examples/pytorch/codefuse/huggingface_convert.py` script to convert checkpoint files from HuggingFace to FasterTransformer.
```
export MODEL_NAME=codefuse
export TENSOR_PARA_SIZE=2

python ../examples/pytorch/codefuse/huggingface_convert.py \
       -o ../models/${MODEL_NAME}/fastertransformer \
       -i ../models/${MODEL_NAME}/transformers \
       -infer_gpu_num ${TENSOR_PARA_SIZE} \
       -processes 20 \
       -weight_data_type fp16 \
       -model_name gptneox
```

You can use `examples/pytorch/codefuse/quant_and_save.py` script to convert fp16 or fp32 FasterTransformer checkpoint files to int8 files and scales, getting faster model load speed and smaller checkpoint files.
```
export MODEL_NAME=codefuse
export TENSOR_PARA_SIZE=2

python ../examples/pytorch/codefuse/quant_and_save.py \
       --in_dir ../models/${MODEL_NAME}/fastertransformer/${TENSOR_PARA_SIZE}-gpu \
       --out_dir ../models/${MODEL_NAME}/fastertransformer/${TENSOR_PARA_SIZE}-gpu_int8 \
       --lib_path ../build/lib/libth_common.so \
       --tensor_para_size ${TENSOR_PARA_SIZE} \
       --use_gptj_residual \
       --data_type fp16
```

You can use `examples/pytorch/codefuse/codefuse_example.py` to run model inference.
```
export MODEL_NAME=codefuse

# fp16 1gpu
python ../examples/pytorch/codefuse/codefuse_example.py \
       --ckpt_path ../models/${MODEL_NAME}/fastertransformer/1-gpu \
       --tokenizer_path ../models/${MODEL_NAME}/transformers

# int8 1gpu
python ../examples/pytorch/codefuse/codefuse_example.py \
       --ckpt_path ../models/${MODEL_NAME}/fastertransformer/1-gpu_int8 \
       --tokenizer_path ../models/${MODEL_NAME}/transformers \
       --int8_mode 1 \
       --enable_int8_weights 1

# fp16 2gpus
torchrun --nproc_per_node 2 ../examples/pytorch/codefuse/codefuse_example.py \
         --world_size 2 \
         --ckpt_path ../models/${MODEL_NAME}/fastertransformer/2-gpu \
         --tokenizer_path ../models/${MODEL_NAME}/transformers

# int8 2gpus
torchrun --nproc_per_node 2 ../examples/pytorch/codefuse/codefuse_example.py \
         --world_size 2 \
         --ckpt_path ../models/${MODEL_NAME}/fastertransformer/2-gpu_int8 \
         --tokenizer_path ../models/${MODEL_NAME}/transformers \
         --int8_mode 1 \
         --enable_int8_weights 1
```