# coding=utf-8
# Copyright (c) 2019-2022 NVIDIA CORPORATION. All rights reserved.
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

"""Helper functions for training models with pytorch-quantization"""

import pickle
import re
import time
import numpy as np
import torch
import random

import pytorch_quantization as quantization
import pytorch_quantization.nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import calib

class Logger:
    def info(self, s):
        print("INFO:", s)
    def warn(self, s):
        print("WARN:", s)
logger = Logger()

name_width = 50 # max width of layer names
qname_width = name_width + 20 # max width of quantizer names


def add_arguments(parser):
    """Add arguments to parser for functions defined in quant_trainer."""

    group = parser.add_argument_group('quant_trainer arguments')
    group.add_argument('--wprec', type=int, default=8,
                        help='weight precision')
    group.add_argument('--aprec', type=int, default=8,
                        help='activation precision')
    group.add_argument('--quant-per-tensor', action='store_true',
                        help='per tensor weight scaling')
    group.add_argument('--quant-disable', action='store_true',
                        help='disable all quantizers')
    group.add_argument('--quant-disable-keyword', type=str, nargs='+',
                        help='disable quantizers by keyword')
    group.add_argument('--calibrator', default='percentile',
                       help='which quantization range calibrator to use')
    group.add_argument('--percentile', default=99.99, type=float,
                       help='percentile for PercentileCalibrator')
    group.add_argument('--fuse-qkv', action='store_true',
                       help='use the same scale factor for qkv')
    group.add_argument('--narrow-range', action='store_true',
                        help='use [-127, 127] range for activations rather than [-128, 127]')
    group.add_argument('--quant-mode', type=str, default="ft2",
                        help='predefined quantization mode, choices: ["ft1", "ft2", "ft3", "trt"](Deprecated)')


def set_args(args):
    args.wprec = 8
    args.aprec = 8
    args.quant_per_tensor = True
    args.quant_disable = False
    args.quant_disable_keyword = ['final_input', 'layernorm_input', 'softmax_input', 'local_input', 'residual_input']
    if args.int8_mode == 2:
        args.quant_disable_keyword.append('[2n]._aftergemm')
    args.fuse_qkv = True
    args.narrow_range = False
    return args

def set_default_quantizers(args):
    """Set default quantizers before creating the model."""

    if args.calibrator == 'max':
        calib_method = 'max'
    elif args.calibrator == 'percentile':
        if args.percentile is None:
            raise ValueError('Specify --percentile when using percentile calibrator')
        calib_method = 'histogram'
    elif args.calibrator == 'mse':
        calib_method = 'histogram'
    elif args.calibrator == 'entropy':
        calib_method = 'histogram'
    else:
        raise ValueError(F'Invalid calibrator {args.calibrator}')

    input_desc = QuantDescriptor(num_bits=args.aprec,
                                 calib_method=calib_method,
                                 narrow_range=args.narrow_range,
                                 )
    weight_desc = QuantDescriptor(num_bits=args.wprec,
                                  axis=(None if args.quant_per_tensor else (0,)),
                                  )
    quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
    quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)


def configure_model(model, args, calib=False):
    """Function called before the training loop."""

    logger.info('Configuring Model for Quantization')
    logger.info(F'using quantization package {quantization.__file__}')

    if not calib:
        if args.quant_disable:
            set_quantizer_by_name(model, [''], _disabled=True)

        if args.quant_disable_keyword:
            set_quantizer_by_name(model, args.quant_disable_keyword, _disabled=True)

        if args.fuse_qkv:
           fuse_qkv(model, args)

    print('Configure calib={}'.format(str(calib)))


def enable_calibration(model):
    """Enable calibration of all *_input_quantizer modules in model."""

    logger.info("Enabling Calibration")
    for name, module in model.named_modules():
        if name.endswith("_quantizer"):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
            logger.info(F"{name:{qname_width}}: {module}")

def finish_calibration(model, args):
    """Disable calibration and load amax for all "*_input_quantizer modules in model."""

    logger.info("Loading calibrated amax")
    for name, module in model.named_modules():
        if name.endswith("_quantizer"):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                elif args.calibrator == "percentile":
                    module.load_calib_amax("percentile", percentile=args.percentile)
                else:
                    module.load_calib_amax(args.calibrator)
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()
    if args.fuse_qkv:
        fuse_qkv(model, args)
    model.cuda()
    print_quant_summary(model)


def fuse_qkv(model, args):
    """Adjust quantization ranges to match an implementation where the QKV projections are implemented with a single GEMM.
    Force the weight and output scale factors to match by taking the max of (Q,K,V).
    """

    def fuse3(qq, qk, qv):
        if not hasattr(qq, '_amax') or not hasattr(qk, '_amax') or not hasattr(qv, '_amax'):
            logger.warn('missing amax buffer, unable to fuse')
            return
        q = qq._amax.detach().item()
        k = qk._amax.detach().item()
        v = qv._amax.detach().item()

        amax = max(q, k, v)
        qq._amax.fill_(amax)
        qk._amax.fill_(amax)
        qv._amax.fill_(amax)
        logger.info(f'          q={q:7.4f} k={k:7.4f} v={v:7.4f} -> {amax:7.4f}')

    for name, mod in model.named_modules():
        if name.endswith('.attention.self'):
            logger.info(f'FUSE_QKV: {name:{name_width}}')
            fuse3(mod.matmul_q_input_quantizer, mod.matmul_k_input_quantizer, mod.matmul_v_input_quantizer)
            fuse3(mod.query._weight_quantizer, mod.key._weight_quantizer, mod.value._weight_quantizer)
            fuse3(mod.query._aftergemm_quantizer, mod.key._aftergemm_quantizer, mod.value._aftergemm_quantizer)


def print_quant_summary(model):
    """Print summary of all quantizer modules in the model."""

    counters = {'quantizers': 0, 'enabled_quantizers': 0,
                'weights': 0, 'quant_weights': 0, 'sparse_weights': 0,
                'params': 0, 'sparse_params': 0}
    for name, mod in model.named_modules():
        if isinstance(mod, quantization.nn.TensorQuantizer):
            print(f'{name:80} {mod}')
            counters['quantizers'] += 1
            if not mod._disabled:
                counters['enabled_quantizers'] += 1

        for pname, param in mod.named_parameters():
            if '.' in pname:
                continue
            counters['params'] += param.numel()
            weight_quantizer = getattr(mod, '_weight_quantizer', None)
            if pname == 'weight':
                counters['weights'] += param.numel()
                if weight_quantizer is not None and not weight_quantizer._disabled:
                    counters['quant_weights'] += param.numel()
                counters['sparse_weights'] += param.eq(0).sum().item()
            counters['sparse_params'] += param.eq(0).sum().item()

    def print_fraction(a, b, counters, desc):
        va = counters[a]
        vb = counters[b]
        pct = va/vb * 100 if vb != 0 else float('NaN')
        print(f'{counters[a]:12}/{vb:12} ({pct:6.2f}%) {desc}')
    print_fraction('enabled_quantizers', 'quantizers', counters, 'TensorQuantizers enabled')
    print_fraction('quant_weights', 'weights', counters, 'Quantized weights')
    print_fraction('sparse_weights', 'weights', counters, 'Zero weights')
    print_fraction('weights', 'params', counters, 'Weight parameters')
    print('\n\n')


def set_quantizer(name, mod, quantizer, k ,v):
    """Set attributes for mod.quantizer."""

    quantizer_mod = getattr(mod, quantizer, None)
    if quantizer_mod is not None:
        assert hasattr(quantizer_mod, k)
        setattr(quantizer_mod, k, v)
    else:
        logger.warn(f'{name} has no {quantizer}')


def set_quantizers(name, mod, which='both', **kwargs):
    """Set quantizer attributes for mod."""

    s = f'Warning: changing {which} quantizers of {name:{qname_width}}'
    for k, v in kwargs.items():
        s += (f' {k}={v}')
        if which in ['input', 'both']:
            set_quantizer(name, mod, '_input_quantizer', k, v)
        if which in ['weight', 'both']:
            set_quantizer(name, mod, '_weight_quantizer', k, v)


def set_quantizer_by_name(model, names, **kwargs):
    """Set quantizer attributes for layers where name contains a substring in names."""

    for name, mod in model.named_modules():
        if hasattr(mod, '_input_quantizer') or hasattr(mod, '_weight_quantizer'):
            for n in names:
                if re.search(n, name):
                    set_quantizers(name, mod, **kwargs)
        elif name.endswith('_quantizer'):
            for n in names:
                if re.search(n, name):
                    s = f'Warning: changing {name:{name_width}}'
                    for k, v in kwargs.items():
                        s += (f' {k}={v}')
                        setattr(mod, k, v)
