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

#include "SwinOp.h"

namespace th = torch;
namespace torch_ext {

template class SwinTransformerFunc<float>;
template class SwinTransformerFunc<half>;

SwinTransformerClass::SwinTransformerClass(std::vector<th::Tensor> w,
                                           th::Tensor              depths,
                                           th::Tensor              num_heads,
                                           int64_t                 max_batch,
                                           int64_t                 img_size,
                                           int64_t                 patch_size,
                                           int64_t                 in_chans,
                                           int64_t                 embed_dim,
                                           int64_t                 window_size,
                                           bool                    ape,
                                           bool                    patch_norm,
                                           int64_t                 layer_num,
                                           double                  mlp_ratio,
                                           bool                    qkv_bias,
                                           double                  qk_scale,
                                           int64_t                 version):
    st_(w[0].scalar_type()), depths_(depths), num_heads_(num_heads), weights_(w)
{

    output_dim_ = embed_dim;
    for (int i = 0; i < layer_num - 1; i++) {
        output_dim_ *= 2;
    }

    for (int i = 0; i < weights_.size(); i++) {
        CHECK_INPUT(weights_[i], st_);
    }

    CHECK_TYPE(depths, at::ScalarType::Int);
    CHECK_TYPE(num_heads, at::ScalarType::Int);

    switch (st_) {
        case at::ScalarType::Float:
            swin_transformer_func_ = new SwinTransformerFunc<float>(max_batch,
                                                                    img_size,
                                                                    patch_size,
                                                                    in_chans,
                                                                    embed_dim,
                                                                    window_size,
                                                                    get_ptr<int>(depths),
                                                                    get_ptr<int>(num_heads),
                                                                    ape,
                                                                    patch_norm,
                                                                    layer_num,
                                                                    mlp_ratio,
                                                                    qkv_bias,
                                                                    qk_scale,
                                                                    version,
                                                                    weights_);
            break;
        case at::ScalarType::Half:
            swin_transformer_func_ = new SwinTransformerFunc<half>(max_batch,
                                                                   img_size,
                                                                   patch_size,
                                                                   in_chans,
                                                                   embed_dim,
                                                                   window_size,
                                                                   get_ptr<int>(depths),
                                                                   get_ptr<int>(num_heads),
                                                                   ape,
                                                                   patch_norm,
                                                                   layer_num,
                                                                   mlp_ratio,
                                                                   qkv_bias,
                                                                   qk_scale,
                                                                   version,
                                                                   weights_);
            break;
        default:
            throw std::runtime_error("Wrong th::Tensor type.");
    }
    info_int_      = torch::empty({11}, torch::dtype(torch::kInt64));
    info_int_[0]   = max_batch;
    info_int_[1]   = img_size;
    info_int_[2]   = patch_size;
    info_int_[3]   = in_chans;
    info_int_[4]   = embed_dim;
    info_int_[5]   = window_size;
    info_int_[6]   = (int64_t)ape;
    info_int_[7]   = (int64_t)patch_norm;
    info_int_[8]   = layer_num;
    info_int_[9]   = (int64_t)qkv_bias;
    info_int_[10]  = (int64_t)version;
    info_float_    = torch::empty({2}, torch::dtype(torch::kFloat64));
    info_float_[0] = mlp_ratio;
    info_float_[1] = qk_scale;
}

std::vector<th::Tensor> SwinTransformerClass::get_pickle_info() const
{
    std::vector<th::Tensor> tmp(weights_);
    tmp.push_back(depths_);
    tmp.push_back(num_heads_);
    tmp.push_back(info_int_);
    tmp.push_back(info_float_);
    return tmp;
}

SwinTransformerClass::~SwinTransformerClass()
{
    delete swin_transformer_func_;
}

th::Tensor SwinTransformerClass::forward(th::Tensor input)
{
    CHECK_INPUT(input, st_);
    int  batch_size = input.size(0);
    auto output =
        torch::empty({batch_size, output_dim_}, torch::dtype(input.dtype()).device(torch::kCUDA).requires_grad(false));
    swin_transformer_func_->forward(batch_size, input, output);
    return output;
}

th::Tensor gen_relative_pos_bias(th::Tensor    table,
                                 th::Tensor    relative_position_bias_index,
                                 const int64_t window_size,
                                 const int64_t head_num,
                                 th::Tensor    cpb_mlp_weight1,
                                 th::Tensor    cpb_mlp_bias1,
                                 th::Tensor    cpb_mlp_weight2,
                                 const int64_t version)
{
    const at::ScalarType _st = table.scalar_type();
    CHECK_INPUT(table, _st);
    switch (_st) {
        case at::ScalarType::Float:
            return gen_relative_pos_bias_impl<float>(table,
                                                     relative_position_bias_index,
                                                     cpb_mlp_weight1,
                                                     cpb_mlp_bias1,
                                                     cpb_mlp_weight2,
                                                     window_size,
                                                     head_num,
                                                     version);
        case at::ScalarType::Half:
            return gen_relative_pos_bias_impl<half>(table,
                                                    relative_position_bias_index,
                                                    cpb_mlp_weight1,
                                                    cpb_mlp_bias1,
                                                    cpb_mlp_weight2,
                                                    window_size,
                                                    head_num,
                                                    version);
        default:
            throw std::runtime_error("Wrong th::Tensor type.");
    }
}

th::Tensor transform_trt_mask(th::Tensor mask, const int64_t B, const int64_t S, bool use_int8)
{
    const at::ScalarType _st = mask.scalar_type();
    CHECK_INPUT(mask, _st);
    switch (_st) {
        case at::ScalarType::Half:
            return transform_trt_mask_impl<half>(mask, B, S, use_int8);
        case at::ScalarType::Float:
        default:
            throw std::runtime_error("Wrong th::Tensor type for transform_mask.");
    }
}

}  // namespace torch_ext

static auto swinTransformerTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::SwinTransformerClass>("SwinTransformerClass")
#else
    torch::jit::class_<torch_ext::SwinTransformerClass>("SwinTransformer", "Class")
#endif
        .def(torch::jit::init<std::vector<th::Tensor>,
                              th::Tensor,
                              th::Tensor,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              bool,
                              bool,
                              int64_t,
                              double,
                              bool,
                              double,
                              int64_t>())
        .def("forward", &torch_ext::SwinTransformerClass::forward)
        .def_pickle(
            [](const c10::intrusive_ptr<torch_ext::SwinTransformerClass>& self) -> std::vector<th::Tensor> {
                return self->get_pickle_info();
            },
            [](std::vector<th::Tensor> state) -> c10::intrusive_ptr<torch_ext::SwinTransformerClass> {
                int                                     state_size = state.size();
                std::vector<th::Tensor>::const_iterator first      = state.begin();
                std::vector<th::Tensor>::const_iterator last       = state.begin() + (state_size - 4);
                std::vector<th::Tensor>                 weights(first, last);
                int                                     idx         = state.size() - 2;
                int                                     i           = 0;
                int64_t                                 max_batch   = state[idx][i++].item().to<int>();
                int64_t                                 img_size    = state[idx][i++].item().to<int>();
                int64_t                                 patch_size  = state[idx][i++].item().to<int>();
                int64_t                                 in_chans    = state[idx][i++].item().to<int>();
                int64_t                                 embed_dim   = state[idx][i++].item().to<int>();
                int64_t                                 window_size = state[idx][i++].item().to<int>();
                bool                                    ape         = state[idx][i++].item().to<bool>();
                bool                                    patch_norm  = state[idx][i++].item().to<bool>();
                int64_t                                 layer_num   = state[idx][i++].item().to<int>();
                bool                                    qkv_bias    = state[idx][i++].item().to<bool>();
                int64_t                                 version     = state[idx][i++].item().to<int>();
                idx                                                 = state.size() - 1;
                double mlp_ratio                                    = state[idx][0].item().to<double>();
                double qk_scale                                     = state[idx][1].item().to<double>();
                return c10::make_intrusive<torch_ext::SwinTransformerClass>(weights,
                                                                            state[state_size - 4],
                                                                            state[state_size - 3],
                                                                            max_batch,
                                                                            img_size,
                                                                            patch_size,
                                                                            in_chans,
                                                                            embed_dim,
                                                                            window_size,
                                                                            ape,
                                                                            patch_norm,
                                                                            layer_num,
                                                                            mlp_ratio,
                                                                            qkv_bias,
                                                                            qk_scale,
                                                                            version);
            });

static auto gen_relative_pos_bias =
    torch::RegisterOperators("fastertransformer::gen_relative_pos_bias", &torch_ext::gen_relative_pos_bias);

static auto transform_trt_mask =
    torch::RegisterOperators("fastertransformer::transform_trt_mask", &torch_ext::transform_trt_mask);