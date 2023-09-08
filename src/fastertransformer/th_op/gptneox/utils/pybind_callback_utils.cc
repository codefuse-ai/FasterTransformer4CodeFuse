#include "src/fastertransformer/th_op/gptneox/utils/pybind_callback_utils.h"

namespace ft = fastertransformer;
namespace th = torch;

namespace pybind_callback {

__ctx__::__ctx__() {}

__ctx__::__ctx__(std::function<void(py::dict)>& custom_callback_,
                 th::Tensor&                    input_lengths_,
                 int                            batch_size_,
                 int                            beam_width_,
                 int                            end_id_):
    custom_callback(custom_callback_),
    input_lengths(input_lengths_),
    last_seq_length(batch_size_ * beam_width_, 0),
    end_id(end_id_)
{
}

void callback(std::unordered_map<std::string, ft::Tensor>* tensor_map, void* ctx_)
{
    __ctx__* ctx = (__ctx__*)ctx_;

    std::function<void(py::dict)> custom_callback          = ctx->custom_callback;
    const int                     batch_size               = tensor_map->at("output_ids").shape[0];
    const int                     beam_width               = tensor_map->at("output_ids").shape[1];
    const int                     total_request_output_len = tensor_map->at("output_ids").shape[2];
    const size_t                  type_size                = ft::Tensor::getTypeSize(tensor_map->at("output_ids").type);

    th::Tensor input_lengths = ctx->input_lengths;

    int* output_lengths_data     = tensor_map->at("sequence_length").getPtr<int>();
    int* output_lengths_cpu_data = (int*)malloc(batch_size * beam_width * type_size);
    cudaDeviceSynchronize();
    cudaMemcpy(
        output_lengths_cpu_data, output_lengths_data, batch_size * beam_width * type_size, cudaMemcpyDeviceToHost);

    int* input_lengths_data     = torch_ext::get_ptr<int>(input_lengths);
    int* input_lengths_cpu_data = (int*)malloc(batch_size * type_size);
    cudaDeviceSynchronize();
    cudaMemcpy(input_lengths_cpu_data, input_lengths_data, batch_size * type_size, cudaMemcpyDeviceToHost);

    int max_input_length = 0;
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        if (input_lengths_cpu_data[batch_idx] > max_input_length) {
            max_input_length = input_lengths_cpu_data[batch_idx];
        }
    }

    int* padding_lengths_data = (int*)malloc(batch_size * type_size);
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        padding_lengths_data[batch_idx] = max_input_length - input_lengths_cpu_data[batch_idx];
    }

    int* one_token_buffer = (int*)malloc(type_size);

    py::list last_tokens_list;
    py::list idxs_list;
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        py::list last_tokens_beam_list;
        py::list idxs_beam_list;
        int      input_length   = input_lengths_cpu_data[batch_idx];
        int      padding_length = padding_lengths_data[batch_idx];
        for (int beam_idx = 0; beam_idx < beam_width; beam_idx++) {
            int batchxbeam_idx    = batch_idx * beam_width + beam_idx;
            int seq_length        = ((int*)output_lengths_cpu_data)[batchxbeam_idx] - padding_length;
            int seq_idx           = seq_length - 1;
            int output_ids_offset = batchxbeam_idx * total_request_output_len + seq_idx;

            cudaDeviceSynchronize();
            cudaMemcpy(one_token_buffer,
                       tensor_map->at("output_ids").getPtrWithOffset(output_ids_offset),
                       type_size,
                       cudaMemcpyDeviceToHost);
            int token = *one_token_buffer;

            if (seq_length == ctx->last_seq_length[batchxbeam_idx]) {
                token = ctx->end_id;
            }
            else {
                ctx->last_seq_length[batchxbeam_idx] = seq_length;
            }

            last_tokens_beam_list.append(token);
            idxs_beam_list.append(seq_idx - input_length);
        }
        last_tokens_list.append(last_tokens_beam_list);
        idxs_list.append(idxs_beam_list);
    }

    py::dict message_dict;
    message_dict["last_tokens"] = last_tokens_list;
    message_dict["idxs"]        = idxs_list;

    free(output_lengths_cpu_data);
    free(input_lengths_cpu_data);
    free(padding_lengths_data);
    free(one_token_buffer);

    custom_callback(message_dict);
}
}  // namespace pybind_callback