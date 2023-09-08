#include "src/fastertransformer/th_op/th_utils.h"

namespace ft = fastertransformer;
namespace th = torch;

namespace pybind_callback {

struct __ctx__ {
    __ctx__();
    __ctx__(std::function<void(py::dict)>& custom_callback_,
            th::Tensor&                    input_lengths_,
            const int                      batch_size_,
            const int                      beam_width_,
            const int                      end_id_);

    std::function<void(py::dict)> custom_callback;
    th::Tensor                    input_lengths;

    std::vector<int> last_seq_length;
    int              end_id;
};

void callback(std::unordered_map<std::string, ft::Tensor>* tensor_map, void* ctx_);

}  // namespace pybind_callback