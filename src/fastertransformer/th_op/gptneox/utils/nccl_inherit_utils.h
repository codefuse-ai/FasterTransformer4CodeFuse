#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include <c10d/ProcessGroupNCCL.hpp>

namespace ft = fastertransformer;

namespace nccl_inherit {

class HackGroupNCCL: public c10d::ProcessGroupNCCL {
public:
    void initNCCLComm(size_t rank, size_t size, const char* nccl_name, ncclUniqueId& ncclID, ncclComm_t& comm);
};

void ftNcclInitialize(ft::NcclParam& tensor_para,
                      ft::NcclParam& pipeline_para,
                      const int      tensor_para_size,
                      const int      pipeline_para_size,
                      size_t         rank,
                      HackGroupNCCL* nccl_hack_group);
}  // namespace nccl_inherit