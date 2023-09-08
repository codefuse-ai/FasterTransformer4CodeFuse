#include "src/fastertransformer/th_op/gptneox/utils/nccl_inherit_utils.h"

namespace ft = fastertransformer;

namespace nccl_inherit {

// if want to use nccl_name, maybe only support pytorch >= 1.12
void HackGroupNCCL::initNCCLComm(
    size_t rank, size_t size, const char* nccl_name, ncclUniqueId& ncclID, ncclComm_t& comm)
{
    if (rank == 0) {
        ncclGetUniqueId(&ncclID);
    }
#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 12))
    broadcastUniqueNCCLID(&ncclID, true, nccl_name, rank);
#elif defined(TORCH_VERSION_MAJOR)                                                                                     \
    && (TORCH_VERSION_MAJOR > 1 || (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 8))
    broadcastUniqueNCCLID(&ncclID, c10d::OpType::SEND, nccl_name, rank);
#else
    broadcastUniqueNCCLID(&ncclID);
#endif
    NCCLCHECK(ncclCommInitRank(&comm, size, ncclID, rank));
}

void ftNcclInitialize(ft::NcclParam& tensor_para,
                      ft::NcclParam& pipeline_para,
                      const int      tensor_para_size,
                      const int      pipeline_para_size,
                      size_t         rank,
                      HackGroupNCCL* nccl_hack_group)
{
    if (tensor_para_size == 1 && pipeline_para_size == 1) {
        FT_LOG_WARNING("Skip NCCL initialization since requested tensor/pipeline parallel sizes are equals to 1.");
        tensor_para.rank_         = 0;
        tensor_para.world_size_   = tensor_para_size;
        pipeline_para.rank_       = 0;
        pipeline_para.world_size_ = pipeline_para_size;
        return;
    }

    ncclUniqueId tp_uid, pp_uid;
    ncclComm_t   tp_nccl_comm, pp_nccl_comm;

    size_t tp_rank = rank % tensor_para_size;
    size_t pp_rank = rank / tensor_para_size;

    char tensor_para_name[32], pipeline_para_name[32];
    sprintf(tensor_para_name, "tp_nccl_comm_%lu", pp_rank);
    sprintf(pipeline_para_name, "pp_nccl_comm_%lu", tp_rank);

    nccl_hack_group->initNCCLComm(tp_rank, tensor_para_size, tensor_para_name, tp_uid, tp_nccl_comm);
    nccl_hack_group->initNCCLComm(pp_rank, pipeline_para_size, pipeline_para_name, pp_uid, pp_nccl_comm);

    tensor_para.world_size_   = tensor_para_size;
    tensor_para.rank_         = tp_rank;
    tensor_para.nccl_uid_     = tp_uid;
    tensor_para.nccl_comm_    = tp_nccl_comm;
    pipeline_para.world_size_ = pipeline_para_size;
    pipeline_para.rank_       = pp_rank;
    pipeline_para.nccl_uid_   = pp_uid;
    pipeline_para.nccl_comm_  = pp_nccl_comm;
    FT_LOG_INFO("NCCL initialized rank=%d tensor_para_size=%d pipeline_para_size=%d tensor_para=%s pipeline_para=%s",
                rank,
                tensor_para_size,
                pipeline_para_size,
                tensor_para.toString().c_str(),
                pipeline_para.toString().c_str());
}
}  // namespace nccl_inherit