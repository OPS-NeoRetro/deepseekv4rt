#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cub/block/block_radix_sort.cuh>

#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "ds4_gpu.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_QK_K 256
#define DS4_CUDA_UNUSED __attribute__((unused))

enum {
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context. */
    DS4_CUDA_ATTENTION_SCORE_CAP = 8192u,
    DS4_CUDA_ATTENTION_RAW_SCORE_CAP = 256u,
    DS4_CUDA_TOPK_MERGE_GROUP = 8u,
    DS4_CUDA_STREAM_EXPERT_DEFAULT = 8u * 64u,
    DS4_CUDA_STREAM_EXPERT_MAX = 61u * 384u
};

struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
};

typedef struct {
    uint8_t scales[CUDA_QK_K / 16];
    uint8_t qs[CUDA_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} cuda_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[CUDA_QK_K / 2];
} cuda_block_q4_K;

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} cuda_block_iq2_xxs;

#include "ds4_cuda_iq2_tables.cuh"

#include "ds4_cuda_runtime.cuh"

#include "ds4_cuda_streaming.cuh"

#include "ds4_cuda_embed_tokens.cuh"

#include "ds4_cuda_matmul_kernels.cuh"

#include "ds4_cuda_q8_kernels.cuh"

#include "ds4_cuda_rms_norm_rope_yarn_kernels.cuh"

#include "ds4_cuda_attention.cuh"

#include "ds4_cuda_hc.cuh"

#include "ds4_cuda_compressor.cuh"

#include "ds4_cuda_moe_router.cuh"

#include "ds4_cuda_indexer.cuh"

#include "ds4_cuda_q2_kernels.cuh"

#include "ds4_cuda_ffn.cuh"

#include "ds4_cuda_moe.cuh"

#include 

#include "ds4_cuda_moe_launch.cuh"

#include "ds4_cuda_hc_launch.cuh"
