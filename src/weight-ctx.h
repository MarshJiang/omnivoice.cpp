#pragma once
// weight-ctx.h: format-independent weight loading context for ggml backends
//
// Manages a ggml_context for weight tensors + their backend buffer.
// Used by gguf-weights.h for all model loaders.
//
// Usage:
//   WeightCtx wctx;
//   wctx_init(&wctx, n_tensors);
//   ggml_tensor * w = <loader>_load_tensor(&wctx, source, "name");
//   wctx_alloc(&wctx, backend);

#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml.h"

#include <cstddef>
#include <cstring>
#include <cstdio>
#include <memory>
#include <vector>

struct WeightCtx {
    struct ggml_context * ctx;
    ggml_backend_buffer_t buffer;

    struct PendingCopy {
        struct ggml_tensor * tensor;
        const void *         src;
        size_t               nbytes;
        size_t               offset;  // byte offset into dst tensor (0 for regular loads)
    };

    std::vector<PendingCopy> pending;

    // Staging buffers for type-converted data, kept alive until wctx_alloc.
    // unique_ptr keeps the data address stable even when the outer vector grows,
    // so src pointers stored in pending stay valid across staging.push_back().
    std::vector<std::unique_ptr<float[]>> staging;
};

static void wctx_init(WeightCtx * wctx, int n_tensors) {
    size_t                  ctx_size = (size_t) n_tensors * ggml_tensor_overhead() + 1024;
    struct ggml_init_params params   = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    wctx->ctx    = ggml_init(params);
    wctx->buffer = NULL;
    wctx->pending.clear();
    wctx->pending.reserve(n_tensors);
}

static bool wctx_alloc_from_buft(WeightCtx * wctx, ggml_backend_buffer_type_t buft) {
    wctx->buffer = ggml_backend_alloc_ctx_tensors_from_buft(wctx->ctx, buft);
    if (!wctx->buffer) {
        fprintf(stderr, "[WeightCtx] FATAL: failed to allocate backend buffer\n");
        return false;
    }
    // Mark as weight buffer so ggml_backend_sched assigns ops to the correct
    // backend based on weight location (avoids fallback through expansion).
    ggml_backend_buffer_set_usage(wctx->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    size_t total = 0;
    for (size_t i = 0; i < wctx->pending.size();) {
        const struct ggml_tensor * tensor = wctx->pending[i].tensor;
        size_t                     j      = i + 1;
        while (j < wctx->pending.size() && wctx->pending[j].tensor == tensor) {
            j++;
        }

        const size_t tensor_bytes = ggml_nbytes(tensor);
        const bool   whole_write  = j == i + 1 && wctx->pending[i].offset == 0 &&
                                   wctx->pending[i].nbytes == tensor_bytes;
        if (whole_write) {
            const auto & pc = wctx->pending[i];
            ggml_backend_tensor_set(pc.tensor, pc.src, 0, pc.nbytes);
            total += pc.nbytes;
            i = j;
            continue;
        }

        // Some model loaders fuse adjacent source tensors (for example Q/K/V)
        // into one destination. Backends that repack quantized weights need the
        // complete tensor in a single offset-zero upload, so merge only the
        // current tensor's pieces and release the temporary buffer immediately.
        std::vector<uint8_t> merged(tensor_bytes);
        size_t               expected_offset = 0;
        for (size_t k = i; k < j; k++) {
            const auto & pc = wctx->pending[k];
            if (pc.offset != expected_offset || pc.offset + pc.nbytes > tensor_bytes) {
                fprintf(stderr, "[WeightCtx] FATAL: invalid segmented upload for '%s'\n", tensor->name);
                return false;
            }
            memcpy(merged.data() + pc.offset, pc.src, pc.nbytes);
            expected_offset += pc.nbytes;
            total += pc.nbytes;
        }
        if (expected_offset != tensor_bytes) {
            fprintf(stderr, "[WeightCtx] FATAL: incomplete segmented upload for '%s'\n", tensor->name);
            return false;
        }
        ggml_backend_tensor_set(wctx->pending[i].tensor, merged.data(), 0, tensor_bytes);
        i = j;
    }
    fprintf(stderr, "[WeightCtx] Loaded %zu tensors, %.1f MB into backend\n", wctx->pending.size(),
            (float) total / (1024 * 1024));
    wctx->pending.clear();
    wctx->staging.clear();
    return true;
}

static bool wctx_alloc(WeightCtx * wctx, ggml_backend_t backend) {
    return wctx_alloc_from_buft(wctx, ggml_backend_get_default_buffer_type(backend));
}

static void wctx_free(WeightCtx * wctx) {
    if (wctx->buffer) {
        ggml_backend_buffer_free(wctx->buffer);
    }
    if (wctx->ctx) {
        ggml_free(wctx->ctx);
    }
    wctx->buffer = NULL;
    wctx->ctx    = NULL;
}
