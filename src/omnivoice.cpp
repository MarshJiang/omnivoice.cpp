// omnivoice.cpp: public ABI implementation.
//
// Every entry declared in omnivoice.h lives here under one extern "C" block
// so the symbols carry C linkage and are linkable from C, Rust, Go, Python
// ctypes and any other binding generator. The struct ov_context opaque
// handle owns one BackendPair, one PipelineTTS, one PipelineCodec
// (optional), one BPETokenizer and one VoiceDesign instance. ov_init walks
// the load chain in dependency order and unwinds whatever it already
// allocated when any step fails. ov_free mirrors that order in reverse.

#include "omnivoice.h"

#include "audio-postproc.h"
#include "backend.h"
#include "bpe.h"
#include "ov-error.h"
#include "pipeline-codec.h"
#include "pipeline-tts.h"
#include "version.h"
#include "voice-design.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// Internal definition of the opaque handle. C++ types are fine here
// because nothing in this struct ever crosses the public ABI boundary :
// callers only ever see `struct ov_context *`.
struct ov_context {
    BackendPair   bp;
    PipelineTTS   pt;
    PipelineCodec pc;
    BPETokenizer  tok;
    VoiceDesign   vd;
    bool          codec_loaded;

    // 0 = idle, 1 = establishing the cancellation baseline, 2 = active.
    std::atomic<int>      synthesis_state{ 0 };
    std::atomic<uint64_t> cancellation_epoch{ 0 };
    std::atomic<uint64_t> synthesis_cancellation_epoch{ 0 };
    std::atomic<int>  synthesis_stage{ OV_SYNTHESIS_STAGE_IDLE };
    std::atomic<int>  synthesis_step{ 0 };
    std::atomic<int>  synthesis_total_steps{ 0 };
    std::atomic<int>  synthesis_chunk{ 0 };
    std::atomic<int>  synthesis_total_chunks{ 0 };
};

// Thread-local backing store for ov_last_error(). std::string sized once
// per thread, grows on demand, never freed across calls: the std runtime
// reclaims it on thread exit. An empty string means "no error recorded on
// this thread yet", which ov_last_error() exposes as "".
static thread_local std::string g_last_error;

void ov_set_error_v(const char * fmt, va_list ap) {
    if (!fmt) {
        g_last_error.clear();
        return;
    }
    // Two-pass vsnprintf: first call sizes the buffer, second writes the
    // message. va_copy keeps the original ap valid for the second pass.
    va_list ap2;
    va_copy(ap2, ap);
    int needed = std::vsnprintf(nullptr, 0, fmt, ap2);
    va_end(ap2);
    if (needed < 0) {
        g_last_error = "ov_set_error: vsnprintf failed";
        return;
    }
    g_last_error.resize(static_cast<size_t>(needed));
    std::vsnprintf(g_last_error.data(), static_cast<size_t>(needed) + 1, fmt, ap);
}

void ov_set_error(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    ov_set_error_v(fmt, ap);
    va_end(ap);
}

// Formats a message with printf semantics and throws std::runtime_error.
// The catch sites at the ABI boundary inspect the what() string and feed
// it into ov_set_error so the user-visible diagnostic is identical
// whether the failure used the bool-return path or the throw path.
void ov_throw(const char * fmt, ...) {
    char buf[1024];
    if (fmt) {
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
    } else {
        buf[0] = '\0';
    }
    throw std::runtime_error(buf);
}

// Process-wide log callback. Atomic so ov_log_set can replace it without
// locking: write happens with memory_order_release, every reader sees a
// fully published callback pointer paired with its user_data slot.
// std::atomic on a function pointer is lock-free on every platform we
// target. user_data is a plain pointer because it is only ever published
// alongside cb under the same release ordering.
static std::atomic<ov_log_cb> g_log_cb{ nullptr };
static void *                 g_log_cb_user = nullptr;

void ov_log_set(ov_log_cb cb, void * user_data) {
    g_log_cb_user = user_data;
    g_log_cb.store(cb, std::memory_order_release);
}

// Routes one log line to the installed callback or to stderr. Two-pass
// vsnprintf sizes the heap buffer when the message exceeds the stack
// scratchpad, which keeps the common case allocation-free.
void ov_log(enum ov_log_level level, const char * fmt, ...) {
    if (!fmt) {
        return;
    }

    char    stackbuf[512];
    char *  buf    = stackbuf;
    int     needed = 0;
    va_list ap;
    va_start(ap, fmt);
    {
        va_list ap2;
        va_copy(ap2, ap);
        needed = std::vsnprintf(stackbuf, sizeof(stackbuf), fmt, ap2);
        va_end(ap2);
    }
    if (needed < 0) {
        va_end(ap);
        return;
    }
    std::string heapbuf;
    if ((size_t) needed >= sizeof(stackbuf)) {
        heapbuf.resize((size_t) needed);
        std::vsnprintf(heapbuf.data(), (size_t) needed + 1, fmt, ap);
        buf = heapbuf.data();
    }
    va_end(ap);

    ov_log_cb cb = g_log_cb.load(std::memory_order_acquire);
    if (cb) {
        cb(level, buf, g_log_cb_user);
    } else {
        std::fprintf(stderr, "%s\n", buf);
    }
}

static bool ov_context_should_cancel(void * user_data) {
    auto * ov = static_cast<ov_context *>(user_data);
    return ov && ov->cancellation_epoch.load(std::memory_order_acquire) !=
                     ov->synthesis_cancellation_epoch.load(std::memory_order_relaxed);
}

static void ov_context_update_progress(void *                   user_data,
                                       enum ov_synthesis_stage stage,
                                       int                     step,
                                       int                     total_steps,
                                       int                     chunk,
                                       int                     total_chunks) {
    auto * ov = static_cast<ov_context *>(user_data);
    if (!ov) {
        return;
    }
    ov->synthesis_step.store(step, std::memory_order_relaxed);
    ov->synthesis_total_steps.store(total_steps, std::memory_order_relaxed);
    ov->synthesis_chunk.store(chunk, std::memory_order_relaxed);
    ov->synthesis_total_chunks.store(total_chunks, std::memory_order_relaxed);
    ov->synthesis_stage.store(stage, std::memory_order_release);
}

extern "C" {

const char * ov_version(void) {
    // OMNIVOICE_VERSION is a string literal injected by tools/version.cmake
    // ("<git-hash> (<date>)"), so its storage already has process lifetime
    // and no formatting wrapper is needed.
    return OMNIVOICE_VERSION;
}

const char * ov_last_error(void) {
    // c_str() on an empty std::string is guaranteed to point to a NUL
    // byte by C++11, so callers never have to NULL-check the result.
    return g_last_error.c_str();
}

void ov_audio_free(struct ov_audio * a) {
    if (!a) {
        return;
    }
    if (a->samples) {
        std::free(a->samples);
    }
    a->samples     = nullptr;
    a->n_samples   = 0;
    a->sample_rate = 0;
    a->channels    = 0;
}

void ov_init_default_params(struct ov_init_params * p) {
    p->abi_version = OV_ABI_VERSION;
    p->model_path  = nullptr;
    p->codec_path  = nullptr;
    p->use_fa      = true;
    p->clamp_fp16  = false;
}

void ov_tts_default_params(struct ov_tts_params * p) {
    p->abi_version             = OV_ABI_VERSION;
    p->text                    = nullptr;
    p->lang                    = nullptr;
    p->instruct                = nullptr;
    p->T_override              = 0;
    p->chunk_duration_sec      = 15.0f;
    p->chunk_threshold_sec     = 30.0f;
    p->denoise                 = true;
    p->preprocess_prompt       = true;
    p->mg_num_step             = 32;
    p->mg_guidance_scale       = 2.0f;
    p->mg_t_shift              = 0.1f;
    p->mg_layer_penalty_factor = 5.0f;
    p->mg_position_temperature = 5.0f;
    p->mg_class_temperature    = 0.0f;
    p->mg_seed                 = 42;
    p->ref_audio_tokens        = nullptr;
    p->ref_T                   = 0;
    p->ref_audio_24k           = nullptr;
    p->ref_n_samples           = 0;
    p->ref_text                = nullptr;
    p->dump_dir                = nullptr;
    p->cancel                  = nullptr;
    p->cancel_user_data        = nullptr;
    p->on_chunk                = nullptr;
    p->on_chunk_user_data      = nullptr;
    p->postproc                = true;
}

struct ov_context * ov_init(const struct ov_init_params * params) {
    if (!params || !params->model_path) {
        ov_set_error("ov_init: params or model_path is NULL");
        ov_log(OV_LOG_ERROR, "[OmniVoice] ov_init requires a model_path");
        return nullptr;
    }
    if (params->abi_version > OV_ABI_VERSION) {
        ov_set_error("ov_init: params->abi_version %d > OV_ABI_VERSION %d (binding compiled against a newer header)",
                     params->abi_version, OV_ABI_VERSION);
        ov_log(OV_LOG_ERROR, "[OmniVoice] ov_init params struct is from a newer ABI (%d > %d)", params->abi_version,
               OV_ABI_VERSION);
        return nullptr;
    }

    ov_log(OV_LOG_INFO, "[OmniVoice] omnivoice.cpp %s", ov_version());

    // new ov_context() value-initialises every field: POD aggregates
    // (BackendPair, PipelineTTS, PipelineCodec) are zero-init, std
    // containers in BPETokenizer construct empty, codec_loaded falls to
    // false. Only VoiceDesign needs explicit population below.
    ov_context * ov = new ov_context();
    voice_design_init(&ov->vd);

    // The load chain runs inside a try block. Any failure deep in the GGUF
    // reader, the audio tokenizer load or the LM weight load throws via
    // ov_throw; the catch funnels every variant into one cleanup via
    // ov_free, which is idempotent on partial state (NULL-safe sched, NULL
    // GGUF handles, refcount-correct backend release).
    try {
        ov->bp = backend_init("LM");
        if (!ov->bp.backend) {
            ov_throw("ov_init: backend_init failed (no GGML backend available)");
        }

        if (!pipeline_tts_load(&ov->pt, params->model_path, ov->bp, params->use_fa, params->clamp_fp16)) {
            ov_throw("ov_init: pipeline_tts_load failed for '%s'", params->model_path);
        }

        // BPE tokenizer payload lives inside the same LM GGUF as the weights.
        // Load the base vocab + the OmniVoice-specific special tokens in one
        // shot.
        if (!load_bpe_from_gguf(&ov->tok, params->model_path) ||
            !bpe_load_omnivoice_specials(&ov->tok, params->model_path)) {
            ov_throw("ov_init: BPE / OmniVoice specials load failed for '%s'", params->model_path);
        }

        if (params->codec_path) {
            BackendPair codec_bp = ov->bp;
            const char * primary_backend = ggml_backend_name(ov->bp.backend);
            if (primary_backend && strncmp(primary_backend, "HTP", 3) == 0 &&
                std::getenv("OMNIVOICE_CODEC_HEXAGON") == nullptr) {
                // DAC inference requires codec-specific Hexagon kernels. Keep
                // generic HTP builds on CPU unless the caller explicitly opts in.
                codec_bp.backend     = ov->bp.cpu_backend;
                codec_bp.cpu_backend = ov->bp.cpu_backend;
                codec_bp.has_gpu     = false;
                ov_log(OV_LOG_WARN, "[Load] Codec backend: CPU (HTP codec fallback)");
            } else if (primary_backend && strncmp(primary_backend, "HTP", 3) == 0) {
                ov_log(OV_LOG_INFO, "[Load] Codec backend: HTP");
            }
            if (!pipeline_codec_load(&ov->pc, params->codec_path, codec_bp)) {
                ov_throw("ov_init: pipeline_codec_load failed for '%s'", params->codec_path);
            }
            ov->codec_loaded = true;
        }
    } catch (const std::exception & e) {
        ov_set_error("%s", e.what());
        ov_log(OV_LOG_ERROR, "[OmniVoice] %s", e.what());
        ov_free(ov);
        return nullptr;
    }

    return ov;
}

void ov_free(struct ov_context * ov) {
    if (!ov) {
        return;
    }
    if (ov->codec_loaded) {
        pipeline_codec_free(&ov->pc);
    }
    pipeline_tts_free(&ov->pt);
    backend_release(ov->bp.backend, ov->bp.cpu_backend);
    delete ov;
}

const char * ov_backend_name(const struct ov_context * ov) {
    if (!ov || !ov->bp.backend) {
        return "";
    }
    return ggml_backend_name(ov->bp.backend);
}

void ov_cancel(struct ov_context * ov) {
    if (!ov) {
        return;
    }
    int state = ov->synthesis_state.load(std::memory_order_acquire);
    while (state == 1) {
        std::this_thread::yield();
        state = ov->synthesis_state.load(std::memory_order_acquire);
    }
    if (state == 2) {
        ov->cancellation_epoch.fetch_add(1, std::memory_order_acq_rel);
    }
}

void ov_get_synthesis_progress(const struct ov_context * ov, struct ov_synthesis_progress * out) {
    if (!out) {
        return;
    }
    if (!ov) {
        *out = {};
        out->stage = OV_SYNTHESIS_STAGE_IDLE;
        return;
    }

    out->stage        = static_cast<ov_synthesis_stage>(ov->synthesis_stage.load(std::memory_order_acquire));
    out->step         = ov->synthesis_step.load(std::memory_order_relaxed);
    out->total_steps  = ov->synthesis_total_steps.load(std::memory_order_relaxed);
    out->chunk        = ov->synthesis_chunk.load(std::memory_order_relaxed);
    out->total_chunks = ov->synthesis_total_chunks.load(std::memory_order_relaxed);
}

enum ov_status ov_synthesize(struct ov_context * ov, const struct ov_tts_params * params, struct ov_audio * out) {
    if (!ov || !params) {
        ov_set_error("ov_synthesize: ov / params is NULL");
        if (out) {
            ov_audio_free(out);
        }
        return OV_STATUS_INVALID_PARAMS;
    }
    // Streaming mode (on_chunk non NULL) emits through the callback and
    // leaves out unused, so out=NULL is valid there. Buffered mode requires
    // out to receive the synthesised waveform.
    if (!params->on_chunk && !out) {
        ov_set_error("ov_synthesize: out is NULL in buffered mode");
        return OV_STATUS_INVALID_PARAMS;
    }
    if (params->abi_version > OV_ABI_VERSION) {
        ov_set_error(
            "ov_synthesize: params->abi_version %d > OV_ABI_VERSION %d (binding compiled against a newer header)",
            params->abi_version, OV_ABI_VERSION);
        if (out) {
            ov_audio_free(out);
        }
        return OV_STATUS_INVALID_PARAMS;
    }
    if (!ov->codec_loaded) {
        ov_set_error("ov_synthesize: codec not loaded (pass codec_path to ov_init)");
        if (out) {
            ov_audio_free(out);
        }
        ov_log(OV_LOG_ERROR, "[OmniVoice] ov_synthesize requires a codec-loaded handle");
        return OV_STATUS_INVALID_PARAMS;
    }

    int expected = 0;
    if (!ov->synthesis_state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
        ov_set_error("ov_synthesize: another synthesis is already active on this context");
        if (out) {
            ov_audio_free(out);
        }
        return OV_STATUS_INVALID_PARAMS;
    }

    const uint64_t starting_cancellation_epoch = ov->cancellation_epoch.load(std::memory_order_acquire);
    ov->synthesis_cancellation_epoch.store(starting_cancellation_epoch, std::memory_order_release);
    ov->synthesis_state.store(2, std::memory_order_release);
    ov_context_update_progress(ov, OV_SYNTHESIS_STAGE_PREPARING, 0, params->mg_num_step, 0, 0);

    tts_control control = {
        ov_context_should_cancel,
        ov_context_update_progress,
        ov,
    };

    ov_status status = OV_STATUS_GENERATE_FAILED;
    // Defense in depth: the synthesis path normally reports failures via
    // ov_status return + ov_set_error. A future load-style throw or any
    // std::bad_alloc deep inside the GGML backend is caught here and
    // converted to OV_STATUS_GENERATE_FAILED so an exception never crosses
    // the extern "C" boundary.
    try {
        status = pipeline_tts_synthesize(&ov->pt, &ov->pc, &ov->tok, &ov->vd, params, out, &control);
    } catch (const std::exception & e) {
        ov_set_error("%s", e.what());
        ov_log(OV_LOG_ERROR, "[OmniVoice] %s", e.what());
        if (out) {
            ov_audio_free(out);
        }
        status = OV_STATUS_GENERATE_FAILED;
    } catch (...) {
        ov_set_error("ov_synthesize: unknown native exception");
        ov_log(OV_LOG_ERROR, "[OmniVoice] unknown native synthesis exception");
        if (out) {
            ov_audio_free(out);
        }
        status = OV_STATUS_GENERATE_FAILED;
    }

    if (status == OV_STATUS_OK && ov_context_should_cancel(ov)) {
        if (out) {
            ov_audio_free(out);
        }
        status = OV_STATUS_CANCELLED;
    }

    enum ov_synthesis_stage final_stage = OV_SYNTHESIS_STAGE_FAILED;
    if (status == OV_STATUS_OK) {
        final_stage = OV_SYNTHESIS_STAGE_COMPLETE;
    } else if (status == OV_STATUS_CANCELLED) {
        final_stage = OV_SYNTHESIS_STAGE_CANCELLED;
    }
    ov_context_update_progress(
        ov,
        final_stage,
        ov->synthesis_step.load(std::memory_order_relaxed),
        ov->synthesis_total_steps.load(std::memory_order_relaxed),
        ov->synthesis_chunk.load(std::memory_order_relaxed),
        ov->synthesis_total_chunks.load(std::memory_order_relaxed));
    ov->synthesis_state.store(0, std::memory_order_release);
    return status;
}

int ov_duration_sec_to_tokens(const struct ov_context * ov, float duration_sec) {
    if (!ov || !ov->codec_loaded) {
        ov_set_error("ov_duration_sec_to_tokens: codec not loaded");
        ov_log(OV_LOG_ERROR, "[OmniVoice] ov_duration_sec_to_tokens requires a codec-loaded handle");
        return 1;
    }
    return pipeline_tts_duration_sec_to_tokens(&ov->pc, duration_sec);
}

int ov_num_codebooks(const struct ov_context * ov) {
    if (!ov) {
        ov_set_error("ov_num_codebooks: ov is NULL");
        return 0;
    }
    return ov->pt.lm.num_audio_codebook;
}

void ov_voice_ref_free(struct ov_voice_ref * ref) {
    if (!ref) {
        return;
    }
    if (ref->ref_codes) {
        std::free(ref->ref_codes);
    }
    ref->ref_codes     = nullptr;
    ref->ref_T         = 0;
    ref->num_codebooks = 0;
}

enum ov_status ov_extract_voice_ref(struct ov_context *   ov,
                                    const float *         ref_audio_24k,
                                    int                   ref_n_samples,
                                    struct ov_voice_ref * out) {
    if (out) {
        ov_voice_ref_free(out);
    }
    if (!ov || !ref_audio_24k || !out) {
        ov_set_error("ov_extract_voice_ref: ov, ref_audio_24k or out is NULL");
        return OV_STATUS_INVALID_PARAMS;
    }
    if (!ov->codec_loaded) {
        ov_set_error("ov_extract_voice_ref: codec not loaded");
        return OV_STATUS_GENERATE_FAILED;
    }
    if (ref_n_samples < ov->pc.hop_length) {
        ov_set_error("ov_extract_voice_ref: ref_audio_24k too short for RVQ encode (%d samples, need at least %d)",
                     ref_n_samples, ov->pc.hop_length);
        return OV_STATUS_INVALID_PARAMS;
    }

    try {
        // Match the omnivoice-codec CLI and the --ref-wav synth path:
        // RMS auto-gain, silence trim, then truncation to the hop boundary.
        std::vector<float> buf(ref_audio_24k, ref_audio_24k + ref_n_samples);
        ref_preprocess_audio(buf, 24000, true);

        const int n_aligned = ((int) buf.size() / ov->pc.hop_length) * ov->pc.hop_length;
        if (n_aligned <= 0) {
            ov_set_error("ov_extract_voice_ref: input too short after preprocessing (%zu samples, hop %d)", buf.size(),
                         ov->pc.hop_length);
            return OV_STATUS_INVALID_PARAMS;
        }

        std::vector<int32_t> codes = pipeline_codec_encode(&ov->pc, buf.data(), n_aligned);
        if (codes.empty()) {
            ov_set_error("ov_extract_voice_ref: pipeline_codec_encode returned empty codes");
            return OV_STATUS_GENERATE_FAILED;
        }

        const int K = ov->pt.lm.num_audio_codebook;
        if (K <= 0) {
            ov_set_error("ov_extract_voice_ref: invalid codebook count %d", K);
            return OV_STATUS_GENERATE_FAILED;
        }
        if ((codes.size() % (size_t) K) != 0) {
            ov_set_error("ov_extract_voice_ref: encoded code count %zu is not divisible by %d", codes.size(), K);
            return OV_STATUS_GENERATE_FAILED;
        }

        const size_t codes_bytes = codes.size() * sizeof(int32_t);
        int32_t *    codes_copy  = (int32_t *) std::malloc(codes_bytes);
        if (!codes_copy) {
            ov_set_error("ov_extract_voice_ref: malloc failed for %zu code bytes", codes_bytes);
            return OV_STATUS_OOM;
        }
        std::memcpy(codes_copy, codes.data(), codes_bytes);

        out->ref_codes     = codes_copy;
        out->ref_T         = (int) (codes.size() / (size_t) K);
        out->num_codebooks = K;

        ov_log(OV_LOG_INFO, "[OmniVoice] Extracted voice ref: K=%d T=%d (%d/%d samples)", out->num_codebooks,
               out->ref_T, n_aligned, ref_n_samples);
        return OV_STATUS_OK;
    } catch (const std::bad_alloc &) {
        ov_set_error("ov_extract_voice_ref: out of memory");
        ov_voice_ref_free(out);
        return OV_STATUS_OOM;
    } catch (const std::exception & e) {
        ov_set_error("%s", e.what());
        ov_log(OV_LOG_ERROR, "[OmniVoice] %s", e.what());
        ov_voice_ref_free(out);
        return OV_STATUS_GENERATE_FAILED;
    }
}

}  // extern "C"
