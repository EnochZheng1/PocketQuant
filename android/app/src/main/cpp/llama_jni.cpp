/**
 * JNI bindings for llama.cpp inference engine.
 *
 * Modeled on llama.cpp's Android example (ai_chat.cpp).
 * Manages global model/context state and provides token-by-token generation.
 *
 * IMPORTANT - C++ memory management:
 * All llama.cpp resources (model, context, sampler, batch) are manually managed.
 * They MUST be explicitly freed in nativeUnload() before reloading a new model,
 * or the device will run out of memory with no warning. There is no garbage collector.
 * Free order: sampler -> batch -> context -> model (reverse of allocation order).
 */

#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>
#include <cmath>
#include <sstream>
#include <unistd.h>

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "turboquant.h"

#define TAG "RemoteLLM"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr int   N_THREADS_MIN      = 2;
constexpr int   N_THREADS_MAX      = 4;
constexpr int   N_THREADS_HEADROOM = 2;
constexpr int   DEFAULT_CTX_SIZE   = 2048;
constexpr int   OVERFLOW_HEADROOM  = 4;
constexpr int   BATCH_SIZE         = 512;
constexpr float DEFAULT_TEMP       = 0.3f;

// ---------------------------------------------------------------------------
// Global state (one model loaded at a time)
// ---------------------------------------------------------------------------
static llama_model                 *g_model          = nullptr;
static llama_context               *g_context        = nullptr;
static common_sampler              *g_sampler        = nullptr;
static llama_batch                  g_batch;
static common_chat_templates_ptr    g_chat_templates;

// TurboQuant compressed KV cache (Milestone C)
static tq_cache                    *g_turbo_cache    = nullptr;
static bool                         g_turbo_enabled  = false;
static int                          g_current_ctx_size = DEFAULT_CTX_SIZE;

// Chat history and position tracking
static std::vector<common_chat_msg> g_chat_msgs;
static llama_pos                    g_system_pos     = 0;
static llama_pos                    g_cur_pos        = 0;
static llama_pos                    g_stop_pos       = 0;

// Short-term generation state
static std::string                  g_cached_chars;
static std::ostringstream           g_assistant_ss;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void android_log_callback(ggml_log_level level, const char *text, void * /*user_data*/) {
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: LOGE("%s", text); break;
        case GGML_LOG_LEVEL_WARN:  LOGW("%s", text); break;
        case GGML_LOG_LEVEL_INFO:  LOGI("%s", text); break;
        default:                   LOGD("%s", text); break;
    }
}

static void reset_long_term_states(bool clear_kv = true) {
    g_chat_msgs.clear();
    g_system_pos = 0;
    g_cur_pos    = 0;
    if (clear_kv && g_context) {
        llama_memory_clear(llama_get_memory(g_context), false);
    }
}

static void reset_short_term_states() {
    g_stop_pos = 0;
    g_cached_chars.clear();
    g_assistant_ss.str("");
}

static void shift_context() {
    const int n_discard = (g_cur_pos - g_system_pos) / 2;
    LOGI("shift_context: discarding %d tokens (pos %d -> %d)",
         n_discard, g_cur_pos, g_cur_pos - n_discard);
    llama_memory_seq_rm(llama_get_memory(g_context), 0, g_system_pos, g_system_pos + n_discard);
    llama_memory_seq_add(llama_get_memory(g_context), 0, g_system_pos + n_discard, g_cur_pos, -n_discard);

    // Keep turbo cache in sync — preserve system prompt, shift only user tokens
    if (g_turbo_enabled && g_turbo_cache) {
        tq_cache_shift(g_turbo_cache, n_discard, g_system_pos);
    }

    g_cur_pos  -= n_discard;
    g_stop_pos -= n_discard;
}

/**
 * Decode tokens in batches, writing directly to g_cur_pos.
 * If context fills mid-batch, shift_context() runs and g_cur_pos adjusts
 * automatically — no stale local position variable.
 */
static int decode_tokens_in_batches(
    const std::vector<llama_token> &tokens,
    bool logit_last = false
) {
    for (int i = 0; i < (int)tokens.size(); i += BATCH_SIZE) {
        const int n = std::min((int)tokens.size() - i, BATCH_SIZE);
        common_batch_clear(g_batch);

        // Check against global position — shift if needed
        if (g_cur_pos + n >= g_current_ctx_size - OVERFLOW_HEADROOM) {
            LOGW("decode_tokens_in_batches: context full at %d, shifting", g_cur_pos);
            shift_context();
        }

        for (int j = 0; j < n; j++) {
            bool want_logit = logit_last && (i + j == (int)tokens.size() - 1);
            // Use and increment global position directly
            common_batch_add(g_batch, tokens[i + j], g_cur_pos++, {0}, want_logit);
        }

        if (llama_decode(g_context, g_batch) != 0) {
            LOGE("decode_tokens_in_batches: llama_decode failed");
            return 1;
        }
    }
    return 0;
}

static std::string chat_add_and_format(const std::string &role, const std::string &content) {
    common_chat_msg msg;
    msg.role    = role;
    msg.content = content;
    auto formatted = common_chat_format_single(
        g_chat_templates.get(), g_chat_msgs, msg, role == "user", false);
    g_chat_msgs.push_back(msg);
    return formatted;
}

/**
 * Validate that a C string is valid UTF-8.
 * Incomplete multi-byte sequences return false.
 */
static bool is_valid_utf8(const char *str) {
    if (!str) return true;
    const auto *bytes = (const unsigned char *)str;
    while (*bytes != 0x00) {
        int num;
        if      ((*bytes & 0x80) == 0x00) num = 1;
        else if ((*bytes & 0xE0) == 0xC0) num = 2;
        else if ((*bytes & 0xF0) == 0xE0) num = 3;
        else if ((*bytes & 0xF8) == 0xF0) num = 4;
        else return false;
        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) return false;
            bytes += 1;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// JNI exports
// ---------------------------------------------------------------------------
extern "C" {

/**
 * Initialize the llama backend and logging.
 */
JNIEXPORT void JNICALL
Java_com_remotellm_LlamaModule_nativeInit(JNIEnv *env, jobject, jstring jNativeLibDir) {
    llama_log_set(android_log_callback, nullptr);

    const char *path = env->GetStringUTFChars(jNativeLibDir, nullptr);
    LOGI("Loading backends from %s", path);
    ggml_backend_load_all_from_path(path);
    env->ReleaseStringUTFChars(jNativeLibDir, path);

    llama_backend_init();
    LOGI("Backend initialized");
}

/**
 * Load a GGUF model from the given file path.
 * Returns 0 on success, 1 on failure.
 */
JNIEXPORT jint JNICALL
Java_com_remotellm_LlamaModule_nativeLoadModel(JNIEnv *env, jobject, jstring jModelPath, jint nThreads, jboolean useTurbo) {
    const char *path = env->GetStringUTFChars(jModelPath, nullptr);
    LOGI("Loading model from: %s (turbo=%d)", path, (int)useTurbo);

    llama_model_params model_params = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(path, model_params);
    env->ReleaseStringUTFChars(jModelPath, path);

    if (!model) {
        LOGE("Failed to load model");
        return 1;
    }
    g_model = model;

    // Create context
    const int threads = std::max(N_THREADS_MIN,
        std::min(N_THREADS_MAX, nThreads));
    LOGI("Using %d threads", threads);

    // Create TurboQuant cache if requested
    g_turbo_enabled = (bool)useTurbo;
    if (g_turbo_enabled) {
        int n_layers  = llama_model_n_layer(g_model);
        int n_heads   = llama_model_n_head_kv(g_model);
        int n_embd    = llama_model_n_embd(g_model);
        int n_head_q  = llama_model_n_head(g_model);
        int head_dim  = n_embd / n_head_q;
        int ctx_size  = DEFAULT_CTX_SIZE * 4;  // turbo: 4x context (set to g_current_ctx_size after)

        LOGI("TurboQuant: %d layers, %d kv_heads, head_dim=%d, ctx=%d",
             n_layers, n_heads, head_dim, ctx_size);

        g_turbo_cache = tq_cache_create(n_layers, n_heads, head_dim,
                                         ctx_size, 42, 0.2f);
    }

    llama_context_params ctx_params = llama_context_default_params();
    g_current_ctx_size         = g_turbo_enabled ? DEFAULT_CTX_SIZE * 4 : DEFAULT_CTX_SIZE;
    ctx_params.n_ctx           = g_current_ctx_size;
    ctx_params.n_batch         = BATCH_SIZE;
    ctx_params.n_ubatch        = BATCH_SIZE;
    ctx_params.n_threads       = threads;
    ctx_params.n_threads_batch = threads;

    if (g_turbo_enabled && g_turbo_cache) {
        // Force F32 KV types for easy NEON access in compression callbacks
        ctx_params.type_k          = GGML_TYPE_F32;
        ctx_params.type_v          = GGML_TYPE_F32;
        // Force flash attention ON for non-transposed V layout
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        ctx_params.turbo_cache     = true;
        ctx_params.turbo_cache_ptr = (void *)g_turbo_cache;
    }

    g_context = llama_init_from_model(g_model, ctx_params);
    if (!g_context) {
        LOGE("Failed to create context");
        if (g_turbo_cache) { tq_cache_free(g_turbo_cache); g_turbo_cache = nullptr; }
        llama_model_free(g_model);
        g_model = nullptr;
        return 1;
    }

    if (g_turbo_enabled) {
        LOGI("TurboQuant cache attached to context");
    }

    // Create batch
    g_batch = llama_batch_init(BATCH_SIZE, 0, 1);

    // Load chat template
    g_chat_templates = common_chat_templates_init(g_model, "");

    // Create sampler (temperature 0.3)
    common_params_sampling sparams;
    sparams.temp = DEFAULT_TEMP;
    g_sampler = common_sampler_init(g_model, sparams);

    LOGI("Model loaded successfully (turbo=%d)", g_turbo_enabled);
    return 0;
}

/**
 * Process the system prompt. Tokenizes and decodes it into the KV cache.
 * Returns 0 on success.
 */
JNIEXPORT jint JNICALL
Java_com_remotellm_LlamaModule_nativeProcessSystemPrompt(JNIEnv *env, jobject, jstring jPrompt) {
    reset_long_term_states();
    reset_short_term_states();

    const char *prompt = env->GetStringUTFChars(jPrompt, nullptr);
    std::string formatted(prompt);

    bool has_template = common_chat_templates_was_explicit(g_chat_templates.get());
    if (has_template) {
        formatted = chat_add_and_format("system", prompt);
    }
    env->ReleaseStringUTFChars(jPrompt, prompt);

    auto tokens = common_tokenize(g_context, formatted, has_template, has_template);

    if ((int)tokens.size() > g_current_ctx_size - OVERFLOW_HEADROOM) {
        LOGE("System prompt too long: %d tokens", (int)tokens.size());
        return 1;
    }

    if (decode_tokens_in_batches(tokens)) {
        LOGE("Failed to decode system prompt");
        return 2;
    }

    // g_cur_pos was already advanced by decode_tokens_in_batches
    g_system_pos = g_cur_pos;
    return 0;
}

/**
 * Process a user prompt. Tokenizes and decodes it, preparing for generation.
 * Returns 0 on success.
 */
JNIEXPORT jint JNICALL
Java_com_remotellm_LlamaModule_nativeProcessUserPrompt(JNIEnv *env, jobject, jstring jPrompt, jint maxTokens) {
    reset_short_term_states();

    const char *prompt = env->GetStringUTFChars(jPrompt, nullptr);
    std::string formatted(prompt);

    bool has_template = common_chat_templates_was_explicit(g_chat_templates.get());
    if (has_template) {
        formatted = chat_add_and_format("user", prompt);
    }
    env->ReleaseStringUTFChars(jPrompt, prompt);

    auto tokens = common_tokenize(g_context, formatted, has_template, has_template);

    // Truncate if too long
    const int max_size = g_current_ctx_size - OVERFLOW_HEADROOM;
    if ((int)tokens.size() > max_size) {
        LOGW("User prompt truncated from %d to %d tokens", (int)tokens.size(), max_size);
        tokens.resize(max_size);
    }

    if (decode_tokens_in_batches(tokens, true)) {
        LOGE("Failed to decode user prompt");
        return 2;
    }

    // g_cur_pos was already advanced by decode_tokens_in_batches — don't add prompt_size again
    g_stop_pos = g_cur_pos + maxTokens;
    return 0;
}

/**
 * Generate the next token. Returns:
 * - A non-empty string with the token text (may be empty string for partial UTF-8)
 * - null when generation is complete (EOG or stop position reached)
 */
JNIEXPORT jstring JNICALL
Java_com_remotellm_LlamaModule_nativeGenerateNextToken(JNIEnv *env, jobject) {
    // Context overflow -> shift
    if (g_cur_pos >= g_current_ctx_size - OVERFLOW_HEADROOM) {
        LOGW("Context full at %d/%d, shifting", g_cur_pos, g_current_ctx_size);
        shift_context();
    }

    // Stop at marked position
    if (g_cur_pos >= g_stop_pos) {
        LOGI("Stop position reached: %d", g_stop_pos);
        return nullptr;
    }

    // Sample
    llama_token token = common_sampler_sample(g_sampler, g_context, -1);
    common_sampler_accept(g_sampler, token, true);

    // Decode the new token for next iteration
    common_batch_clear(g_batch);
    common_batch_add(g_batch, token, g_cur_pos, {0}, true);
    if (llama_decode(g_context, g_batch) != 0) {
        LOGE("llama_decode failed for generated token");
        return nullptr;
    }
    g_cur_pos++;

    // Check for end-of-generation
    if (llama_vocab_is_eog(llama_model_get_vocab(g_model), token)) {
        LOGI("End of generation reached");
        chat_add_and_format("assistant", g_assistant_ss.str());
        return nullptr;
    }

    // Convert token to text, with UTF-8 buffering
    auto piece = common_token_to_piece(g_context, token);
    g_cached_chars += piece;

    jstring result;
    if (is_valid_utf8(g_cached_chars.c_str())) {
        result = env->NewStringUTF(g_cached_chars.c_str());
        g_assistant_ss << g_cached_chars;
        g_cached_chars.clear();
    } else {
        // Incomplete UTF-8 sequence — return empty string, keep caching
        result = env->NewStringUTF("");
    }
    return result;
}

/**
 * Unload the model and free ALL resources.
 * MUST be called before loading a new model or when the app is destroyed.
 * Failure to call this will leak all model memory (~1-6GB).
 */
JNIEXPORT void JNICALL
Java_com_remotellm_LlamaModule_nativeUnload(JNIEnv *, jobject) {
    reset_long_term_states(false);  // don't touch KV cache, we're about to free it
    reset_short_term_states();

    // Free in reverse allocation order: sampler -> batch -> context -> model
    if (g_sampler) {
        common_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    g_chat_templates.reset();
    llama_batch_free(g_batch);
    if (g_context) {
        llama_free(g_context);
        g_context = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    // Free turbo cache AFTER model/context (they may reference it)
    if (g_turbo_cache) {
        tq_cache_free(g_turbo_cache);
        g_turbo_cache = nullptr;
    }
    g_turbo_enabled = false;
    LOGI("Model unloaded, all resources freed");
}

/**
 * Shut down the llama backend entirely.
 */
JNIEXPORT void JNICALL
Java_com_remotellm_LlamaModule_nativeShutdown(JNIEnv *, jobject) {
    llama_backend_free();
    LOGI("Backend shut down");
}

/**
 * Get TurboQuant compression stats.
 * Returns [compressionRatio, fallbackRate, totalEntries] or empty if not enabled.
 */
JNIEXPORT jfloatArray JNICALL
Java_com_remotellm_LlamaModule_nativeGetTurboStats(JNIEnv *env, jobject) {
    jfloatArray result = env->NewFloatArray(3);
    if (!g_turbo_cache) return result;

    tq_cache_stats stats;
    tq_cache_get_stats(g_turbo_cache, &stats);

    float data[3] = {
        stats.compression_ratio,
        stats.fallback_rate,
        (float)stats.total_entries
    };
    env->SetFloatArrayRegion(result, 0, 3, data);
    return result;
}

/**
 * Get model info for benchmarking.
 * Returns a float array: [model_size_mb, n_params_billions]
 */
JNIEXPORT jfloatArray JNICALL
Java_com_remotellm_LlamaModule_nativeGetModelInfo(JNIEnv *env, jobject) {
    jfloatArray result = env->NewFloatArray(2);
    if (!g_model) return result;

    float info[2];
    info[0] = (float)(llama_model_size(g_model) / (1024.0 * 1024.0));  // MB
    info[1] = (float)(llama_model_n_params(g_model) / 1e9);            // billions
    env->SetFloatArrayRegion(result, 0, 2, info);
    return result;
}

} // extern "C"
