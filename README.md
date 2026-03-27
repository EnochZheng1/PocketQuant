# PocketQuant

A TurboQuant-inspired edge inference engine that runs large language models natively on Android. Uses radical KV cache compression via Hadamard rotation and 1-bit quantization to fit 8B+ parameter models into mobile device memory.

## Architecture

```
React Native UI  <-->  JNI Bridge (Turbo Module)  <-->  llama.cpp + TurboQuant Engine (C++)
```

**Three-layer stack:**

- **Frontend** — React Native chat interface with model picker, turbo mode toggle, in-app model downloads, token streaming, and real-time benchmarks
- **Bridge** — Java Turbo Module with background `ExecutorService` for inference, `DeviceEventEmitter` for streaming tokens, wakelock during generation
- **Engine** — Forked llama.cpp with custom `llama_kv_cache_turbo` subclass that intercepts KV writes/reads via `ggml_map_custom1_inplace` ops, compressing through the TurboQuant NEON pipeline

## Features

- **TurboQuant 1-bit KV cache** — toggle in model picker, compresses KV cache 13-24x via FWHT rotation + 1-bit quantization
- In-app model downloader (one-tap download from HuggingFace with progress bar)
- Model picker (TinyLlama 1.1B, Qwen2.5 7B, Qwen2.5 14B)
- Token-by-token streaming from C++ to React Native
- Real-time benchmark overlay (TPS, memory, TTFT, compression ratio, fallback rate)
- Long-press to copy message text
- Wakelock prevents CPU throttling during inference
- Auto-push models to device on build (Gradle `pushModels` task)

## KV Cache Compression (TurboQuant)

The `engine/turboquant/` library implements radical KV cache compression, integrated directly into llama.cpp's attention mechanism:

1. **Randomized Hadamard Rotation** — Sign-flip + Fast Walsh-Hadamard Transform spreads information uniformly across all dimensions (applied AFTER RoPE positional encoding)
2. **1-Bit Quantization** — Each rotated value reduced to its sign bit + per-head FP32 scale factor (~1.1 bits/value)
3. **Structured QJL Error Detection** — Sub-sampled Hadamard projection (O(d log d), not O(d²)) detects high-error entries, falling back to FP32 for those

**Compression for Qwen2.5 7B at 8K context:**

| | Standard F32 | TurboQuant 1-bit |
|---|---|---|
| KV cache memory | ~900 MB | ~38 MB |
| Compression ratio | 1x | **24x** |

All kernels use ARM NEON intrinsics with `alignas(16)` stack buffers. FWHT butterfly stages use `vtrn`/`vcombine` for h=1/h=2 to avoid overlapping NEON reads. Thread-partitioned callbacks (`ith`/`nth` stride) for parallel compression during prompt processing.

## How It Integrates with llama.cpp

The engine uses a minimal fork of llama.cpp (14 lines changed across 5 files + 2 new files):

- `llama_kv_cache_context::cpy_k/cpy_v/get_k/get_v` made `virtual` (4 words added)
- `llama_kv_cache_turbo` subclass overrides these methods to chain `ggml_map_custom1_inplace` ops
- **Store path**: after standard `ggml_set_rows` writes K data, a custom op compresses it into `tq_cache` via NEON
- **Read path**: before attention reads K data, a custom op decompresses from `tq_cache` into the ggml view buffer
- Flash attention forced ON when turbo enabled (non-transposed V layout for simpler stride math)
- KV types forced to F32 when turbo enabled (eliminates F16 conversion in callbacks)

## Prerequisites

- Node.js >= 22.x
- JDK 17 (Temurin recommended)
- Android SDK (API 35+)
- Android NDK 27.1.12297006
- CMake 3.22.1 (via Android SDK)
- Physical arm64 Android device

## Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/EnochZheng1/PocketQuant.git
cd PocketQuant

# Install dependencies
npm install

# Build, install, and auto-push models to device
npm run android
```

First native build compiles llama.cpp + TurboQuant (~10-15 minutes). Subsequent builds are incremental.

## Models

Models download directly in the app — tap the header to open the model picker, then tap any model to download and run. Toggle **TurboQuant 1-bit Cache** for compressed mode.

| Model | Size | Standard TPS | Turbo TPS | RAM (Standard) | RAM (Turbo) |
|-------|------|-------------|-----------|----------------|-------------|
| TinyLlama 1.1B | 638 MB | 10-30 t/s | TBD | ~1 GB | ~0.5 GB |
| Qwen2.5 7B | 4.4 GB | 3-8 t/s | TBD | ~5 GB | ~2 GB |
| Qwen2.5 14B | 8.5 GB | 1-4 t/s | TBD | ~10 GB | ~5 GB |

## Project Structure

```
PocketQuant/
├── src/                              # React Native app
│   ├── screens/ChatScreen.tsx        # Chat UI + model picker + turbo toggle
│   ├── components/
│   │   ├── ModelPicker.tsx           # Model selection + download + turbo switch
│   │   ├── MessageBubble.tsx         # Message display with copy support
│   │   ├── InputBar.tsx              # Text input
│   │   └── BenchmarkOverlay.tsx      # TPS/memory/compression overlay
│   └── hooks/
│       ├── useLlama.ts              # Native module hook with streaming + turbo
│       └── useModelDownloader.ts    # HuggingFace download with progress
├── specs/NativeLlama.ts             # Turbo Module spec (codegen)
├── android/app/src/main/
│   ├── java/com/remotellm/          # Java Turbo Module + JNI bridge
│   ├── jni/CMakeLists.txt           # Builds appmodules + remotellm + turboquant + llama
│   └── cpp/llama_jni.cpp            # C++ JNI bindings with turbo cache creation
├── engine/
│   ├── llama.cpp/                   # Forked llama.cpp with turbo KV cache support
│   │   └── src/
│   │       ├── llama-kv-cache-turbo.h/.cpp  # Turbo cache subclass (NEW)
│   │       ├── llama-kv-cache.h             # virtual on 4 methods (MODIFIED)
│   │       └── llama-model.cpp              # Conditional turbo construction (MODIFIED)
│   └── turboquant/                  # KV cache compression library
│       ├── include/turboquant.h     # Public C API
│       ├── src/
│       │   ├── hadamard.cpp         # Scalar FWHT reference
│       │   ├── hadamard_neon.cpp    # ARM NEON optimized FWHT (vtrn butterfly)
│       │   ├── quantize.cpp         # 1-bit quantize/dequantize + NEON bitpacking
│       │   ├── qjl.cpp             # Structured QJL error detection (sub-sampled FWHT)
│       │   └── turbo_cache.cpp     # Compressed cache (aligned stack buffers, no heap)
│       └── tests/                   # Unit tests (hadamard, quantize, qjl, pipeline)
└── models/                          # Local GGUF files (.gitignored)
```

## Roadmap

| Milestone | Status | Description |
|-----------|--------|-------------|
| A — Baseline Pipeline | Done | React Native + llama.cpp + JNI streaming inference |
| B — Compute Kernels | Done | NEON FWHT rotation + 1-bit quantization + structured QJL |
| C — Engine Integration | Done | Forked llama.cpp with `llama_kv_cache_turbo`, ggml custom ops, turbo UI toggle |
| D — Optimization | Next | Vulkan batch prompt processing, chat history, thermal profiling, release APK |

## License

MIT
