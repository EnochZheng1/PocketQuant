# PocketQuant

A TurboQuant-inspired edge inference engine that runs large language models natively on Android. Uses radical KV cache compression via Hadamard rotation and 1-bit quantization to fit 8B+ parameter models into mobile device memory.

## Architecture

```
React Native UI  <-->  JNI Bridge (Turbo Module)  <-->  llama.cpp + TurboQuant Engine (C++)
```

**Three-layer stack:**

- **Frontend** — React Native chat interface with model picker, in-app model downloads, token streaming, and real-time benchmarks
- **Bridge** — Java Turbo Module with background `ExecutorService` for inference, `DeviceEventEmitter` for streaming tokens
- **Engine** — llama.cpp (C++) for inference + TurboQuant library for KV cache compression via FWHT rotation and 1-bit quantization with ARM NEON

## Features

- In-app model downloader (one-tap download from HuggingFace, auto-loads when done)
- Model picker UI (TinyLlama 1.1B, Qwen2.5 7B, Qwen2.5 14B)
- Token-by-token streaming from C++ to React Native
- Real-time benchmark overlay (tokens/sec, memory, time-to-first-token)
- Long-press to copy message text
- Chat template support (ChatML, Qwen, Llama 3)
- Context shifting for long conversations
- Explicit C++ resource lifecycle management
- Auto-push models to device on build (Gradle `pushModels` task)

## KV Cache Compression (TurboQuant)

The `engine/turboquant/` library implements radical KV cache compression:

1. **Hadamard Rotation** — Fast Walsh-Hadamard Transform spreads information across all dimensions, making each equally important for quantization
2. **1-Bit Quantization** — After rotation, each value is reduced to its sign bit (16x compression vs FP16)
3. **QJL Error Detection** — Structured random projection detects when 1-bit would produce unacceptable error, falling back to FP16 for those entries

**Result**: KV cache drops from ~4 GB to ~300 MB for an 8B model with 8K context (13x compression).

All kernels use ARM NEON intrinsics for optimal mobile performance (~100ns per vector on Cortex-X4).

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

Models are downloaded directly in the app — tap the header to open the model picker, then tap any model to download and run.

| Model | Size | Expected TPS | RAM |
|-------|------|-------------|-----|
| TinyLlama 1.1B | 638 MB | 10-30 t/s | ~1 GB |
| Qwen2.5 7B | 4.4 GB | 3-8 t/s | ~5 GB |
| Qwen2.5 14B | 8.5 GB | 1-4 t/s | ~10 GB |

## Project Structure

```
PocketQuant/
├── src/                              # React Native app
│   ├── screens/ChatScreen.tsx        # Chat UI + model picker integration
│   ├── components/
│   │   ├── ModelPicker.tsx           # Model selection + download UI
│   │   ├── MessageBubble.tsx         # Message display with copy support
│   │   ├── InputBar.tsx              # Text input
│   │   └── BenchmarkOverlay.tsx      # TPS/memory overlay
│   └── hooks/
│       ├── useLlama.ts              # Native module hook with streaming
│       └── useModelDownloader.ts    # HuggingFace download with progress
├── specs/NativeLlama.ts             # Turbo Module spec (codegen)
├── android/app/src/main/
│   ├── java/com/remotellm/          # Java Turbo Module + JNI bridge
│   ├── jni/CMakeLists.txt           # Builds appmodules + remotellm + turboquant
│   └── cpp/llama_jni.cpp            # C++ JNI bindings
├── engine/
│   ├── llama.cpp/                   # Git submodule — inference backend
│   └── turboquant/                  # KV cache compression library
│       ├── include/turboquant.h     # Public API
│       ├── src/
│       │   ├── hadamard.cpp         # Scalar FWHT reference
│       │   ├── hadamard_neon.cpp    # ARM NEON optimized FWHT
│       │   ├── quantize.cpp         # 1-bit quantize/dequantize
│       │   ├── qjl.cpp             # Structured QJL error detection
│       │   └── turbo_cache.cpp     # Compressed cache data structure
│       └── tests/                   # Unit tests
└── models/                          # Local GGUF files (.gitignored)
```

## Roadmap

| Milestone | Status | Description |
|-----------|--------|-------------|
| A — Baseline Pipeline | Done | React Native + llama.cpp + JNI streaming inference |
| B — Compute Kernels | Done | NEON FWHT rotation + 1-bit quantization + QJL error filter |
| C — Engine Integration | Next | Wire turbo_cache into llama.cpp attention mechanism |
| D — Optimization | Planned | Vulkan batch processing, chat history, release APK |

## License

MIT
