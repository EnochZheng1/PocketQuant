# PocketQuant

A TurboQuant-inspired edge inference engine that runs large language models natively on Android. Uses radical KV cache compression via matrix rotation and 1-bit quantization to fit 8B parameter models into mobile device memory.

## Architecture

```
React Native UI  <-->  JNI Bridge (Java Turbo Module)  <-->  llama.cpp Engine (C++)
```

**Three-layer stack:**

- **Frontend** — React Native chat interface with token-by-token streaming and real-time benchmarks (TPS, memory, TTFT)
- **Bridge** — Java Turbo Module using `ExecutorService` for background inference, `DeviceEventEmitter` for streaming tokens back to JS
- **Engine** — llama.cpp compiled via Android NDK/CMake with Vulkan compute support, custom JNI bindings with UTF-8 buffering and explicit memory management

## Features

- Token-by-token streaming from C++ to React Native UI
- Real-time benchmark overlay (tokens/sec, memory usage, time-to-first-token)
- Chat template support (ChatML, Llama 3, etc.)
- Context shifting for long conversations
- Explicit C++ resource lifecycle management (no leaks on model reload)
- `largeHeap` + WakeLock support for sustained inference on large models
- Ninja build system for fast incremental C++ compilation on Windows

## Prerequisites

- Node.js >= 22.x
- JDK 17 (Temurin recommended)
- Android SDK (API 35+)
- Android NDK 27.1.12297006
- CMake 3.22.1 (via Android SDK)
- Physical arm64 Android device (emulator not recommended — no NEON)

## Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/EnochZheng1/PocketQuant.git
cd PocketQuant

# Install JS dependencies
npm install

# Build and run on connected device
npx react-native run-android
```

First native build compiles all of llama.cpp (~10-15 minutes). Subsequent builds are incremental.

## Loading a Model

Push a GGUF model to the device's app-specific storage (no permissions required):

```bash
# Create the model directory
adb shell mkdir -p /storage/emulated/0/Android/data/com.remotellm/files/models/

# Push a model (start small)
adb push tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  /storage/emulated/0/Android/data/com.remotellm/files/models/
```

The app auto-loads from this path on startup. For 8B models (~4.7GB), expect 30-60s load time.

## Project Structure

```
PocketQuant/
├── src/                          # React Native app
│   ├── screens/ChatScreen.tsx    # Main chat interface
│   ├── components/               # MessageBubble, InputBar, BenchmarkOverlay
│   ├── hooks/useLlama.ts        # Native module hook with throttled streaming
│   └── types.ts
├── specs/NativeLlama.ts          # Turbo Module spec (codegen)
├── android/app/src/main/
│   ├── java/com/remotellm/       # Java Turbo Module + JNI bridge
│   └── cpp/
│       ├── CMakeLists.txt        # Builds JNI + links llama.cpp
│       └── llama_jni.cpp         # C++ JNI bindings
├── engine/llama.cpp/             # Git submodule
└── models/                       # Local GGUF files (.gitignored)
```

## Roadmap

| Milestone | Status | Description |
|-----------|--------|-------------|
| A — Baseline Pipeline | Done | React Native + llama.cpp + JNI with streaming inference |
| B — Custom Compute Kernels | Planned | Vulkan shaders for rotation matrix + 1-bit QJL filter |
| C — Engine Integration | Planned | Rewrite attention mechanism with compressed KV cache |
| D — Optimization & Deployment | Planned | Polished UI, thermal profiling, release APK |

## Target Hardware

Optimized for flagship Android SoCs (Snapdragon 8 Gen 3+). Expected performance:

| Model | Size | TPS | Memory |
|-------|------|-----|--------|
| TinyLlama 1.1B Q4_K_M | 670 MB | 10-30 t/s | ~1 GB |
| Llama 3.1 8B Q4_K_M | 4.7 GB | 3-8 t/s | ~5-6 GB |

## License

MIT
