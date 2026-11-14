# VexDesk - Real-Time On-Device AI Engine & Flutter Client

VexDesk is a high-performance on-device conversational AI system powered by **StreamVex C++ (llama.cpp core)** and a modern **Pure Flutter Material 3** interface.

---

## 🏗️ Architecture

```
VexDesk/
├── .github/
│   └── workflows/
│       └── build_apk.yml       # Cloud CI/CD: Builds ARM64 native engine & Flutter APK
├── apps/
│   └── vexdesk_client/         # 100% Pure Flutter Native Client (Dart)
│       ├── lib/
│       │   ├── bridge/         # Dart FFI bindings for StreamVex C++
│       │   ├── models/         # Chat messages, stats, model definitions
│       │   ├── state/          # Streaming chat and model hub controllers
│       │   └── ui/             # Dark-Teal Material 3 UI screens & widgets
│       └── android/            # Android Gradle & JNI configuration
├── core_engine/                # High-Performance C++ Inference Engine
│   ├── CMakeLists.txt          # Android NDK / ARM64 NEON build configuration
│   ├── include/                # StreamVex C API headers
│   ├── src/                    # StreamVex engine, token streamers, JNI bridge
│   └── third_party/llama.cpp   # Full llama.cpp C++ inference backend (GGUF, NEON)
└── models/                     # Local GGUF models storage
```

---

## ⚡ Key Features

- **100% Native Pure Flutter**: Zero WebView, zero HTML wrappers, smooth hardware-accelerated animations.
- **llama.cpp C++ Core**: Autoregressive transformer inference running locally on Android via ARM64 NEON SIMD.
- **Zero-Copy Memory-Mapped Weight Streaming (`mmap`)**: Keeps resident RAM minimal while streaming multi-gigabyte GGUF models directly from flash storage.
- **Real-Time Token Streaming**: Low-latency token generation using Dart FFI `NativeCallable` callbacks.
- **GitHub Actions Cloud CI/CD**: Automatic cloud builds for release APK on every commit.

---

## 🚀 Building & Running

### Cloud Build (GitHub Actions)
Every push to `main` automatically triggers `.github/workflows/build_apk.yml`:
1. Sets up Ubuntu runner with Java 17, Android NDK (r26b), and Flutter 3.24.
2. Compiles `core_engine/` into `arm64-v8a/libstreamvex.so`.
3. Compiles Flutter release APK (`VexDesk-release.apk`).
4. Uploads the ready-to-install `.apk` to GitHub Actions Artifacts.

### Local Flutter Run
```bash
cd apps/vexdesk_client
flutter pub get
flutter run --release
```
