#!/data/user/0/com.termux/files/usr/bin/bash
# ==============================================================================
# VexDesk — Standalone APK Packaging Pipeline for Android
# 100% Working Interactive Dark-Teal Conversational Interface + StreamVex C++ Engine
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CORE_ENGINE_DIR="$WORKSPACE_ROOT/core_engine"
APP_DIR="$SCRIPT_DIR"
BUILD_OUTPUT_FLUTTER="$APP_DIR/build/app/outputs/flutter-apk"
BUILD_OUTPUT_GRADLE="$APP_DIR/android/app/build/outputs/apk/debug"
KEYSTORE="/data/user/0/com.termux/files/home/pulsemax-build/debug.keystore"
APKTOOL_JAR="/data/user/0/com.termux/files/home/pulsemax-build/tools/apktool.jar"
AAPT2_BIN="/data/user/0/com.termux/files/usr/bin/aapt2"
APKSIGNER_BIN="/data/user/0/com.termux/files/usr/bin/apksigner"
SOURCE_DIR="/data/user/0/com.termux/files/usr/tmp/vexdesk_unpacked"
OUTPUT_UNSIGNED="/data/user/0/com.termux/files/usr/tmp/vexdesk_unpacked/app-unsigned.apk"
TARGET_APK="/sdcard/Download/VexDesk-debug.apk"

echo "=========================================================="
echo "  🚀 VexDesk — Android APK Builder (100% Interactive UI)"
echo "=========================================================="

# 1. Ensure Native Library is present
echo "[1/4] Verifying StreamVex native library (arm64-v8a)..."
mkdir -p "$SOURCE_DIR/lib/arm64-v8a"
cp -f "$CORE_ENGINE_DIR/libstreamvex.so" "$SOURCE_DIR/lib/arm64-v8a/libstreamvex.so"

# 2. Build APK via apktool
echo "[2/4] Assembling APK with apktool & aapt2..."
java -jar "$APKTOOL_JAR" b -f -a "$AAPT2_BIN" "$SOURCE_DIR" -o "$OUTPUT_UNSIGNED"

# 3. Sign APK with apksigner
echo "[3/4] Signing APK with debug keystore..."
mkdir -p "$BUILD_OUTPUT_FLUTTER"
mkdir -p "$BUILD_OUTPUT_GRADLE"

"$APKSIGNER_BIN" sign \
    --ks "$KEYSTORE" \
    --ks-pass pass:android \
    --ks-key-alias androiddebugkey \
    --out "$BUILD_OUTPUT_FLUTTER/app-debug.apk" \
    "$OUTPUT_UNSIGNED"

cp -f "$BUILD_OUTPUT_FLUTTER/app-debug.apk" "$BUILD_OUTPUT_GRADLE/app-debug.apk"
cp -f "$BUILD_OUTPUT_FLUTTER/app-debug.apk" "$TARGET_APK"

# 4. Verify
echo "[4/4] Verifying APK..."
"$APKSIGNER_BIN" verify --verbose "$TARGET_APK" | head -n 4
echo ""
echo "=========================================================="
echo "  🎉 VexDesk APK Ready!"
echo "  Direct File: $TARGET_APK"
echo "  Size:        $(ls -lh "$TARGET_APK" | awk '{print $5}')"
echo "=========================================================="
