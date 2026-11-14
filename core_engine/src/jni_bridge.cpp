#include <jni.h>
#include <string>
#include <vector>
#include <cstring>
#include "streamvex.h"

static thread_local JNIEnv* tls_env = nullptr;
static thread_local jobject tls_thiz = nullptr;
static thread_local jmethodID tls_on_token_mid = nullptr;

static void jni_token_callback(const char* token) {
    if (tls_env && tls_thiz && tls_on_token_mid && token) {
        jstring jtok = tls_env->NewStringUTF(token);
        if (jtok) {
            tls_env->CallVoidMethod(tls_thiz, tls_on_token_mid, jtok);
            tls_env->DeleteLocalRef(jtok);
        }
    }
}

extern "C" {

JNIEXPORT jint JNICALL Java_com_vexdesk_client_NativeBridge_initEngine(
    JNIEnv* env, jobject thiz, jstring modelPath) {
    (void)thiz;
    if (!modelPath) return -1;
    const char* c_path = env->GetStringUTFChars(modelPath, nullptr);
    if (!c_path) return -2;

    int res = init_engine(c_path);
    env->ReleaseStringUTFChars(modelPath, c_path);
    return (jint)res;
}

JNIEXPORT jint JNICALL Java_com_vexdesk_client_NativeBridge_nativeInitEngine(
    JNIEnv* env, jobject thiz, jstring modelPath) {
    return Java_com_vexdesk_client_NativeBridge_initEngine(env, thiz, modelPath);
}

JNIEXPORT jint JNICALL Java_com_vexdesk_client_NativeBridge_streamPrompt(
    JNIEnv* env, jobject thiz, jstring prompt) {
    if (!prompt) return -1;

    jclass clazz = env->GetObjectClass(thiz);
    jmethodID mid = env->GetMethodID(clazz, "onTokenReceived", "(Ljava/lang/String;)V");
    if (!mid) return -2;

    tls_env = env;
    tls_thiz = thiz;
    tls_on_token_mid = mid;

    const char* c_prompt = env->GetStringUTFChars(prompt, nullptr);
    int res = stream_token(c_prompt, jni_token_callback);
    env->ReleaseStringUTFChars(prompt, c_prompt);

    tls_env = nullptr;
    tls_thiz = nullptr;
    tls_on_token_mid = nullptr;

    return (jint)res;
}

JNIEXPORT jint JNICALL Java_com_vexdesk_client_NativeBridge_nativeStreamPrompt(
    JNIEnv* env, jobject thiz, jstring prompt) {
    return Java_com_vexdesk_client_NativeBridge_streamPrompt(env, thiz, prompt);
}

JNIEXPORT jdouble JNICALL Java_com_vexdesk_client_NativeBridge_getMemoryUsage(
    JNIEnv* env, jobject thiz) {
    (void)env;
    (void)thiz;
    return (jdouble)get_memory_usage();
}

JNIEXPORT jdouble JNICALL Java_com_vexdesk_client_NativeBridge_nativeGetMemoryUsage(
    JNIEnv* env, jobject thiz) {
    return Java_com_vexdesk_client_NativeBridge_getMemoryUsage(env, thiz);
}

JNIEXPORT jboolean JNICALL Java_com_vexdesk_client_NativeBridge_isEngineReady(
    JNIEnv* env, jobject thiz) {
    (void)env;
    (void)thiz;
    return (jboolean)(is_engine_ready() != 0);
}

JNIEXPORT jboolean JNICALL Java_com_vexdesk_client_NativeBridge_nativeIsEngineReady(
    JNIEnv* env, jobject thiz) {
    return Java_com_vexdesk_client_NativeBridge_isEngineReady(env, thiz);
}

JNIEXPORT void JNICALL Java_com_vexdesk_client_NativeBridge_releaseEngine(
    JNIEnv* env, jobject thiz) {
    (void)env;
    (void)thiz;
    release_engine();
}

JNIEXPORT void JNICALL Java_com_vexdesk_client_NativeBridge_nativeReleaseEngine(
    JNIEnv* env, jobject thiz) {
    Java_com_vexdesk_client_NativeBridge_releaseEngine(env, thiz);
}

JNIEXPORT jstring JNICALL Java_com_vexdesk_client_NativeBridge_getTelemetry(
    JNIEnv* env, jobject thiz) {
    (void)thiz;
    char buf[1024] = {0};
    int res = get_last_generation_telemetry(buf, sizeof(buf));
    if (res != 0) {
        return env->NewStringUTF("{}");
    }
    return env->NewStringUTF(buf);
}

JNIEXPORT jstring JNICALL Java_com_vexdesk_client_NativeBridge_nativeGetTelemetry(
    JNIEnv* env, jobject thiz) {
    return Java_com_vexdesk_client_NativeBridge_getTelemetry(env, thiz);
}

} // extern "C"
