#include <jni.h>
#include <android/log.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <media/NdkImageReader.h>
#include <pthread.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <unistd.h>
#include <atomic>
#include <ctime>
#include <sched.h>
#include <sys/resource.h>

#define S7_SCREEN_W       1440
#define S7_SCREEN_H       2560
#define FOV               360
#define COLOR_TOL         90
#define SCAN_STEP         2
#define SCAN_ROW_SKIP     1
#define HEAD_ZONE_RATIO   0.42f
#define LOST_FRAMES_MAX   20
#define MIN_CLUSTER       2
#define DENSE_HEAD_SCAN   1

#define HEAD_LOCK_PX      40
#define HARD_LOCK_PX      12
#define SUPER_LOCK_PX     5
#define STICKY_MS         400

#define PREDICT_FRAMES    8
#define VEL_WEIGHT_LO     0.9f
#define VEL_WEIGHT_MID    1.6f
#define VEL_WEIGHT_HI     2.4f
#define VEL_WEIGHT_SUPER  3.0f

#define AIM_DUR_LO        14
#define AIM_DUR_MID       8
#define AIM_DUR_HI        4
#define AIM_DUR_SUPER     2

#define TAP_DURATION      6
#define FIRE_BTN_X_RATIO  0.82f
#define FIRE_BTN_Y_RATIO  0.62f
#define FIRE_BURST_COUNT  2
#define FIRE_BURST_DELAY  10

#define LOG_TAG "AIMLOCK"

static JavaVM*        g_vm       = nullptr;
static jobject        g_service  = nullptr;
static AImageReader*  g_reader   = nullptr;
static ANativeWindow* g_window   = nullptr;
static pthread_t      g_thread;
static std::atomic<bool> g_running{false};
static std::atomic<bool> g_enabled{false};

static int g_screenW = S7_SCREEN_W;
static int g_screenH = S7_SCREEN_H;

static volatile int g_targetX = -1;
static volatile int g_targetY = -1;
static volatile int g_lockLevel = 0;
static volatile int g_lostFrames = 0;
static volatile int g_fireBtnX = 0;
static volatile int g_fireBtnY = 0;
static volatile long g_lastSeenMs = 0;

static float g_prevX[PREDICT_FRAMES] = {0};
static float g_prevY[PREDICT_FRAMES] = {0};
static int   g_prevIdx = 0;
static int   g_prevFill = 0;

static jclass    c_Path = nullptr;
static jclass    c_Stroke = nullptr;
static jclass    c_Builder = nullptr;
static jclass    c_Svc = nullptr;
static jmethodID m_PathCtor = nullptr;
static jmethodID m_MoveTo = nullptr;
static jmethodID m_LineTo = nullptr;
static jmethodID m_StrokeCtor = nullptr;
static jmethodID m_BuilderCtor = nullptr;
static jmethodID m_AddStroke = nullptr;
static jmethodID m_Build = nullptr;
static jmethodID m_Dispatch = nullptr;

static inline void sleep_ms(int ms) {
    timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000L;
    nanosleep(&ts, nullptr);
}

static inline long now_ms() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000L + ts.tv_nsec / 1000000L;
}

static inline double dist_to_center(int x, int y) {
    double dx = x - g_screenW * 0.5;
    double dy = y - g_screenH * 0.5;
    return sqrt(dx * dx + dy * dy);
}

static void set_thread_opts() {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(4, &set);
    CPU_SET(5, &set);
    CPU_SET(6, &set);
    CPU_SET(7, &set);
    sched_setaffinity(0, sizeof(set), &set);
    setpriority(PRIO_PROCESS, 0, -20);
}

static inline int detect_red(const uint8_t* px, int tr, int tg, int tb, int tol) {
    const int r = px[0];
    const int g = px[1];
    const int b = px[2];
    return (unsigned)(r - tr + tol) <= (unsigned)(2 * tol) &&
           (unsigned)(g - tg + tol) <= (unsigned)(2 * tol) &&
           (unsigned)(b - tb + tol) <= (unsigned)(2 * tol);
}

static void scan_dense_head(const uint8_t* __restrict buf,
                            int w, int h,
                            int rowStride, int pixelStride,
                            int yStart, int yEnd,
                            int xStart, int xEnd,
                            int* outX, int* outY, int* outCount) {
    int tr = 0xFF, tg = 0x20, tb = 0x20;
    long sx = 0, sy = 0;
    int cnt = 0;

    if (yStart < 0) yStart = 0;
    if (yEnd > h) yEnd = h;
    if (xStart < 0) xStart = 0;
    if (xEnd > w) xEnd = w;

    for (int y = yStart; y < yEnd; ++y) {
        const uint8_t* __restrict row = buf + (size_t)y * rowStride;
        for (int x = xStart; x < xEnd; ++x) {
            const uint8_t* px = row + (size_t)x * pixelStride;
            if (detect_red(px, tr, tg, tb, COLOR_TOL)) {
                sx += x;
                sy += y;
                ++cnt;
            }
        }
    }
    if (cnt > 0) {
        *outX = (int)(sx / cnt);
        *outY = (int)(sy / cnt);
    }
    *outCount = cnt;
}

static void find_target(const uint8_t* __restrict buf,
                        int w, int h,
                        int rowStride, int pixelStride) {
    const int cx = w >> 1;
    const int cy = h >> 1;
    const int tr = 0xFF, tg = 0x20, tb = 0x20;

    const int y0 = (cy - FOV) < 0 ? 0 : cy - FOV;
    const int y1 = (cy + FOV) > h ? h : cy + FOV;
    const int x0 = (cx - FOV) < 0 ? 0 : cx - FOV;
    const int x1 = (cx + FOV) > w ? w : cx + FOV;

    int minX = w, minY = h, maxX = 0, maxY = 0;
    int count = 0;
    long cxSum = 0, cySum = 0;

    for (int y = y0; y < y1; y += SCAN_STEP * SCAN_ROW_SKIP) {
        const uint8_t* __restrict row = buf + (size_t)y * rowStride;
        for (int x = x0; x < x1; x += SCAN_STEP) {
            const uint8_t* px = row + (size_t)x * pixelStride;
            if (detect_red(px, tr, tg, tb, COLOR_TOL)) {
                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
                cxSum += x;
                cySum += y;
                ++count;
            }
        }
    }

    if (count < MIN_CLUSTER) {
        ++g_lostFrames;
        long dt = now_ms() - g_lastSeenMs;
        if (g_lostFrames > LOST_FRAMES_MAX && dt > STICKY_MS) {
            g_targetX = -1;
            g_targetY = -1;
            g_lockLevel = 0;
            g_prevFill = 0;
            g_prevIdx = 0;
            memset(g_prevX, 0, sizeof(g_prevX));
            memset(g_prevY, 0, sizeof(g_prevY));
        }
        return;
    }
    g_lostFrames = 0;
    g_lastSeenMs = now_ms();

    int bboxH = maxY - minY;
    int bboxW = maxX - minX;
    if (bboxH < 8 || bboxW < 4) return;

    int headTop    = minY - 6;
    int headBottom = minY + (int)(bboxH * HEAD_ZONE_RATIO);
    if (headBottom < minY + 18) headBottom = minY + 18;

    int hx = (int)(cxSum / count);
    int hy = (minY + headBottom) / 2;

    int dhx = hx, dhy = hy, dcnt = 0;
    int hxStart = hx - bboxW;
    int hxEnd   = hx + bboxW;
    scan_dense_head(buf, w, h, rowStride, pixelStride,
                    headTop, headBottom,
                    hxStart, hxEnd,
                    &dhx, &dhy, &dcnt);
    if (dcnt > count / 2) {
        hx = dhx;
        hy = dhy;
    }

    float predictedX = (float)hx;
    float predictedY = (float)hy;

    if (g_prevFill >= 2) {
        float vx = 0, vy = 0;
        int n = 0;
        for (int i = 0; i < g_prevFill; ++i) {
            if (g_prevX[i] == 0 && g_prevY[i] == 0) continue;
            vx += (hx - g_prevX[i]);
            vy += (hy - g_prevY[i]);
            ++n;
        }
        if (n > 0) {
            vx /= n;
            vy /= n;
            float weight;
            if (g_lockLevel >= 3) weight = VEL_WEIGHT_SUPER;
            else if (g_lockLevel == 2) weight = VEL_WEIGHT_HI;
            else if (g_lockLevel == 1) weight = VEL_WEIGHT_MID;
            else weight = VEL_WEIGHT_LO;
            predictedX += vx * weight;
            predictedY += vy * weight;
        }
    }

    g_prevX[g_prevIdx] = (float)hx;
    g_prevY[g_prevIdx] = (float)hy;
    g_prevIdx = (g_prevIdx + 1) % PREDICT_FRAMES;
    if (g_prevFill < PREDICT_FRAMES) ++g_prevFill;

    if (predictedX < 0) predictedX = 0;
    if (predictedY < 0) predictedY = 0;
    if (predictedX >= g_screenW) predictedX = g_screenW - 1;
    if (predictedY >= g_screenH) predictedY = g_screenH - 1;

    g_targetX = (int)predictedX;
    g_targetY = (int)predictedY;

    double dc = dist_to_center(g_targetX, g_targetY);
    if (dc < SUPER_LOCK_PX) g_lockLevel = 3;
    else if (dc < HARD_LOCK_PX) g_lockLevel = 2;
    else if (dc < HEAD_LOCK_PX) g_lockLevel = 1;
    else g_lockLevel = 0;
}

static void send_gesture(int x1, int y1, int x2, int y2, int dur) {
    if (!g_service) return;
    JNIEnv* env = nullptr;
    bool attached = false;
    if (g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_EDETACHED) {
        if (g_vm->AttachCurrentThread(&env, nullptr) != 0) return;
        attached = true;
    }

    jobject path = env->NewObject(c_Path, m_PathCtor);
    env->CallVoidMethod(path, m_MoveTo, (jfloat)x1, (jfloat)y1);
    env->CallVoidMethod(path, m_LineTo, (jfloat)x2, (jfloat)y2);

    jobject stroke = env->NewObject(c_Stroke, m_StrokeCtor,
                                    path, (jlong)0, (jlong)dur);
    jobject builder = env->NewObject(c_Builder, m_BuilderCtor);
    env->CallObjectMethod(builder, m_AddStroke, stroke);
    jobject gesture = env->CallObjectMethod(builder, m_Build);
    env->CallBooleanMethod(g_service, m_Dispatch, gesture, nullptr, nullptr);

    env->DeleteLocalRef(path);
    env->DeleteLocalRef(stroke);
    env->DeleteLocalRef(builder);
    env->DeleteLocalRef(gesture);

    if (attached) g_vm->DetachCurrentThread();
}

static void fire_shot() {
    int fx = g_fireBtnX > 0 ? g_fireBtnX : (int)(g_screenW * FIRE_BTN_X_RATIO);
    int fy = g_fireBtnY > 0 ? g_fireBtnY : (int)(g_screenH * FIRE_BTN_Y_RATIO);

    for (int i = 0; i < FIRE_BURST_COUNT; ++i) {
        JNIEnv* env = nullptr;
        bool attached = false;
        if (g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_EDETACHED) {
            if (g_vm->AttachCurrentThread(&env, nullptr) != 0) return;
            attached = true;
        }

        jobject path = env->NewObject(c_Path, m_PathCtor);
        env->CallVoidMethod(path, m_MoveTo, (jfloat)fx, (jfloat)fy);

        jobject stroke = env->NewObject(c_Stroke, m_StrokeCtor,
                                        path, (jlong)0, (jlong)TAP_DURATION);
        jobject builder = env->NewObject(c_Builder, m_BuilderCtor);
        env->CallObjectMethod(builder, m_AddStroke, stroke);
        jobject gesture = env->CallObjectMethod(env, builder, m_Build);
        env->CallBooleanMethod(g_service, m_Dispatch, gesture, nullptr, nullptr);

        env->DeleteLocalRef(path);
        env->DeleteLocalRef(stroke);
        env->DeleteLocalRef(builder);
        env->DeleteLocalRef(gesture);

        if (attached) g_vm->DetachCurrentThread();
        if (i < FIRE_BURST_COUNT - 1) sleep_ms(FIRE_BURST_DELAY);
    }
}

static inline void aim_at(int tx, int ty) {
    const int cx = g_screenW >> 1;
    const int cy = g_screenH >> 1;

    int dur;
    if (g_lockLevel >= 3) dur = AIM_DUR_SUPER;
    else if (g_lockLevel == 2) dur = AIM_DUR_HI;
    else if (g_lockLevel == 1) dur = AIM_DUR_MID;
    else dur = AIM_DUR_LO;

    send_gesture(cx, cy, tx, ty, dur);

    if (g_lockLevel >= 2) {
        fire_shot();
    }
}

static void* capture_loop(void*) {
    set_thread_opts();
    int frame_skip = 0;

    while (g_running.load()) {
        if (!g_enabled.load() || !g_reader) {
            sleep_ms(30);
            g_lockLevel = 0;
            continue;
        }
        if (++frame_skip & 1) {
            sleep_ms(1);
            continue;
        }

        AImage* img = nullptr;
        if (AImageReader_acquireLatestImage(g_reader, &img) != AMEDIA_OK || !img) {
            sleep_ms(1);
            continue;
        }

        int32_t w = 0, h = 0;
        AImage_getWidth(img, &w);
        AImage_getHeight(img, &h);

        uint8_t* data = nullptr;
        int len = 0;
        AImage_getPlaneData(img, 0, &data, &len);

        int32_t rs = 0, ps = 0;
        AImage_getPlaneRowStride(img, 0, &rs);
        AImage_getPlanePixelStride(img, 0, &ps);

        if (data) {
            find_target(data, w, h, rs, ps);
            if (g_targetX >= 0 && g_targetY >= 0) {
                aim_at(g_targetX, g_targetY);
            }
        }

        AImage_delete(img);
        sleep_ms(1);
    }
    return nullptr;
}

extern "C" JNIEXPORT void JNICALL
Java_com_palofsc_aimlock_AimlockService_nativeInit(JNIEnv* env, jobject thiz, jint sw, jint sh) {
    env->GetJavaVM(&g_vm);
    g_service = env->NewGlobalRef(thiz);
    g_screenW = sw > 0 ? sw : S7_SCREEN_W;
    g_screenH = sh > 0 ? sh : S7_SCREEN_H;
    g_fireBtnX = (int)(g_screenW * FIRE_BTN_X_RATIO);
    g_fireBtnY = (int)(g_screenH * FIRE_BTN_Y_RATIO);

    c_Path = reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("android/graphics/Path")));
    c_Stroke = reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass(
        "android/accessibilityservice/GestureDescription$StrokeDescription")));
    c_Builder = reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass(
        "android/accessibilityservice/GestureDescription$Builder")));
    c_Svc = reinterpret_cast<jclass>(env->NewGlobalRef(env->GetObjectClass(thiz)));

    m_PathCtor = env->GetMethodID(c_Path, "<init>", "()V");
    m_MoveTo = env->GetMethodID(c_Path, "moveTo", "(FF)V");
    m_LineTo = env->GetMethodID(c_Path, "lineTo", "(FF)V");

    m_StrokeCtor = env->GetMethodID(c_Stroke, "<init>",
        "(Landroid/graphics/Path;JJ)V");

    m_BuilderCtor = env->GetMethodID(c_Builder, "<init>", "()V");
    m_AddStroke = env->GetMethodID(c_Builder, "addStroke",
        "(Landroid/accessibilityservice/GestureDescription$StrokeDescription;)"
        "Landroid/accessibilityservice/GestureDescription$Builder;");
    m_Build = env->GetMethodID(c_Builder, "build",
        "()Landroid/accessibilityservice/GestureDescription;");

    m_Dispatch = env->GetMethodID(c_Svc, "dispatchGesture",
        "(Landroid/accessibilityservice/GestureDescription;"
        "Landroid/accessibilityservice/AccessibilityService$GestureResultCallback;"
        "Landroid/os/Handler;)Z");
}

extern "C" JNIEXPORT void JNICALL
Java_com_palofsc_aimlock_AimlockService_nativeStart(JNIEnv* env, jobject,
                                                    jobject surface, jint w, jint h) {
    g_window = ANativeWindow_fromSurface(env, surface);
    ANativeWindow_setBuffersGeometry(g_window, 0, 0, WINDOW_FORMAT_RGBA_8888);
    AImageReader_new(w, h, AIMAGE_FORMAT_RGBA_8888, 6, &g_reader);
    g_running.store(true);
    pthread_create(&g_thread, nullptr, capture_loop, nullptr);
}

extern "C" JNIEXPORT void JNICALL
Java_com_palofsc_aimlock_AimlockService_nativeStop(JNIEnv*, jobject) {
    g_running.store(false);
    pthread_join(g_thread, nullptr);
    if (g_reader) { AImageReader_delete(g_reader); g_reader = nullptr; }
    if (g_window) { ANativeWindow_release(g_window); g_window = nullptr; }
}

extern "C" JNIEXPORT void JNICALL
Java_com_palofsc_aimlock_AimlockService_nativeToggle(JNIEnv*, jobject) {
    g_enabled.store(!g_enabled.load());
    g_lockLevel = 0;
    g_lostFrames = 0;
    g_prevIdx = 0;
    g_prevFill = 0;
    g_lastSeenMs = 0;
    memset(g_prevX, 0, sizeof(g_prevX));
    memset(g_prevY, 0, sizeof(g_prevY));
}

extern "C" JNIEXPORT void JNICALL
Java_com_palofsc_aimlock_AimlockService_nativeSetFireButton(JNIEnv*, jobject, jint x, jint y) {
    g_fireBtnX = x;
    g_fireBtnY = y;
}

extern "C" JNIEXPORT void JNICALL
Java_com_palofsc_aimlock_AimlockService_nativeRelease(JNIEnv* env, jobject) {
    if (g_service) { env->DeleteGlobalRef(g_service); g_service = nullptr; }
    if (c_Path) { env->DeleteGlobalRef(c_Path); c_Path = nullptr; }
    if (c_Stroke) { env->DeleteGlobalRef(c_Stroke); c_Stroke = nullptr; }
    if (c_Builder) { env->DeleteGlobalRef(c_Builder); c_Builder = nullptr; }
    if (c_Svc) { env->DeleteGlobalRef(c_Svc); c_Svc = nullptr; }
}
