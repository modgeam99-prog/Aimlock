
#include <jni.h>
#include <android/log.h>
#include <sys/mman.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <thread>
#include <chrono>
#include <string>
#include <vector>
#include <errno.h>

#define LOG_TAG "aimlock"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ================== OFFSET (thay theo version FF) ==================
namespace Offset {
    const uintptr_t LOCAL_PLAYER = 0x0A1B2C3D;
    const uintptr_t ENTITY_LIST  = 0x0B2C3D4E;
    const uintptr_t ENTITY_COUNT = 0x0C3D4E5F;
    const uintptr_t POS_X        = 0x10;
    const uintptr_t POS_Y        = 0x14;
    const uintptr_t POS_Z        = 0x18;
    const uintptr_t TEAM_ID      = 0x20;
    const uintptr_t HEALTH       = 0x24;
    const uintptr_t VIEW_MATRIX  = 0x40;
    const uintptr_t CAMERA_POS   = 0x50;
    const uintptr_t AIM_YAW      = 0x60;
    const uintptr_t AIM_PITCH    = 0x64;
    const uintptr_t SHOOT_FLAG   = 0x68;
    const uintptr_t SCREEN_W     = 0x70;
    const uintptr_t SCREEN_H     = 0x74;
}

// ================== CAU HINH ==================
struct Config {
    float FOV = 200.0f;
    float MAX_DIST = 300.0f;
    bool TEAM_CHECK = true;
    bool RUNNING = true;
} cfg;

// ================== STRUCT ==================
struct Vec3 { float x, y, z; };

// ================== MEMORY ==================
pid_t g_pid = -1;
int g_memfd = -1;

// Tim PID theo ten tien trinh
pid_t FindPid(const char* name) {
    DIR* dir = opendir("/proc");
    if (!dir) return -1;
    dirent* entry;
    pid_t result = -1;
    while ((entry = readdir(dir))) {
        if (entry->d_type != DT_DIR) continue;
        int pid = atoi(entry->d_name);
        if (pid <= 0) continue;
        char path[64];
        snprintf(path, sizeof(path), "/proc/%d/cmdline", pid);
        int fd = open(path, O_RDONLY);
        if (fd < 0) continue;
        char cmd[256] = {0};
        ssize_t n = read(fd, cmd, sizeof(cmd) - 1);
        close(fd);
        if (n > 0 && strstr(cmd, name)) {
            result = pid;
            break;
        }
    }
    closedir(dir);
    return result;
}

// Mo memory cua tien trinh
bool OpenTarget(pid_t pid) {
    g_pid = pid;
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/mem", pid);
    g_memfd = open(path, O_RDWR);
    if (g_memfd < 0) {
        LOGE("Khong mo duoc /proc/%d/mem (errno=%d)", pid, errno);
        return false;
    }
    if (ptrace(PTRACE_ATTACH, pid, 0, 0) < 0) {
        LOGE("PTRACE_ATTACH that bai (errno=%d)", errno);
    }
    waitpid(pid, nullptr, 0);
    LOGI("Da attach PID=%d", pid);
    return true;
}

// Doc memory
template<typename T>
T ReadMem(uintptr_t addr) {
    T val{};
    if (g_memfd < 0) return val;
    pread64(g_memfd, &val, sizeof(T), addr);
    return val;
}

// Ghi memory
template<typename T>
void WriteMem(uintptr_t addr, T val) {
    if (g_memfd < 0) return;
    pwrite64(g_memfd, &val, sizeof(T), addr);
}

// Doc vector vi tri
Vec3 ReadPos(uintptr_t base) {
    return {
        ReadMem<float>(base + Offset::POS_X),
        ReadMem<float>(base + Offset::POS_Y),
        ReadMem<float>(base + Offset::POS_Z)
    };
}

// ================== WORLD TO SCREEN ==================
bool WorldToScreen(const Vec3& w, float* m, int sw, int sh, Vec3& out) {
    float cx = m[0]*w.x + m[4]*w.y + m[8] *w.z + m[12];
    float cy = m[1]*w.x + m[5]*w.y + m[9] *w.z + m[13];
    float cw = m[3]*w.x + m[7]*w.y + m[11]*w.z + m[15];
    if (cw < 0.01f) return false;
    out.x = (sw / 2.0f) * (cx / cw + 1.0f);
    out.y = (sh / 2.0f) * (1.0f - cy / cw);
    out.z = cw;
    return true;
}

// ================== TIM MUC TIEU ==================
uintptr_t FindBestTarget(uintptr_t localBase, float* vm, int sw, int sh) {
    int count = ReadMem<int>(Offset::ENTITY_COUNT);
    uintptr_t listBase = ReadMem<uintptr_t>(Offset::ENTITY_LIST);
    int localTeam = ReadMem<int>(localBase + Offset::TEAM_ID);

    uintptr_t best = 0;
    float bestDist = cfg.FOV;
    float cx = sw / 2.0f;
    float cy = sh / 2.0f;

    for (int i = 0; i < count; i++) {
        uintptr_t ent = ReadMem<uintptr_t>(listBase + i * 4);
        if (!ent) continue;
        if (ReadMem<float>(ent + Offset::HEALTH) <= 0) continue;
        if (cfg.TEAM_CHECK && ReadMem<int>(ent + Offset::TEAM_ID) == localTeam) continue;

        Vec3 p = ReadPos(ent);
        float d = sqrtf(p.x*p.x + p.y*p.y + p.z*p.z);
        if (d > cfg.MAX_DIST) continue;

        Vec3 s;
        if (!WorldToScreen(p, vm, sw, sh, s)) continue;
        float dx = s.x - cx;
        float dy = s.y - cy;
        float sd = sqrtf(dx*dx + dy*dy);
        if (sd < bestDist) {
            bestDist = sd;
            best = ent;
        }
    }
    return best;
}

// ================== GHI GOC NGAM ==================
void WriteAim(uintptr_t localBase, const Vec3& target, const Vec3& camera) {
    float dx = target.x - camera.x;
    float dy = target.y - camera.y;
    float dz = target.z - camera.z;
    float yaw = atan2f(dy, dx) * 57.29578f;
    float pitch = atan2f(dz, sqrtf(dx*dx + dy*dy)) * 57.29578f;
    WriteMem<float>(localBase + Offset::AIM_YAW, yaw);
    WriteMem<float>(localBase + Offset::AIM_PITCH, pitch);
}

// ================== VONG LAP CHINH ==================
void AimLoop() {
    uintptr_t localBase = ReadMem<uintptr_t>(Offset::LOCAL_PLAYER);
    if (!localBase) {
        LOGE("Khong tim thay local player");
        return;
    }

    while (cfg.RUNNING) {
        float vm[16];
        for (int i = 0; i < 16; i++)
            vm[i] = ReadMem<float>(Offset::VIEW_MATRIX + i * 4);
        int sw = ReadMem<int>(Offset::SCREEN_W);
        int sh = ReadMem<int>(Offset::SCREEN_H);

        uintptr_t target = FindBestTarget(localBase, vm, sw, sh);
        if (target) {
            Vec3 tp = ReadPos(target);
            Vec3 cp = ReadPos(localBase + Offset::CAMERA_POS);
            WriteAim(localBase, tp, cp);
            WriteMem<int>(localBase + Offset::SHOOT_FLAG, 1);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// ================== ENTRY POINT (constructor) ==================
__attribute__((constructor))
static void on_load() {
    LOGI("libaimlock loaded");
    pid_t pid = FindPid("com.dts.freefireth");
    if (pid < 0) pid = FindPid("com.dts.freefiremax");
    if (pid < 0) {
        LOGE("Khong tim thay Free Fire");
        return;
    }
    if (!OpenTarget(pid)) return;
    std::thread(AimLoop).detach();
}

__attribute__((destructor))
static void on_unload() {
    LOGI("libaimlock unloaded");
    cfg.RUNNING = false;
    if (g_pid > 0) ptrace(PTRACE_DETACH, g_pid, 0, 0);
    if (g_memfd >= 0) close(g_memfd);
}
