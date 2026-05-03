/*
 * @Filename: demo_face.cpp
 * @Author: Hongying He
 * @Email: hongying.he@smartsenstech.com
 * @Date: 2025-12-30 14-57-47
 * @Copyright (c) 2025 SmartSens
 *
 * 优化版：针对 NPU INT8 量化噪声设计的 4 层抗抖管线
 *   第1层: 时序中值预滤波（3帧滑窗取中位数，去除异常值）
 *   第2层: Kalman 滤波（大R值，重度平滑NPU噪声）
 *   第3层: 输出量化（snap-to-pixel，消除亚像素抖动）
 *   第4层: OSD 限频重绘（限制重绘频率，消除闪烁）
 */
#include <fstream>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <fcntl.h>
#include <regex>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <utility>
#include <limits>
#include "include/utils.hpp"

using namespace std;

// 全局变量和同步对象
std::mutex mtx_image;                           // 图像队列互斥锁
std::condition_variable cv_image_ready;          // 图像就绪条件变量
std::atomic<bool> stop_inference(false);         // 停止推理标志
std::atomic<int> frame_count(0);                 // 帧计数器
std::atomic<int> g_eye_draw_mode(1);             // 0: circle, 1: box
VISUALIZER* g_visualizer = nullptr;              // 全局可视化器指针

namespace {
// ============================================================================
// 配置参数（平衡低延迟 + 抗NPU噪声）
// ============================================================================
constexpr float kEyeInferConfThreshold = 0.10f;    // 推理阈值
constexpr float kEyeDisplayScoreThreshold = 0.20f; // 首次显示阈值
constexpr float kEyeDisplayScoreThresholdTracked = 0.12f; // 已跟踪后保持阈值（提高以抑制漂移噪声）
constexpr float kEyeMinBoxSize = 3.0f;             // 过滤极小框
constexpr int   kEyeHoldFrames = 0;                // 丢检不保持，避免预测漂移
constexpr int   kClearAfterMissFrames = 3;          // 连续空帧后清屏，避免残留漂移
constexpr float kEyeDotFixedRadius = 3.0f;          // 固定眼点半径（缩小为原来的3像素，避免过大遮挡）
constexpr float kEyeOutputDeadzonePx = 0.60f;       // 小位移直接保持，减少静态抖动
constexpr float kEyeOutputMotionScalePx = 4.5f;     // 超出 deadzone 后的过渡尺度
constexpr float kEyeOutputSmoothAlphaCalm = 0.18f;  // 静止时更稳
constexpr float kEyeOutputSmoothAlphaActive = 0.72f;  // 运动时更跟手
constexpr float kEyeOutputSizeAlpha = 0.24f;        // 眼框尺寸输出平滑
constexpr float kFaceInferConfThreshold = 0.45f;    // 人脸检测阈值
constexpr int   kFaceInferInterval = 3;             // 人脸ROI刷新频率，降低ROI过期风险
constexpr int   kFaceRoiHoldFrames = 10;            // 人脸ROI保持
constexpr float kPoseInferConfThreshold = 0.22f;          // 手势/pose 推理阈值
constexpr float kPoseDisplayScoreThreshold = 0.62f;       // 手势首次显示/新目标阈值
constexpr float kPoseDisplayScoreThresholdTracked = 0.48f;  // 已锁定同类目标保持阈值
constexpr float kPoseDisplayScoreThresholdSwitch = 0.70f;   // 切类阈值
constexpr float kPoseImmediateAcceptScore = 0.82f;        // 超高置信度单帧直通阈值
constexpr int   kPoseInferInterval = 3;                   // 手势推理调度周期
constexpr int   kPoseInferPhase = 1;                      // 稳态时仅在该 phase 推理
constexpr int   kPoseStableAgeForSparseInfer = 4;         // 稳态判定帧数
constexpr float kPoseStableScoreForSparseInfer = 0.74f;   // 稳态判定分数
constexpr int   kPoseAcquireConfirmFrames = 2;            // 新目标上屏前需要连续确认
constexpr int   kPoseAcquireClearAfterMissFrames = 1;     // 未上屏候选允许的空帧数
constexpr int   kPoseHoldFrames = 3;                      // 手势结果短时保持，覆盖分时推理空窗
constexpr int   kPoseClearAfterMissFrames = 3;            // 连续空帧后清屏
constexpr float kPoseTrackSmoothAlpha = 0.34f;            // 手势框基础平滑系数
constexpr float kPoseTrackSmoothAlphaFast = 0.72f;        // 手势快速运动时的跟随系数
constexpr float kPoseTrackMatchMinIoU = 0.08f;            // 手势跟踪最小IoU
constexpr float kPoseTrackMaxCenterDistPx = 128.0f;       // 手势跟踪最大中心距离
constexpr float kPoseAcquireMatchMinIoU = 0.03f;          // 候选确认阶段最小IoU
constexpr float kPoseAcquireMaxCenterDistPx = 112.0f;     // 候选确认阶段最大中心距离
constexpr float kPoseClassSwitchMargin = 0.08f;           // 切换类别需要的额外置信优势
constexpr int   kPoseClassSwitchConfirmFrames = 2;        // 切换类别需要连续确认次数
constexpr int   kPoseOkClassId = 1;                       // OK 手势类别
constexpr float kPoseOkMetricBias = 0.06f;                // OK 候选优先级偏置
constexpr float kPoseOkThresholdRelax = 0.06f;            // OK 过滤阈值放宽
constexpr float kPoseOkFastAcquireScore = 0.74f;          // OK 高分时快速上屏
constexpr float kPoseOkSwitchInMarginRelax = 0.04f;       // 切入 OK 时降低门槛
constexpr float kPoseOkSwitchOutExtraMargin = 0.06f;      // 从 OK 切走时提高门槛
constexpr int   kPoseOkSwitchOutConfirmFrames = 3;        // 从 OK 切走时需要更多确认

// ============================================================================
// Kalman滤波参数（针对CPU高精度亚像素坐标，必须大幅干掉R以消除延迟！）
// CPU直接算出来的是精准质心，所以 R 应当极小，否则会导致滞后(漂移)
// ============================================================================
constexpr float kKalmanQPos = 2.0f;      // 位置过程噪声（大幅提高以紧紧跟手）
constexpr float kKalmanQVel = 0.5f;      // 速度过程噪声
constexpr float kKalmanRCalm = 4.0f;     // 静止时过滤眼皮微表情带来的阈值扰动
constexpr float kKalmanRActive = 0.5f;   // 运动时绝对信任当前帧（消除拖尾/漂移）
constexpr float kResidualThresh = 3.0f;  // 残差超过3px就认为是主动运动

// IoU匹配参数
constexpr float kIoUMatchThreshold = 0.10f;
constexpr float kTrackMaxCenterDistMinPx = 14.0f;
constexpr float kTrackMaxCenterDistScale = 1.8f;
constexpr float kTrackAreaRatioMin = 0.35f;
constexpr float kTrackAreaRatioMax = 2.80f;

// 输出量化参数
constexpr float kOutputSnapGrid = 1.0f;    // 坐标取整到像素

// OSD 重绘参数（移除限频，仅保留位移门控）
constexpr float kRedrawMinDelta = 1.5f;     // 移动<1.5px不重绘

// 中值预滤波已移除：它增加 1-2 帧延迟，用残差自适应 Kalman 替代

// CPU 轻量精定位参数：在 NPU 眼框（包含整个眼睛/眉毛轮廓）内寻找最暗区（瞳孔）中心
constexpr bool  kEnableCpuPupilRefine = true;
// 因为 NPU 现在框的是大眼睛范围，边缘会包含眼影、眉毛、或者皮肤
constexpr float kPupilInnerMarginXRatio = 0.10f;      // 适度切除
constexpr float kPupilInnerTopMarginRatio = 0.20f;    // 适度切除顶部眉毛
constexpr float kPupilInnerBottomMarginRatio = 0.10f; // 适度切除底部
// 放宽提取域，防止光线暗时瞳孔破碎导致找不到
constexpr float kPupilDarkPercentile = 0.15f;         
constexpr int   kPupilThresholdBias = 6;              // 阈值自适应偏移
constexpr float kPupilMinAreaRatio = 0.0025f;
constexpr float kPupilMaxAreaRatio = 0.50f;
// 允许瞳孔在整个眼眶范围内任意游走（不再被限制在中心）
constexpr float kPupilMaxShiftRatio = 0.50f;          // 允许偏离中心 50%（斜视时瞳孔在框的边缘）
constexpr float kPupilMaxShiftPixels = 15.0f;         // 绝对允许偏移量放宽
// 核心：100% 相信 CPU 给出的无抖动精确坐标！！绝不动摇！！
constexpr float kPupilBlendRatio = 0.84f;
constexpr int   kPupilMinRoiSide = 10;

// ============================================================================
// 残差自适应 Kalman 滤波器
// 核心思路：用测量残差(预测值与测量值的差)判断是静止还是运动
//   残差小 → 静止/NPU噪声 → R大 → 平滑
//   残差大 → 真实运动      → R小 → 瞬间跟手
// ============================================================================
struct KalmanLite1D {
    bool initialized = false;
    float x = 0.0f;   // 位置估计
    float v = 0.0f;   // 速度估计
    float P00 = 100.0f;
    float P01 = 0.0f;
    float P11 = 100.0f;

    void Init(float z) {
        initialized = true;
        x = z;
        v = 0.0f;
        P00 = 20.0f;
        P01 = 0.0f;
        P11 = 20.0f;
    }

    void Predict(float q_pos, float q_vel) {
        x = x + v;
        P00 = P00 + 2.0f * P01 + P11 + q_pos;
        P01 = P01 + P11;
        P11 = P11 + q_vel;
    }

    // 残差自适应更新：R 根据残差大小动态决定
    void Update(float z) {
        if (!initialized) {
            Init(z);
            return;
        }
        float residual = std::abs(z - x);
        // 残差大 → 真正在动 → R小(跟手)
        // 残差小 → NPU噪声 → R大(平滑)
        float R;
        if (residual > kResidualThresh) {
            R = kKalmanRActive;   // 运动：R=5，立即跟上
        } else {
            // 在 calm 和 active 之间平滑过渡
            float t = residual / kResidualThresh;
            R = kKalmanRCalm * (1.0f - t * t) + kKalmanRActive * (t * t);
        }

        float y = z - x;
        float S = P00 + R;
        if (S < 1e-6f) S = 1e-6f;
        float K0 = P00 / S;
        float K1 = P01 / S;
        x = x + K0 * y;
        v = v + K1 * y;
        P00 = (1.0f - K0) * P00;
        P01 = (1.0f - K0) * P01;
        P11 = P11 - K1 * P01;
    }

    float Speed() const { return std::abs(v); }
};

// ============================================================================
// 眼睛跟踪器（精简版：无中值预滤波 → 零额外延迟）
// ============================================================================
struct EyeTrack {
    KalmanLite1D kcx, kcy, kw, kh;
    int miss = 0;
    int age = 0;
    bool output_ready = false;
    float out_cx = 0.0f;
    float out_cy = 0.0f;
    float out_w = 1.0f;
    float out_h = 1.0f;

    bool IsReady() const {
        return kcx.initialized && kcy.initialized;
    }

    void Reset() {
        kcx = KalmanLite1D();
        kcy = KalmanLite1D();
        kw = KalmanLite1D();
        kh = KalmanLite1D();
        miss = 0;
        age = 0;
        output_ready = false;
        out_cx = 0.0f;
        out_cy = 0.0f;
        out_w = 1.0f;
        out_h = 1.0f;
    }

    void Predict() {
        if (!IsReady()) return;
        kcx.Predict(kKalmanQPos, kKalmanQVel);
        kcy.Predict(kKalmanQPos, kKalmanQVel);
        kw.Predict(kKalmanQPos * 0.2f, kKalmanQVel * 0.1f);
        kh.Predict(kKalmanQPos * 0.2f, kKalmanQVel * 0.1f);
    }

    void UpdateOutputState() {
        const float target_cx = kcx.x;
        const float target_cy = kcy.x;
        const float target_w = std::max(1.0f, kw.x);
        const float target_h = std::max(1.0f, kh.x);

        if (!output_ready) {
            output_ready = true;
            out_cx = target_cx;
            out_cy = target_cy;
            out_w = target_w;
            out_h = target_h;
        } else {
            const float dx = target_cx - out_cx;
            const float dy = target_cy - out_cy;
            const float dist = std::sqrt(dx * dx + dy * dy);
            if (dist > kEyeOutputDeadzonePx) {
                const float motion =
                    Clamp01((dist - kEyeOutputDeadzonePx) / std::max(0.5f, kEyeOutputMotionScalePx));
                const float alpha =
                    kEyeOutputSmoothAlphaCalm +
                    (kEyeOutputSmoothAlphaActive - kEyeOutputSmoothAlphaCalm) * motion;
                out_cx += alpha * dx;
                out_cy += alpha * dy;
            }
            out_w += kEyeOutputSizeAlpha * (target_w - out_w);
            out_h += kEyeOutputSizeAlpha * (target_h - out_h);
        }

        out_cx = std::round(out_cx * 2.0f) * 0.5f;
        out_cy = std::round(out_cy * 2.0f) * 0.5f;
        out_w = std::max(1.0f, out_w);
        out_h = std::max(1.0f, out_h);
    }

    void UpdateWithBox(const std::array<float, 4>& box) {
        const float bw = std::max(1.0f, box[2] - box[0]);
        const float bh = std::max(1.0f, box[3] - box[1]);
        const float bx = 0.5f * (box[0] + box[2]);
        const float by = 0.5f * (box[1] + box[3]);

        // 直接 Predict + Update，零额外延迟
        Predict();
        kcx.Update(bx);
        kcy.Update(by);
        kw.Update(bw);
        kh.Update(bh);
        UpdateOutputState();
        miss = 0;
        age += 1;
    }

    std::array<float, 4> ToBox() const {
        float bw = std::max(1.0f, kw.x);
        float bh = std::max(1.0f, kh.x);
        
        // 增加 0.5px 极小静区(Deadzone)：如果微小抖动，直接抹平，解决静止时晃动
        float cx = kcx.x;
        float cy = kcy.x;
        // 把最终输出稍微对齐到 0.5 亚像素网络，防止显示层剧烈横跳
        cx = std::round(cx * 2.0f) * 0.5f;
        cy = std::round(cy * 2.0f) * 0.5f;

        return {cx - 0.5f * bw, cy - 0.5f * bh,
                cx + 0.5f * bw, cy + 0.5f * bh};
    }

    std::array<float, 4> ToDisplayBox() const {
        if (!output_ready) {
            return ToBox();
        }
        return {out_cx - 0.5f * out_w, out_cy - 0.5f * out_h,
                out_cx + 0.5f * out_w, out_cy + 0.5f * out_h};
    }
};

float Clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

bool IsOkPoseClass(int cls) {
    return cls == kPoseOkClassId;
}

float BoxWidthFast(const std::array<float, 4>& box) {
    return std::max(0.0f, box[2] - box[0]);
}

float BoxHeightFast(const std::array<float, 4>& box) {
    return std::max(0.0f, box[3] - box[1]);
}

float BoxAreaFast(const std::array<float, 4>& box) {
    return BoxWidthFast(box) * BoxHeightFast(box);
}

float BoxCenterXFast(const std::array<float, 4>& box) {
    return 0.5f * (box[0] + box[2]);
}

float BoxCenterYFast(const std::array<float, 4>& box) {
    return 0.5f * (box[1] + box[3]);
}

float BoxCenterDistanceFast(const std::array<float, 4>& a,
                            const std::array<float, 4>& b) {
    const float dx = BoxCenterXFast(a) - BoxCenterXFast(b);
    const float dy = BoxCenterYFast(a) - BoxCenterYFast(b);
    return std::sqrt(dx * dx + dy * dy);
}

float BoxIoUFast(const std::array<float, 4>& a,
                 const std::array<float, 4>& b) {
    const float ix1 = std::max(a[0], b[0]);
    const float iy1 = std::max(a[1], b[1]);
    const float ix2 = std::min(a[2], b[2]);
    const float iy2 = std::min(a[3], b[3]);
    const float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    const float uni = BoxAreaFast(a) + BoxAreaFast(b) - inter;
    if (uni < 1e-6f) {
        return 0.0f;
    }
    return inter / uni;
}

std::array<float, 4> BlendBoxes(const std::array<float, 4>& prev_box,
                                const std::array<float, 4>& det_box,
                                float alpha) {
    const float a = Clamp01(alpha);
    std::array<float, 4> out = prev_box;
    for (int i = 0; i < 4; ++i) {
        out[i] = (1.0f - a) * prev_box[i] + a * det_box[i];
    }
    return out;
}

bool PoseSpatialMatch(const std::array<float, 4>& ref_box,
                      const std::array<float, 4>& det_box,
                      float min_iou,
                      float max_center_dist_px) {
    if (BoxIoUFast(ref_box, det_box) >= min_iou) {
        return true;
    }
    return BoxCenterDistanceFast(ref_box, det_box) <= max_center_dist_px;
}

float PoseCenterPreference(const std::array<float, 4>& box,
                           float img_w,
                           float img_h) {
    const float half_w = std::max(1.0f, img_w * 0.5f);
    const float half_h = std::max(1.0f, img_h * 0.5f);
    const float nx = std::fabs(BoxCenterXFast(box) - img_w * 0.5f) / half_w;
    const float ny = std::fabs(BoxCenterYFast(box) - img_h * 0.5f) / half_h;
    return Clamp01(1.0f - (0.65f * nx + 0.35f * ny));
}

float PoseAreaPreference(const std::array<float, 4>& box,
                         float img_w,
                         float img_h) {
    const float img_area = std::max(1.0f, img_w * img_h);
    const float area_ratio = BoxAreaFast(box) / img_area;
    const float target_ratio = 0.065f;
    const float tolerance = 0.065f;
    return Clamp01(1.0f - std::fabs(area_ratio - target_ratio) / tolerance);
}

float PoseQualityBias(const std::array<float, 4>& box,
                      float img_w,
                      float img_h) {
    const float center_pref = PoseCenterPreference(box, img_w, img_h);
    const float area_pref = PoseAreaPreference(box, img_w, img_h);
    return 0.55f * center_pref + 0.45f * area_pref;
}

float ComputePoseTrackAlpha(const std::array<float, 4>& prev_box,
                            const std::array<float, 4>& det_box,
                            float det_score,
                            bool same_class) {
    const float prev_size = std::max(18.0f, std::sqrt(std::max(1.0f, BoxAreaFast(prev_box))));
    const float motion = Clamp01(BoxCenterDistanceFast(prev_box, det_box) / (0.65f * prev_size + 18.0f));
    const float confidence = Clamp01((det_score - kPoseDisplayScoreThresholdTracked) / 0.30f);
    float alpha = kPoseTrackSmoothAlpha +
                  (kPoseTrackSmoothAlphaFast - kPoseTrackSmoothAlpha) * (0.70f * motion + 0.30f * confidence);
    if (!same_class) {
        alpha = std::max(alpha, 0.58f);
    }
    return Clamp01(alpha);
}

struct PoseTrack {
    bool active = false;
    std::array<float, 4> box = {0.0f, 0.0f, 0.0f, 0.0f};
    int cls = -1;
    float score = 0.0f;
    int miss = 0;
    int age = 0;
    int pending_cls = -1;
    int pending_count = 0;
    bool acquire_active = false;
    std::array<float, 4> acquire_box = {0.0f, 0.0f, 0.0f, 0.0f};
    int acquire_cls = -1;
    float acquire_score = 0.0f;
    int acquire_hits = 0;
    int acquire_miss = 0;

    void ResetAcquire() {
        acquire_active = false;
        acquire_box = {0.0f, 0.0f, 0.0f, 0.0f};
        acquire_cls = -1;
        acquire_score = 0.0f;
        acquire_hits = 0;
        acquire_miss = 0;
    }

    void Reset() {
        active = false;
        box = {0.0f, 0.0f, 0.0f, 0.0f};
        cls = -1;
        score = 0.0f;
        miss = 0;
        age = 0;
        pending_cls = -1;
        pending_count = 0;
        ResetAcquire();
    }

    bool IsStable() const {
        return active &&
               age >= kPoseStableAgeForSparseInfer &&
               score >= kPoseStableScoreForSparseInfer &&
               pending_count == 0 &&
               miss == 0;
    }

    void Init(const std::array<float, 4>& det_box, int det_cls, float det_score) {
        active = true;
        box = det_box;
        cls = det_cls;
        score = det_score;
        miss = 0;
        age = 1;
        pending_cls = -1;
        pending_count = 0;
        ResetAcquire();
    }

    void NoteMiss() {
        if (active) {
            miss += 1;
            pending_cls = -1;
            pending_count = 0;
            return;
        }
        if (acquire_active) {
            acquire_miss += 1;
            if (acquire_miss > kPoseAcquireClearAfterMissFrames) {
                ResetAcquire();
            }
        }
    }

    void UpdateAcquire(const std::array<float, 4>& det_box, int det_cls, float det_score) {
        if (det_score >= kPoseImmediateAcceptScore ||
            (IsOkPoseClass(det_cls) && det_score >= kPoseOkFastAcquireScore)) {
            Init(det_box, det_cls, det_score);
            return;
        }

        if (!acquire_active) {
            acquire_active = true;
            acquire_box = det_box;
            acquire_cls = det_cls;
            acquire_score = det_score;
            acquire_hits = 1;
            acquire_miss = 0;
            return;
        }

        const bool same_cls = (det_cls == acquire_cls);
        const bool spatial_match = PoseSpatialMatch(acquire_box,
                                                    det_box,
                                                    kPoseAcquireMatchMinIoU,
                                                    kPoseAcquireMaxCenterDistPx);
        if (same_cls && spatial_match) {
            acquire_box = BlendBoxes(acquire_box, det_box, 0.45f);
            acquire_score = std::max(det_score, 0.60f * acquire_score + 0.40f * det_score);
            acquire_hits += 1;
            acquire_miss = 0;
            if (acquire_hits >= kPoseAcquireConfirmFrames) {
                Init(acquire_box, acquire_cls, acquire_score);
            }
            return;
        }

        const float replace_margin = spatial_match ? 0.10f : 0.04f;
        if (!same_cls || det_score >= acquire_score + replace_margin) {
            acquire_box = det_box;
            acquire_cls = det_cls;
            acquire_score = det_score;
            acquire_hits = 1;
            acquire_miss = 0;
        }
    }

    void Update(const std::array<float, 4>& det_box, int det_cls, float det_score) {
        if (!active) {
            UpdateAcquire(det_box, det_cls, det_score);
            return;
        }

        const float prev_score = score;
        const bool same_cls = (det_cls == cls);
        box = BlendBoxes(box, det_box, ComputePoseTrackAlpha(box, det_box, det_score, same_cls));
        score = same_cls ? (0.55f * score + 0.45f * det_score)
                         : (0.70f * score + 0.30f * det_score);
        miss = 0;
        age += 1;

        if (same_cls) {
            pending_cls = -1;
            pending_count = 0;
            return;
        }

        if (pending_cls != det_cls) {
            pending_cls = det_cls;
            pending_count = 1;
        } else {
            pending_count += 1;
        }

        float switch_threshold =
            std::max(kPoseDisplayScoreThresholdSwitch, prev_score + kPoseClassSwitchMargin);
        int switch_confirm_frames = kPoseClassSwitchConfirmFrames;
        if (IsOkPoseClass(det_cls)) {
            switch_threshold -= kPoseOkSwitchInMarginRelax;
        }
        if (IsOkPoseClass(cls) && !IsOkPoseClass(det_cls)) {
            switch_threshold += kPoseOkSwitchOutExtraMargin;
            switch_confirm_frames = std::max(switch_confirm_frames, kPoseOkSwitchOutConfirmFrames);
        }

        if (det_score >= switch_threshold ||
            pending_count >= switch_confirm_frames) {
            cls = det_cls;
            score = std::max(det_score, score);
            pending_cls = -1;
            pending_count = 0;
        }
    }
};

// ============================================================================
// 全局状态
// ============================================================================
EyeTrack g_eye_tracks[2];
PoseTrack g_pose_track;
std::vector<std::array<float, 4>> g_last_drawn_boxes;
std::vector<int> g_last_drawn_classes;
int g_last_draw_frame = -10000;
bool g_has_active_overlay = false;
int g_consecutive_empty_frames = 0;

bool ShouldRunPoseInference(int frame_id, bool has_pose_overlay) {
    const int phase = frame_id % kPoseInferInterval;
    if (!has_pose_overlay || !g_pose_track.active) {
        return phase != 0;
    }
    return g_pose_track.IsStable() ? (phase == kPoseInferPhase) : (phase != 0);
}

// ============================================================================
// 工具函数
// ============================================================================

float BoxArea(const std::array<float, 4>& b) {
    return std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
}

float ComputeIoU(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    float ix1 = std::max(a[0], b[0]);
    float iy1 = std::max(a[1], b[1]);
    float ix2 = std::min(a[2], b[2]);
    float iy2 = std::min(a[3], b[3]);
    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float uni = BoxArea(a) + BoxArea(b) - inter;
    if (uni < 1e-6f) return 0.0f;
    return inter / uni;
}

float CenterDist2(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    // use true center difference to avoid 4x amplified distance gate
    float dx = 0.5f * ((a[0] + a[2]) - (b[0] + b[2]));
    float dy = 0.5f * ((a[1] + a[3]) - (b[1] + b[3]));
    return dx * dx + dy * dy;
}

int SelectPrimaryPoseIndex(const std::vector<std::array<float, 4>>& boxes,
                           const std::vector<int>& class_ids,
                           const std::vector<float>& scores,
                           float img_w,
                           float img_h) {
    const size_t n = std::min(boxes.size(), std::min(class_ids.size(), scores.size()));
    if (n == 0) {
        return -1;
    }

    int best_idx = -1;
    float best_metric = -std::numeric_limits<float>::infinity();
    int fallback_idx = -1;
    float fallback_metric = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < n; ++i) {
        const float quality_bias = 0.08f * PoseQualityBias(boxes[i], img_w, img_h);
        const float class_bias = IsOkPoseClass(class_ids[i]) ? kPoseOkMetricBias : 0.0f;
        const float base_metric = scores[i] + quality_bias + class_bias;
        if (base_metric > fallback_metric) {
            fallback_metric = base_metric;
            fallback_idx = static_cast<int>(i);
        }

        float metric = base_metric;
        if (g_pose_track.active) {
            const float iou = ComputeIoU(g_pose_track.box, boxes[i]);
            const float dist2 = CenterDist2(g_pose_track.box, boxes[i]);
            const float dist = std::sqrt(std::max(0.0f, dist2));
            const float proximity =
                std::max(0.0f, 1.0f - dist / std::max(1.0f, kPoseTrackMaxCenterDistPx));
            if (iou < kPoseTrackMatchMinIoU && proximity <= 0.0f) {
                continue;
            }
            metric += 0.28f * iou + 0.18f * proximity;
            if (class_ids[i] == g_pose_track.cls) {
                metric += 0.06f;
            }
        } else if (g_pose_track.acquire_active) {
            const float iou = BoxIoUFast(g_pose_track.acquire_box, boxes[i]);
            const float dist = BoxCenterDistanceFast(g_pose_track.acquire_box, boxes[i]);
            const float proximity =
                std::max(0.0f, 1.0f - dist / std::max(1.0f, kPoseAcquireMaxCenterDistPx));
            metric += 0.18f * iou + 0.10f * proximity;
            if (class_ids[i] == g_pose_track.acquire_cls) {
                metric += 0.05f;
            }
        }

        if (metric > best_metric) {
            best_metric = metric;
            best_idx = static_cast<int>(i);
        }
    }
    if (best_idx >= 0) {
        return best_idx;
    }
    if (fallback_idx >= 0 && scores[fallback_idx] >= kPoseImmediateAcceptScore) {
        return fallback_idx;
    }
    return -1;
}

bool SelectPrimaryFaceRoi(const FaceDetectionResult& face_result,
                          std::array<float, 4>* face_box_out) {
    if (face_result.boxes.empty() || face_result.scores.empty()) {
        return false;
    }
    size_t best_idx = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    const size_t n = std::min(face_result.boxes.size(), face_result.scores.size());
    for (size_t i = 0; i < n; ++i) {
        if (face_result.scores[i] > best_score) {
            best_score = face_result.scores[i];
            best_idx = i;
        }
    }
    *face_box_out = face_result.boxes[best_idx];
    return true;
}

std::vector<std::array<float, 4>> FilterEyesByFaceRoi(
    const std::vector<std::array<float, 4>>& eye_boxes,
    const std::array<float, 4>& face_box,
    float img_w,
    float img_h) {
    const float fx1 = std::max(0.0f, std::min(face_box[0], img_w - 1.0f));
    const float fy1 = std::max(0.0f, std::min(face_box[1], img_h - 1.0f));
    const float fx2 = std::max(0.0f, std::min(face_box[2], img_w - 1.0f));
    const float fy2 = std::max(0.0f, std::min(face_box[3], img_h - 1.0f));
    const float fw = std::max(1.0f, fx2 - fx1);
    const float fh = std::max(1.0f, fy2 - fy1);
    const float roi_x1 = std::max(0.0f, fx1 - 0.08f * fw);
    const float roi_x2 = std::min(img_w - 1.0f, fx2 + 0.08f * fw);
    const float roi_y1 = std::max(0.0f, fy1 - 0.05f * fh);
    const float roi_y2 = std::min(img_h - 1.0f, fy1 + 0.62f * fh);
    std::vector<std::array<float, 4>> filtered;
    filtered.reserve(eye_boxes.size());
    for (const auto& box : eye_boxes) {
        const float bw = std::max(1.0f, box[2] - box[0]);
        const float bh = std::max(1.0f, box[3] - box[1]);
        const float cx = 0.5f * (box[0] + box[2]);
        const float cy = 0.5f * (box[1] + box[3]);
        const bool in_roi = (cx >= roi_x1 && cx <= roi_x2 && cy >= roi_y1 && cy <= roi_y2);
        const bool size_ok = (bw <= 0.80f * fw && bh <= 0.80f * fh);
        if (in_roi && size_ok) {
            filtered.push_back(box);
        }
    }
    return filtered;
}

std::array<float, 4> ClampBoxToImage(const std::array<float, 4>& box, float img_w, float img_h) {
    std::array<float, 4> out = box;
    out[0] = std::max(0.0f, std::min(out[0], img_w - 1.0f));
    out[1] = std::max(0.0f, std::min(out[1], img_h - 1.0f));
    out[2] = std::max(0.0f, std::min(out[2], img_w - 1.0f));
    out[3] = std::max(0.0f, std::min(out[3], img_h - 1.0f));
    if (out[2] < out[0]) std::swap(out[2], out[0]);
    if (out[3] < out[1]) std::swap(out[3], out[1]);
    return out;
}

int ClampInt(int v, int lo, int hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

bool ComputeAdaptiveDarkThreshold(const uint8_t* y_plane,
                                  int img_w,
                                  int roi_x,
                                  int roi_y,
                                  int roi_w,
                                  int roi_h,
                                  int* threshold_out) {
    if (y_plane == nullptr || threshold_out == nullptr || roi_w <= 0 || roi_h <= 0) {
        return false;
    }

    std::array<int, 256> hist = {0};
    for (int y = 0; y < roi_h; ++y) {
        const uint8_t* row = y_plane + (roi_y + y) * img_w + roi_x;
        for (int x = 0; x < roi_w; ++x) {
            hist[row[x]] += 1;
        }
    }

    const int roi_area = roi_w * roi_h;
    const int dark_rank = std::max(1, static_cast<int>(roi_area * kPupilDarkPercentile));
    int cumulative = 0;
    int percentile_value = 0;
    for (int i = 0; i < 256; ++i) {
        cumulative += hist[i];
        if (cumulative >= dark_rank) {
            percentile_value = i;
            break;
        }
    }

    *threshold_out = ClampInt(percentile_value + kPupilThresholdBias, 0, 255);
    return true;
}

struct BlobCandidate {
    float score = -1.0f;
    float cx = 0.0f;
    float cy = 0.0f;
};

bool FindPupilCenterFromDarkBlob(const uint8_t* y_plane,
                                 int img_w,
                                 int roi_x,
                                 int roi_y,
                                 int roi_w,
                                 int roi_h,
                                 int threshold,
                                 float* cx_out,
                                 float* cy_out) {
    if (y_plane == nullptr || cx_out == nullptr || cy_out == nullptr) {
        return false;
    }
    if (roi_w <= 0 || roi_h <= 0) {
        return false;
    }

    const int roi_area = roi_w * roi_h;
    const int min_area = std::max(4, static_cast<int>(roi_area * kPupilMinAreaRatio));
    const int max_area = std::max(min_area + 1, static_cast<int>(roi_area * kPupilMaxAreaRatio));

    std::vector<uint8_t> visited(static_cast<size_t>(roi_area), 0);
    std::vector<int> queue;
    queue.reserve(static_cast<size_t>(roi_area));

    BlobCandidate best;
    const float roi_cx = static_cast<float>(roi_x) + 0.5f * static_cast<float>(roi_w - 1);
    const float roi_cy = static_cast<float>(roi_y) + 0.5f * static_cast<float>(roi_h - 1);

    for (int ry = 0; ry < roi_h; ++ry) {
        for (int rx = 0; rx < roi_w; ++rx) {
            const int seed_idx = ry * roi_w + rx;
            if (visited[seed_idx] != 0) {
                continue;
            }

            const uint8_t seed_val = y_plane[(roi_y + ry) * img_w + (roi_x + rx)];
            if (seed_val > threshold) {
                visited[seed_idx] = 1;
                continue;
            }

            queue.clear();
            queue.push_back(seed_idx);
            visited[seed_idx] = 1;

            int area = 0;
            float sum_x = 0.0f;
            float sum_y = 0.0f;
            float sum_intensity = 0.0f;
            int min_rx = rx;
            int min_ry = ry;
            int max_rx = rx;
            int max_ry = ry;
            bool touches_border = false;

            for (size_t q = 0; q < queue.size(); ++q) {
                const int idx = queue[q];
                const int cy = idx / roi_w;
                const int cx = idx - cy * roi_w;
                const int gx = roi_x + cx;
                const int gy = roi_y + cy;
                const uint8_t pix = y_plane[gy * img_w + gx];

                area += 1;
                sum_x += static_cast<float>(gx);
                sum_y += static_cast<float>(gy);
                sum_intensity += static_cast<float>(pix);

                min_rx = std::min(min_rx, cx);
                min_ry = std::min(min_ry, cy);
                max_rx = std::max(max_rx, cx);
                max_ry = std::max(max_ry, cy);
                if (cx == 0 || cy == 0 || cx == roi_w - 1 || cy == roi_h - 1) {
                    touches_border = true;
                }

                const int nx[4] = {cx - 1, cx + 1, cx, cx};
                const int ny[4] = {cy, cy, cy - 1, cy + 1};
                for (int k = 0; k < 4; ++k) {
                    if (nx[k] < 0 || ny[k] < 0 || nx[k] >= roi_w || ny[k] >= roi_h) {
                        continue;
                    }
                    const int nidx = ny[k] * roi_w + nx[k];
                    if (visited[nidx] != 0) {
                        continue;
                    }
                    const uint8_t npix = y_plane[(roi_y + ny[k]) * img_w + (roi_x + nx[k])];
                    if (npix <= threshold) {
                        visited[nidx] = 1;
                        queue.push_back(nidx);
                    } else {
                        visited[nidx] = 1;
                    }
                }
            }

            if (area < min_area || area > max_area) {
                continue;
            }

            const float blob_w = static_cast<float>(max_rx - min_rx + 1);
            const float blob_h = static_cast<float>(max_ry - min_ry + 1);
            const float aspect = blob_w / std::max(1.0f, blob_h);
            
            // 宽容的高宽比：哪怕眯眼、极度侧视，也不能丢！
            if (aspect < 0.15f || aspect > 4.00f) {
                continue;
            }

            // 宽容的填充率：防止反光点把瞳孔切成 c 形导致填充率过低被滤除
            const float fill_ratio = static_cast<float>(area) / (blob_w * blob_h);
            if (fill_ratio < 0.20f) {
                continue;
            }

            const float cx = sum_x / static_cast<float>(area);
            const float cy = sum_y / static_cast<float>(area);
            const float mean_dark = sum_intensity / static_cast<float>(area);
            const float dx = cx - roi_cx;
            const float dy = cy - roi_cy;
            const float dist2 = dx * dx + dy * dy;

            // 回归以“极暗度”为主，辅以面积，弱化距离惩罚（让瞳孔能走到视线边缘）
            float score = (255.0f - mean_dark) * std::sqrt(static_cast<float>(area)) * fill_ratio /
                          (1.0f + 0.002f * dist2);
                          
            // 触碰边界的稍微扣分（因为眼框裁取得很紧凑，真的瞳孔是有可能贴边的）
            if (touches_border) {
                score *= 0.65f;
            }

            if (score > best.score) {
                best.score = score;
                best.cx = cx;
                best.cy = cy;
            }
        }
    }

    if (best.score <= 0.0f) {
        return false;
    }

    *cx_out = best.cx;
    *cy_out = best.cy;
    return true;
}

bool RefineEyeBoxByCpuPupil(const std::array<float, 4>& eye_box,
                            const uint8_t* y_plane,
                            int img_w,
                            int img_h,
                            std::array<float, 4>* refined_box) {
    if (!kEnableCpuPupilRefine || y_plane == nullptr || refined_box == nullptr) {
        return false;
    }
    if (img_w <= 1 || img_h <= 1) {
        return false;
    }

    const float bw = std::max(1.0f, eye_box[2] - eye_box[0]);
    const float bh = std::max(1.0f, eye_box[3] - eye_box[1]);
    if (bw < static_cast<float>(kPupilMinRoiSide) || bh < static_cast<float>(kPupilMinRoiSide)) {
        return false;
    }

    const int roi_x1 = ClampInt(static_cast<int>(std::floor(eye_box[0] + bw * kPupilInnerMarginXRatio)), 0, img_w - 1);
    const int roi_x2 = ClampInt(static_cast<int>(std::ceil(eye_box[2] - bw * kPupilInnerMarginXRatio)), 0, img_w - 1);
    const int roi_y1 = ClampInt(static_cast<int>(std::floor(eye_box[1] + bh * kPupilInnerTopMarginRatio)), 0, img_h - 1);
    const int roi_y2 = ClampInt(static_cast<int>(std::ceil(eye_box[3] - bh * kPupilInnerBottomMarginRatio)), 0, img_h - 1);

    if (roi_x2 <= roi_x1 || roi_y2 <= roi_y1) {
        return false;
    }

    const int roi_w = roi_x2 - roi_x1 + 1;
    const int roi_h = roi_y2 - roi_y1 + 1;
    if (roi_w < kPupilMinRoiSide || roi_h < kPupilMinRoiSide) {
        return false;
    }

    int threshold = 0;
    if (!ComputeAdaptiveDarkThreshold(y_plane, img_w, roi_x1, roi_y1, roi_w, roi_h, &threshold)) {
        return false;
    }

    float refined_cx = 0.0f;
    float refined_cy = 0.0f;
    if (!FindPupilCenterFromDarkBlob(y_plane,
                                     img_w,
                                     roi_x1,
                                     roi_y1,
                                     roi_w,
                                     roi_h,
                                     threshold,
                                     &refined_cx,
                                     &refined_cy)) {
        return false;
    }

    const float orig_cx = 0.5f * (eye_box[0] + eye_box[2]);
    const float orig_cy = 0.5f * (eye_box[1] + eye_box[3]);
    const float dx = refined_cx - orig_cx;
    const float dy = refined_cy - orig_cy;
    const float max_shift = std::max(kPupilMaxShiftPixels, std::min(bw, bh) * kPupilMaxShiftRatio);
    if ((dx * dx + dy * dy) > (max_shift * max_shift)) {
        return false;
    }

    const float blended_cx = (1.0f - kPupilBlendRatio) * orig_cx + kPupilBlendRatio * refined_cx;
    const float blended_cy = (1.0f - kPupilBlendRatio) * orig_cy + kPupilBlendRatio * refined_cy;

    std::array<float, 4> out = {
        blended_cx - 0.5f * bw,
        blended_cy - 0.5f * bh,
        blended_cx + 0.5f * bw,
        blended_cy + 0.5f * bh
    };
    *refined_box = ClampBoxToImage(out, static_cast<float>(img_w), static_cast<float>(img_h));
    return true;
}

float CenterX(const std::array<float, 4>& box) {
    return 0.5f * (box[0] + box[2]);
}

std::vector<std::array<float, 4>> SortByCenterX(std::vector<std::array<float, 4>> boxes) {
    std::sort(boxes.begin(), boxes.end(), [](const std::array<float, 4>& a,
                                             const std::array<float, 4>& b) {
        return CenterX(a) < CenterX(b);
    });
    return boxes;
}

// ============================================================================
// 核心跟踪：中值预滤波 + IoU匹配 + Kalman + 输出量化
// ============================================================================
std::vector<std::array<float, 4>> GetStableEyeBoxes(
    const FaceDetectionResult& result,
    const std::array<float, 4>* face_roi,
    const uint8_t* y_plane,
    int img_w,
    int img_h) {
    // --- 筛选候选检测框 ---
    const bool has_track = g_eye_tracks[0].IsReady() || g_eye_tracks[1].IsReady();
    const float score_thresh = has_track ? kEyeDisplayScoreThresholdTracked : kEyeDisplayScoreThreshold;
    std::vector<std::pair<std::array<float, 4>, float>> candidates;
    candidates.reserve(result.boxes.size());
    for (size_t i = 0; i < result.boxes.size(); ++i) {
        const float score = (i < result.scores.size()) ? result.scores[i] : 1.0f;
        const auto& box = result.boxes[i];
        const float bw = std::max(0.0f, box[2] - box[0]);
        const float bh = std::max(0.0f, box[3] - box[1]);
        if (score >= score_thresh && bw >= kEyeMinBoxSize && bh >= kEyeMinBoxSize) {
            candidates.push_back({box, score});
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const std::pair<std::array<float, 4>, float>& a,
                 const std::pair<std::array<float, 4>, float>& b) {
                  return a.second > b.second;
              });
    if (candidates.size() > 4) {
        candidates.resize(4);
    }

    std::vector<std::array<float, 4>> det_boxes;
    det_boxes.reserve(candidates.size());
    for (const auto& item : candidates) {
        std::array<float, 4> box = item.first;
        if (face_roi != nullptr) {
            const float cx = 0.5f * (box[0] + box[2]);
            const float cy = 0.5f * (box[1] + box[3]);
            const float fw = std::max(1.0f, (*face_roi)[2] - (*face_roi)[0]);
            const float fh = std::max(1.0f, (*face_roi)[3] - (*face_roi)[1]);
            const float roi_x1 = std::max(0.0f, (*face_roi)[0] - 0.05f * fw);
            const float roi_x2 = std::min(static_cast<float>(img_w - 1), (*face_roi)[2] + 0.05f * fw);
            const float roi_y1 = std::max(0.0f, (*face_roi)[1] - 0.05f * fh);
            const float roi_y2 = std::min(static_cast<float>(img_h - 1), (*face_roi)[1] + 0.62f * fh);
            if (!(cx >= roi_x1 && cx <= roi_x2 && cy >= roi_y1 && cy <= roi_y2)) {
                continue;
            }
        }
        std::array<float, 4> refined_box;
        if (RefineEyeBoxByCpuPupil(box, y_plane, img_w, img_h, &refined_box)) {
            box = refined_box;
        }
        det_boxes.push_back(box);
    }

    // --- IoU + 中心距离匹配 ---
    std::vector<bool> det_used(det_boxes.size(), false);

    for (int t = 0; t < 2; ++t) {
        if (!g_eye_tracks[t].IsReady()) continue;

        std::array<float, 4> predicted = g_eye_tracks[t].ToBox();
        const float pbw = std::max(1.0f, predicted[2] - predicted[0]);
        const float pbh = std::max(1.0f, predicted[3] - predicted[1]);
        const float max_dist = std::max(kTrackMaxCenterDistMinPx,
                                        0.5f * (pbw + pbh) * kTrackMaxCenterDistScale);
        const float max_dist2 = max_dist * max_dist;
        int best_d = -1;
        float best_score = -1.0f;

        for (int d = 0; d < static_cast<int>(det_boxes.size()); ++d) {
            if (det_used[d]) continue;
            float iou = ComputeIoU(predicted, det_boxes[d]);
            const float p_area = std::max(1.0f, BoxArea(predicted));
            const float d_area = std::max(1.0f, BoxArea(det_boxes[d]));
            const float area_ratio = d_area / p_area;
            if (area_ratio < kTrackAreaRatioMin || area_ratio > kTrackAreaRatioMax) {
                continue;
            }
            if (iou < kIoUMatchThreshold) {
                float dist2 = CenterDist2(predicted, det_boxes[d]);
                if (dist2 <= max_dist2) {
                    const float proximity = 1.0f - std::min(1.0f, dist2 / (max_dist2 + 1e-6f));
                    iou = 0.02f + 0.20f * proximity;
                } else {
                    continue;
                }
            }
            if (iou > best_score) {
                best_score = iou;
                best_d = d;
            }
        }

        if (best_d >= 0) {
            g_eye_tracks[t].UpdateWithBox(det_boxes[best_d]);
            det_used[best_d] = true;
        } else {
            g_eye_tracks[t].Predict();
            g_eye_tracks[t].miss += 1;
            if (g_eye_tracks[t].miss > kEyeHoldFrames) {
                g_eye_tracks[t].Reset();
            }
        }
    }

    // --- 未匹配检测框分配给空跟踪器 ---
    for (int d = 0; d < static_cast<int>(det_boxes.size()); ++d) {
        if (det_used[d]) continue;
        for (int t = 0; t < 2; ++t) {
            if (!g_eye_tracks[t].IsReady()) {
                g_eye_tracks[t].UpdateWithBox(det_boxes[d]);
                det_used[d] = true;
                break;
            }
        }
    }

    // --- 输出稳定框 ---
    std::vector<std::array<float, 4>> stable;
    stable.reserve(2);
    for (int i = 0; i < 2; ++i) {
        if (g_eye_tracks[i].IsReady() && g_eye_tracks[i].miss == 0) {
            stable.push_back(
                ClampBoxToImage(g_eye_tracks[i].ToDisplayBox(),
                                static_cast<float>(img_w),
                                static_cast<float>(img_h)));
        }
    }
    return SortByCenterX(stable);
}

// OSD 重绘判断（无限频，仅位移门控防闪烁）
bool ShouldRedraw(const std::vector<std::array<float, 4>>& boxes,
                  const std::vector<int>* class_ids,
                  int frame_id) {
    if (g_last_draw_frame < 0) return true;
    if (boxes.size() != g_last_drawn_boxes.size()) return true;
    if (class_ids != nullptr) {
        if (class_ids->size() != g_last_drawn_classes.size()) return true;
        for (size_t i = 0; i < class_ids->size(); ++i) {
            if ((*class_ids)[i] != g_last_drawn_classes[i]) return true;
        }
    }
    for (size_t i = 0; i < boxes.size(); ++i) {
        const float dx = std::fabs(CenterX(boxes[i]) - CenterX(g_last_drawn_boxes[i]));
        const float dy = std::fabs((boxes[i][1] + boxes[i][3]) * 0.5f - (g_last_drawn_boxes[i][1] + g_last_drawn_boxes[i][3]) * 0.5f);
        if (dx > kRedrawMinDelta || dy > kRedrawMinDelta) return true;
    }
    return false;
}

std::vector<std::array<float, 4>> OffsetBoxesY(
    const std::vector<std::array<float, 4>>& boxes,
    float offset_y) {
    std::vector<std::array<float, 4>> shifted;
    shifted.reserve(boxes.size());
    for (const auto& box : boxes) {
        shifted.push_back({box[0], box[1] + offset_y, box[2], box[3] + offset_y});
    }
    return shifted;
}

const char* PoseClassName(int class_id) {
    switch (class_id) {
        case 0: return "up";
        case 1: return "ok";
        case 2: return "down";
        default: return "unknown";
    }
}

void FilterPoseDetectionsForDisplay(const FaceDetectionResult& result,
                                    float score_threshold,
                                    float img_w,
                                    float img_h,
                                    std::vector<std::array<float, 4>>* boxes_out,
                                    std::vector<int>* class_ids_out,
                                    std::vector<float>* scores_out) {
    boxes_out->clear();
    class_ids_out->clear();
    scores_out->clear();

    const size_t n = std::min(result.boxes.size(),
                              std::min(result.scores.size(), result.class_ids.size()));
    boxes_out->reserve(n);
    class_ids_out->reserve(n);
    scores_out->reserve(n);
    for (size_t i = 0; i < n; ++i) {
        float dynamic_threshold = score_threshold;
        if (IsOkPoseClass(result.class_ids[i])) {
            dynamic_threshold -= kPoseOkThresholdRelax;
        }
        if (g_pose_track.active) {
            const bool same_cls = (result.class_ids[i] == g_pose_track.cls);
            const bool spatial_match = PoseSpatialMatch(g_pose_track.box,
                                                        result.boxes[i],
                                                        kPoseTrackMatchMinIoU,
                                                        kPoseTrackMaxCenterDistPx);
            if (same_cls && spatial_match) {
                dynamic_threshold = kPoseDisplayScoreThresholdTracked;
            } else if (spatial_match) {
                dynamic_threshold = kPoseDisplayScoreThresholdSwitch;
            } else {
                dynamic_threshold = std::max(dynamic_threshold, kPoseDisplayScoreThresholdSwitch);
            }
        } else if (g_pose_track.acquire_active) {
            const bool same_cls = (result.class_ids[i] == g_pose_track.acquire_cls);
            const bool spatial_match = PoseSpatialMatch(g_pose_track.acquire_box,
                                                        result.boxes[i],
                                                        kPoseAcquireMatchMinIoU,
                                                        kPoseAcquireMaxCenterDistPx);
            if (same_cls && spatial_match) {
                dynamic_threshold = std::max(kPoseDisplayScoreThresholdTracked,
                                             score_threshold - 0.04f);
            }
        }

        const float quality_assisted_score =
            result.scores[i] + 0.06f * PoseQualityBias(result.boxes[i], img_w, img_h);
        if (quality_assisted_score < dynamic_threshold) {
            continue;
        }
        boxes_out->push_back(result.boxes[i]);
        class_ids_out->push_back(result.class_ids[i]);
        scores_out->push_back(result.scores[i]);
    }
}

void PrintPoseSummary(const std::vector<int>& class_ids,
                      const std::vector<float>& scores,
                      int frame_id) {
    if (class_ids.empty() || scores.empty()) {
        return;
    }
    size_t best_idx = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    const size_t n = std::min(class_ids.size(), scores.size());
    for (size_t i = 0; i < n; ++i) {
        if (scores[i] > best_score) {
            best_score = scores[i];
            best_idx = i;
        }
    }

    static int last_top_cls = -999;
    static int last_count = -1;
    static int last_log_frame = -1000;
    const int top_cls = class_ids[best_idx];
    const int count = static_cast<int>(n);
    const bool changed = (top_cls != last_top_cls) || (count != last_count);
    if (!changed && (frame_id - last_log_frame) < 30) {
        return;
    }

    last_top_cls = top_cls;
    last_count = count;
    last_log_frame = frame_id;
    printf("[POSE] frame=%d recognized=%s score=%.3f count=%d threshold=%.2f\n",
           frame_id, PoseClassName(top_cls), scores[best_idx], count, kPoseDisplayScoreThreshold);
}

std::array<float, 4> BuildEyeDotBox(const std::array<float, 4>& eye_box, float img_w, float img_h) {
    // 保持亚像素精度输入，只在最后显示前进行一次平滑
    const float cx = 0.5f * (eye_box[0] + eye_box[2]);
    const float cy = 0.5f * (eye_box[1] + eye_box[3]);
    const float r = kEyeDotFixedRadius;
    std::array<float, 4> out = {cx - r, cy - r, cx + r, cy + r};
    out[0] = std::max(0.0f, std::min(out[0], img_w - 1.0f));
    out[1] = std::max(0.0f, std::min(out[1], img_h - 1.0f));
    out[2] = std::max(0.0f, std::min(out[2], img_w - 1.0f));
    out[3] = std::max(0.0f, std::min(out[3], img_h - 1.0f));
    return out;
}
}


// 图像队列结构
struct ImagePair {
    ssne_tensor_t img1;
    ssne_tensor_t img2;
    int frame_id;
};

std::queue<ImagePair> image_queue;
const int MAX_QUEUE_SIZE = 1;

bool g_exit_flag = false;
std::mutex g_mtx;

void keyboard_listener() {
    std::string input;
    std::cout << "键盘监听线程已启动，输入 'q' 退出程序..." << std::endl;
    while (true) {
        std::cin >> input;
        std::lock_guard<std::mutex> lock(g_mtx);
        if (input == "q" || input == "Q") {
            g_exit_flag = true;
            std::cout << "检测到退出指令，通知主线程退出..." << std::endl;
            break;
        } else {
            std::cout << "输入无效（仅 'q' 有效），请重新输入：" << std::endl;
        }
    }
}

bool check_exit_flag() {
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_exit_flag;
}


/**
 * @brief 推理线程函数
 */
void inference_thread_func(POSEDETGRAY* pose_detector,
                           EYEDETGRAY* eye_detector,
                           SCRFDGRAY* face_detector,
                           int dual_display_offset_y,
                           int img_width,
                           int img_height) {
    cout << "[Thread] Inference thread started!" << endl;

    FaceDetectionResult* pose_result = new FaceDetectionResult;
    FaceDetectionResult* eye_result = new FaceDetectionResult;
    FaceDetectionResult* face_result = new FaceDetectionResult;
    std::array<float, 4> last_face_roi = {0.0f, 0.0f, 0.0f, 0.0f};
    bool has_face_roi = false;
    int face_miss_frames = 0;
    bool has_pose_overlay = false;
    int pose_miss_frames = 0;
    
    while (!stop_inference) {
        ImagePair img_pair;
        bool has_image = false;
        
        {
            std::unique_lock<std::mutex> lock(mtx_image);
            cv_image_ready.wait(lock, [] {
                return !image_queue.empty() || stop_inference;
            });
            if (stop_inference && image_queue.empty()) break;
            if (!image_queue.empty()) {
                img_pair = image_queue.front();
                image_queue.pop();
                has_image = true;
            }
        }
        
        if (!has_image) continue;

        const uint8_t* y_plane = reinterpret_cast<const uint8_t*>(get_data(img_pair.img1));
        const int face_infer_phase = img_pair.frame_id % kFaceInferInterval;
        // 人脸低频检测：phase 0
        if (face_detector != nullptr &&
            (face_infer_phase == 0 || !has_face_roi)) {
            face_detector->Predict(&img_pair.img1, face_result, kFaceInferConfThreshold);
            std::array<float, 4> current_face_roi;
            if (SelectPrimaryFaceRoi(*face_result, &current_face_roi)) {
                last_face_roi = current_face_roi;
                has_face_roi = true;
                face_miss_frames = 0;
            } else if (has_face_roi) {
                face_miss_frames += 1;
                if (face_miss_frames > kFaceRoiHoldFrames) {
                    has_face_roi = false;
                }
            }
        }

        // 眼睛实时检测：每帧
        eye_detector->Predict(&img_pair.img1, eye_result, kEyeInferConfThreshold);
        std::vector<std::array<float, 4>> stable_boxes =
            GetStableEyeBoxes(*eye_result,
                              has_face_roi ? &last_face_roi : nullptr,
                              y_plane,
                              img_width,
                              img_height);
        std::vector<std::array<float, 4>> eye_dot_boxes;
        eye_dot_boxes.reserve(stable_boxes.size());
        for (const auto& b : stable_boxes) {
            eye_dot_boxes.push_back(BuildEyeDotBox(
                b, static_cast<float>(img_width), static_cast<float>(img_height)));
        }

        // pose/手势低频检测：phase 1，与 face 交错
        if (pose_detector != nullptr &&
            ShouldRunPoseInference(img_pair.frame_id, has_pose_overlay)) {
            pose_detector->Predict(&img_pair.img2, pose_result, kPoseInferConfThreshold);
            std::vector<std::array<float, 4>> pose_display_boxes;
            std::vector<int> pose_display_classes;
            std::vector<float> pose_display_scores;
            FilterPoseDetectionsForDisplay(*pose_result,
                                           kPoseDisplayScoreThreshold,
                                           static_cast<float>(img_width),
                                           static_cast<float>(img_height),
                                           &pose_display_boxes,
                                           &pose_display_classes,
                                           &pose_display_scores);
            bool pose_updated = false;
            if (!pose_display_boxes.empty()) {
                const int best_idx = SelectPrimaryPoseIndex(
                    pose_display_boxes,
                    pose_display_classes,
                    pose_display_scores,
                    static_cast<float>(img_width),
                    static_cast<float>(img_height));
                if (best_idx >= 0) {
                    g_pose_track.Update(pose_display_boxes[best_idx],
                                        pose_display_classes[best_idx],
                                        pose_display_scores[best_idx]);
                    pose_updated = true;
                    has_pose_overlay = g_pose_track.active;
                    if (has_pose_overlay) {
                        pose_miss_frames = 0;
                        std::vector<int> stable_classes(1, g_pose_track.cls);
                        std::vector<float> stable_scores(1, g_pose_track.score);
                        PrintPoseSummary(stable_classes, stable_scores, img_pair.frame_id);
                    }
                }
            }
            if (!pose_updated) {
                g_pose_track.NoteMiss();
                if (has_pose_overlay) {
                    pose_miss_frames += 1;
                    if (pose_miss_frames > kPoseHoldFrames) {
                        g_pose_track.Reset();
                        has_pose_overlay = false;
                    }
                }
            }
        }

        std::vector<std::array<float, 4>> box_draw;
        std::vector<int> box_draw_classes;
        if (has_face_roi) {
            box_draw.push_back(last_face_roi);
            box_draw_classes.push_back(-1);
        }
        if (has_pose_overlay && g_pose_track.active) {
            std::vector<std::array<float, 4>> stable_pose_boxes(1, g_pose_track.box);
            std::vector<std::array<float, 4>> pose_boxes_display =
                OffsetBoxesY(stable_pose_boxes, static_cast<float>(dual_display_offset_y));
            box_draw.insert(box_draw.end(), pose_boxes_display.begin(), pose_boxes_display.end());
            box_draw_classes.push_back(g_pose_track.cls);
        }

        std::vector<std::array<float, 4>> redraw_boxes = box_draw;
        std::vector<int> redraw_classes = box_draw_classes;
        redraw_boxes.insert(redraw_boxes.end(), eye_dot_boxes.begin(), eye_dot_boxes.end());
        redraw_classes.insert(redraw_classes.end(), eye_dot_boxes.size(), -2);

        // 无检测时处理
        if (redraw_boxes.empty()) {
            g_consecutive_empty_frames += 1;
            if (g_visualizer != nullptr && g_has_active_overlay &&
                g_consecutive_empty_frames >= kPoseClearAfterMissFrames) {
                std::vector<std::array<float, 4>> empty;
                g_visualizer->Draw(empty);
                g_visualizer->DrawCircles(empty);
                g_has_active_overlay = false;
                g_last_drawn_boxes.clear();
                g_last_drawn_classes.clear();
                g_last_draw_frame = img_pair.frame_id;
                g_consecutive_empty_frames = 0;
            }
            continue;
        }

        g_consecutive_empty_frames = 0;

        // 第4层：OSD 限频重绘
        if (g_visualizer != nullptr &&
            ShouldRedraw(redraw_boxes, &redraw_classes, img_pair.frame_id)) {
            g_visualizer->Draw(box_draw, box_draw_classes);
            g_visualizer->DrawCircles(eye_dot_boxes);
            g_last_drawn_boxes = redraw_boxes;
            g_last_drawn_classes = redraw_classes;
            g_last_draw_frame = img_pair.frame_id;
            g_has_active_overlay = true;
        }
    }
    
    delete pose_result;
    delete eye_result;
    delete face_result;
    cout << "[Thread] Inference thread stopped!" << endl;
}

/**
 * @brief 主函数
 */
int main(int argc, char* argv[]) {
    uint8_t load_flag = 0;
    int img_width = 640;
    int img_height = 480;
    const int dual_display_offset_y = img_height;
    
    array<int, 2> eye_det_shape = {640, 480};
    string path_eye_det = "/app_demo/app_assets/models/eye.m1model";
    array<int, 2> pose_det_shape = {640, 480};
    string path_pose_det = "/app_demo/app_assets/models/pose.m1model";
    array<int, 2> face_det_shape = {640, 480};
    string path_face_det = "/app_demo/app_assets/models/face_640x480.m1model";
    g_eye_draw_mode = 0;
    printf("[INFO] Eye draw mode: circle (forced)\n");
    printf("[INFO] Pose class colors: up->1 ok->2 down->3\n");
    printf("[INFO] Pose thresholds: acquire=%.2f tracked=%.2f switch=%.2f immediate=%.2f\n",
           kPoseDisplayScoreThreshold,
           kPoseDisplayScoreThresholdTracked,
           kPoseDisplayScoreThresholdSwitch,
           kPoseImmediateAcceptScore);
    printf("[INFO] Pose infer schedule: fast=2/3 stable=1/3 acquire_confirm=%d hold=%d\n",
           kPoseAcquireConfirmFrames,
           kPoseHoldFrames);
    
    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
    }
    
    array<int, 2> img_shape = {img_width, img_height};
    array<int, 2> display_shape = {img_width, img_height * 2};
    
    VISUALIZER visualizer;
    visualizer.Initialize(display_shape);
    g_visualizer = &visualizer;

    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape);

    SCRFDGRAY face_detector;
    int face_box_len = face_det_shape[0] * face_det_shape[1];
    face_detector.Initialize(path_face_det, &img_shape, &face_det_shape, false, face_box_len);

    EYEDETGRAY eye_detector;
    int eye_box_len = eye_det_shape[0] * eye_det_shape[1];
    eye_detector.Initialize(path_eye_det, &img_shape, &eye_det_shape, eye_box_len);

    POSEDETGRAY pose_detector;
    int pose_box_len = pose_det_shape[0] * pose_det_shape[1];
    pose_detector.Initialize(path_pose_det, &img_shape, &pose_det_shape, pose_box_len, 3);

    cout << "[INFO] Face+Eye+Pose Detection Models initialized!" << endl;
    cout << "sleep for 0.2 second!" << endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    std::thread inference_thread(inference_thread_func,
                                 &pose_detector,
                                 &eye_detector,
                                 &face_detector,
                                 dual_display_offset_y,
                                 img_width,
                                 img_height);
    cout << "[INFO] Inference thread started!" << endl;

    std::thread listener_thread(keyboard_listener);
    
    uint16_t num_frames = 0;
    ssne_tensor_t img_sensor[2];
    ssne_tensor_t output_sensor[2];
    output_sensor[0] = create_tensor(img_width, img_height * 2, SSNE_Y_8, SSNE_BUF_AI);
    output_sensor[1] = create_tensor(img_width, img_height * 2, SSNE_Y_8, SSNE_BUF_AI);
    
    processor.GetDualImage(&img_sensor[0], &img_sensor[1]);
	copy_double_tensor_buffer(img_sensor[0], img_sensor[1], output_sensor[0]);
	copy_double_tensor_buffer(img_sensor[0], img_sensor[1], output_sensor[1]);
 
    int res = set_isp_debug_config(output_sensor[0], output_sensor[1]);

    while (num_frames < 2) {
		res = start_isp_debug_load();
		num_frames++;
	}

    while (!check_exit_flag()) {
        processor.GetDualImage(&img_sensor[0], &img_sensor[1]);
        get_even_or_odd_flag(load_flag);
        if (load_flag == 0) {
            copy_double_tensor_buffer(img_sensor[0], img_sensor[1], output_sensor[0]);
        } else {
            copy_double_tensor_buffer(img_sensor[0], img_sensor[1], output_sensor[1]);
        }
        res = start_isp_debug_load();
        
        {
            std::unique_lock<std::mutex> lock(mtx_image);
            if (image_queue.size() >= MAX_QUEUE_SIZE) {
                image_queue.pop();
            }
            ImagePair img_pair;
            img_pair.img1 = img_sensor[0];
            img_pair.img2 = img_sensor[1];
            img_pair.frame_id = num_frames;
            image_queue.push(img_pair);
            cv_image_ready.notify_one();
        }
        num_frames += 1;
    }
    
    cout << "[INFO] Main loop finished, stopping inference thread..." << endl;
    if (listener_thread.joinable()) {
        listener_thread.join();
    }
    {
        std::unique_lock<std::mutex> lock(mtx_image);
        stop_inference = true;
        cv_image_ready.notify_one();
    }
    if (inference_thread.joinable()) {
        inference_thread.join();
        cout << "[INFO] Inference thread joined successfully!" << endl;
    }
    
    pose_detector.Release();
    face_detector.Release();
    eye_detector.Release();
    processor.Release();
    visualizer.Release();
    
    if (ssne_release()) {
        fprintf(stderr, "SSNE release failed!\n");
        return -1;
    }
    return 0;
}
