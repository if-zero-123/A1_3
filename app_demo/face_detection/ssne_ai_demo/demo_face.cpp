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
constexpr float kEyeDisplayScoreThresholdTracked = 0.08f; // 已跟踪后保持阈值
constexpr float kEyeMinBoxSize = 3.0f;             // 过滤极小框
constexpr int   kEyeHoldFrames = 5;                // 丢检后Kalman预测保持
constexpr int   kClearAfterMissFrames = 8;          // 连续空帧后清屏
constexpr float kEyeDotFixedRadius = 8.0f;          // 固定眼点半径
constexpr float kFaceInferConfThreshold = 0.45f;    // 人脸检测阈值
constexpr int   kFaceInferInterval = 5;             // 人脸每5帧检测（低延迟）
constexpr int   kFaceRoiHoldFrames = 10;            // 人脸ROI保持

// Kalman 滤波参数（残差自适应：静止平滑、运动瞬时跟手）
constexpr float kKalmanQPos = 1.5f;      // 位置过程噪声（大=允许位置快速变化）
constexpr float kKalmanQVel = 0.8f;      // 速度过程噪声（大=允许加速）
constexpr float kKalmanRCalm = 40.0f;    // 静止时R（适度平滑，不过度延迟）
constexpr float kKalmanRActive = 5.0f;   // 运动时R（低=直接跟手）
constexpr float kResidualThresh = 6.0f;  // 残差>此值 → 检测到运动，瞬间切低R

// IoU匹配参数
constexpr float kIoUMatchThreshold = 0.05f;

// 输出量化参数
constexpr float kOutputSnapGrid = 1.0f;    // 坐标取整到像素

// OSD 重绘参数（移除限频，仅保留位移门控）
constexpr float kRedrawMinDelta = 1.5f;     // 移动<1.5px不重绘

// 中值预滤波已移除：它增加 1-2 帧延迟，用残差自适应 Kalman 替代

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
    }

    void Predict() {
        if (!IsReady()) return;
        kcx.Predict(kKalmanQPos, kKalmanQVel);
        kcy.Predict(kKalmanQPos, kKalmanQVel);
        kw.Predict(kKalmanQPos * 0.2f, kKalmanQVel * 0.1f);
        kh.Predict(kKalmanQPos * 0.2f, kKalmanQVel * 0.1f);
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
        miss = 0;
        age += 1;
    }

    std::array<float, 4> ToBox() const {
        float bw = std::max(1.0f, kw.x);
        float bh = std::max(1.0f, kh.x);
        float cx = std::round(kcx.x);
        float cy = std::round(kcy.x);
        bw = std::round(bw);
        bh = std::round(bh);
        return {cx - 0.5f * bw, cy - 0.5f * bh,
                cx + 0.5f * bw, cy + 0.5f * bh};
    }
};

// ============================================================================
// 全局状态
// ============================================================================
EyeTrack g_eye_tracks[2];
std::vector<std::array<float, 4>> g_last_drawn_boxes;
int g_last_draw_frame = -10000;
bool g_has_active_overlay = false;
int g_consecutive_empty_frames = 0;

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
    float dx = (a[0] + a[2]) - (b[0] + b[2]);
    float dy = (a[1] + a[3]) - (b[1] + b[3]);
    return dx * dx + dy * dy;
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
    float img_w,
    float img_h) {
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
        det_boxes.push_back(item.first);
    }

    // --- IoU + 中心距离匹配 ---
    std::vector<bool> det_used(det_boxes.size(), false);

    for (int t = 0; t < 2; ++t) {
        if (!g_eye_tracks[t].IsReady()) continue;

        std::array<float, 4> predicted = g_eye_tracks[t].ToBox();
        int best_d = -1;
        float best_score = -1.0f;

        for (int d = 0; d < static_cast<int>(det_boxes.size()); ++d) {
            if (det_used[d]) continue;
            float iou = ComputeIoU(predicted, det_boxes[d]);
            if (iou < kIoUMatchThreshold) {
                float dist2 = CenterDist2(predicted, det_boxes[d]);
                float diag2 = BoxArea(predicted) * 4.0f;
                if (diag2 > 1.0f && dist2 < diag2 * 4.0f) {
                    iou = 0.01f;
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
        if (g_eye_tracks[i].IsReady() && g_eye_tracks[i].miss <= kEyeHoldFrames) {
            stable.push_back(ClampBoxToImage(g_eye_tracks[i].ToBox(), img_w, img_h));
        }
    }
    return SortByCenterX(stable);
}

// OSD 重绘判断（无限频，仅位移门控防闪烁）
bool ShouldRedraw(const std::vector<std::array<float, 4>>& boxes, int frame_id) {
    if (g_last_draw_frame < 0) return true;
    if (boxes.size() != g_last_drawn_boxes.size()) return true;
    for (size_t i = 0; i < boxes.size(); ++i) {
        const float dx = std::fabs(CenterX(boxes[i]) - CenterX(g_last_drawn_boxes[i]));
        const float dy = std::fabs((boxes[i][1] + boxes[i][3]) * 0.5f - (g_last_drawn_boxes[i][1] + g_last_drawn_boxes[i][3]) * 0.5f);
        if (dx > kRedrawMinDelta || dy > kRedrawMinDelta) return true;
    }
    return false;
}

std::array<float, 4> BuildEyeDotBox(const std::array<float, 4>& eye_box, float img_w, float img_h) {
    const float cx = std::round(0.5f * (eye_box[0] + eye_box[2]));  // snap to pixel
    const float cy = std::round(0.5f * (eye_box[1] + eye_box[3]));
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
void inference_thread_func(EYEDETGRAY* eye_detector,
                           SCRFDGRAY* face_detector,
                           int dual_display_offset_y,
                           int img_width,
                           int img_height) {
    cout << "[Thread] Inference thread started!" << endl;
    
    FaceDetectionResult* det_result1 = new FaceDetectionResult;
    FaceDetectionResult* face_result = new FaceDetectionResult;
    std::array<float, 4> last_face_roi = {0.0f, 0.0f, 0.0f, 0.0f};
    bool has_face_roi = false;
    int face_miss_frames = 0;
    
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
        
        // 人脸低频检测
        if (face_detector != nullptr &&
            ((img_pair.frame_id % kFaceInferInterval) == 0 || !has_face_roi)) {
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

        // 眼睛检测（每帧）
        eye_detector->Predict(&img_pair.img1, det_result1, kEyeInferConfThreshold);
        std::vector<std::array<float, 4>> stable_boxes =
            GetStableEyeBoxes(*det_result1,
                              static_cast<float>(img_width),
                              static_cast<float>(img_height));

        // 构造绘制列表
        std::vector<std::array<float, 4>> eye_dot_boxes;
        eye_dot_boxes.reserve(stable_boxes.size());
        for (const auto& b : stable_boxes) {
            eye_dot_boxes.push_back(BuildEyeDotBox(b,
                                                   static_cast<float>(img_width),
                                                   static_cast<float>(img_height)));
        }

        // 无检测时处理
        if (eye_dot_boxes.empty() && !has_face_roi) {
            g_consecutive_empty_frames += 1;
            if (g_visualizer != nullptr && g_has_active_overlay &&
                g_consecutive_empty_frames >= kClearAfterMissFrames) {
                std::vector<std::array<float, 4>> empty;
                g_visualizer->Draw(empty);
                g_visualizer->DrawCircles(empty);
                g_has_active_overlay = false;
                g_last_drawn_boxes.clear();
                g_last_draw_frame = img_pair.frame_id;
                g_consecutive_empty_frames = 0;
            }
            continue;
        }

        g_consecutive_empty_frames = 0;

        // 第4层：OSD 限频重绘
        if (g_visualizer != nullptr && ShouldRedraw(eye_dot_boxes, img_pair.frame_id)) {
            std::vector<std::array<float, 4>> face_draw;
            if (has_face_roi) {
                face_draw.push_back(last_face_roi);
            }

            g_visualizer->Draw(face_draw);
            if (g_eye_draw_mode == 0) {
                g_visualizer->DrawCircles(eye_dot_boxes);
            } else {
                g_visualizer->Draw(eye_dot_boxes);
            }
            g_last_drawn_boxes = eye_dot_boxes;
            g_last_draw_frame = img_pair.frame_id;
            g_has_active_overlay = true;
        }
    }
    
    delete det_result1;
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
    
    array<int, 2> eye_det_shape = {640, 480};
    string path_eye_det = "/app_demo/app_assets/models/eye.m1model";

    g_eye_draw_mode = 0;
    printf("[INFO] Eye draw mode: circle (forced)\n");

    array<int, 2> face_det_shape = {640, 480};
    string path_face_det = "/app_demo/app_assets/models/face_640x480.m1model";
    
    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
    }
    
    array<int, 2> img_shape = {img_width, img_height};
    const int dual_display_offset_y = 480;
    
    VISUALIZER visualizer;
    visualizer.Initialize(img_shape);
    g_visualizer = &visualizer;

    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape);
    
    SCRFDGRAY face_detector;
    int face_box_len = face_det_shape[0] * face_det_shape[1];
    face_detector.Initialize(path_face_det, &img_shape, &face_det_shape, false, face_box_len);

    EYEDETGRAY eye_detector;
    int eye_box_len = eye_det_shape[0] * eye_det_shape[1];
    eye_detector.Initialize(path_eye_det, &img_shape, &eye_det_shape, eye_box_len);

    cout << "[INFO] Face+Eye Detection Models initialized!" << endl;
    cout << "sleep for 0.2 second!" << endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    std::thread inference_thread(inference_thread_func,
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
