/*
 * @Filename: demo_face.cpp
 * @Author: Hongying He
 * @Email: hongying.he@smartsenstech.com
 * @Date: 2025-12-30 14-57-47
 * @Copyright (c) 2025 SmartSens
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
constexpr float kEyeInferConfThreshold = 0.05f;    // 推理阈值（进一步放宽，优先提升召回）
constexpr float kEyeDisplayScoreThreshold = 0.10f; // 首次显示阈值
constexpr float kEyeDisplayScoreThresholdTracked = 0.05f; // 已跟踪后保持阈值
constexpr float kEyeMinBoxSize = 2.0f;             // 过滤极小框
constexpr int kEyeHoldFrames = 1;                  // 缩短保持，降低拖影和延迟
constexpr int kClearAfterMissFrames = 2;           // 更快清屏，避免残影滞留
constexpr float kEyeDotFixedRadius = 12.0f;        // 固定眼点半径（像素）
constexpr float kFaceInferConfThreshold = 0.45f;   // 人脸检测阈值
constexpr int kFaceInferInterval = 3;              // 分时推理：降低人脸检测频率提升帧率
constexpr int kFaceRoiHoldFrames = 10;             // 人脸短时丢检保持
constexpr float kKalmanQPos = 16.0f;               // 过程噪声（位置），增大以提升跟随性
constexpr float kKalmanQVel = 4.0f;                // 过程噪声（速度）
constexpr float kKalmanR = 4.0f;                   // 观测噪声，减小以降低迟滞

struct AxisKalman {
    bool initialized = false;
    float x = 0.0f;
    float v = 0.0f;
    float p00 = 1.0f;
    float p01 = 0.0f;
    float p10 = 0.0f;
    float p11 = 1.0f;

    void Init(float z) {
        initialized = true;
        x = z;
        v = 0.0f;
        p00 = 10.0f;
        p01 = 0.0f;
        p10 = 0.0f;
        p11 = 10.0f;
    }

    void Predict(float q_pos, float q_vel) {
        if (!initialized) {
            return;
        }
        x += v;
        const float n00 = p00 + p01 + p10 + p11 + q_pos;
        const float n01 = p01 + p11;
        const float n10 = p10 + p11;
        const float n11 = p11 + q_vel;
        p00 = n00;
        p01 = n01;
        p10 = n10;
        p11 = n11;
    }

    void Update(float z, float r) {
        if (!initialized) {
            Init(z);
            return;
        }
        const float y = z - x;
        const float s = p00 + r;
        const float k0 = p00 / s;
        const float k1 = p10 / s;
        x += k0 * y;
        v += k1 * y;

        const float o00 = p00;
        const float o01 = p01;
        const float o10 = p10;
        const float o11 = p11;
        p00 = (1.0f - k0) * o00;
        p01 = (1.0f - k0) * o01;
        p10 = o10 - k1 * o00;
        p11 = o11 - k1 * o01;
    }
};

struct EyeTrack {
    AxisKalman cx;
    AxisKalman cy;
    AxisKalman w;
    AxisKalman h;
    int miss = 0;

    bool IsReady() const {
        return cx.initialized && cy.initialized && w.initialized && h.initialized;
    }

    void Predict() {
        cx.Predict(kKalmanQPos, kKalmanQVel);
        cy.Predict(kKalmanQPos, kKalmanQVel);
        w.Predict(kKalmanQPos, kKalmanQVel);
        h.Predict(kKalmanQPos, kKalmanQVel);
    }

    void UpdateWithBox(const std::array<float, 4>& box) {
        const float bw = std::max(1.0f, box[2] - box[0]);
        const float bh = std::max(1.0f, box[3] - box[1]);
        const float bx = 0.5f * (box[0] + box[2]);
        const float by = 0.5f * (box[1] + box[3]);

        Predict();
        cx.Update(bx, kKalmanR);
        cy.Update(by, kKalmanR);
        w.Update(bw, kKalmanR);
        h.Update(bh, kKalmanR);
        miss = 0;
    }

    std::array<float, 4> ToBox() const {
        const float bw = std::max(1.0f, w.x);
        const float bh = std::max(1.0f, h.x);
        const float x1 = cx.x - 0.5f * bw;
        const float y1 = cy.x - 0.5f * bh;
        const float x2 = cx.x + 0.5f * bw;
        const float y2 = cy.x + 0.5f * bh;
        return {x1, y1, x2, y2};
    }
};

EyeTrack g_eye_tracks[2];
std::vector<std::array<float, 4>> g_last_drawn_boxes;
int g_last_draw_frame = -10000;
bool g_has_active_overlay = false;
int g_consecutive_empty_frames = 0;

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

std::array<float, 4> ClampBoxToImage(const std::array<float, 4>& box,
                                     float img_w,
                                     float img_h) {
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

std::vector<std::array<float, 4>> SortByCenterX(
    std::vector<std::array<float, 4>> boxes) {
    std::sort(boxes.begin(), boxes.end(), [](const std::array<float, 4>& a,
                                             const std::array<float, 4>& b) {
        return CenterX(a) < CenterX(b);
    });
    return boxes;
}

std::vector<std::array<float, 4>> GetStableEyeBoxes(
    const FaceDetectionResult& result,
    float img_w,
    float img_h) {
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
    if (candidates.size() > 2) {
        candidates.resize(2);
    }

    std::vector<std::array<float, 4>> current;
    current.reserve(candidates.size());
    for (const auto& item : candidates) {
        current.push_back(item.first);
    }
    current = SortByCenterX(current);

    for (int i = 0; i < 2; ++i) {
        if (i < static_cast<int>(current.size())) {
            g_eye_tracks[i].UpdateWithBox(current[i]);
            continue;
        }
        if (g_eye_tracks[i].IsReady()) {
            g_eye_tracks[i].Predict();
            g_eye_tracks[i].miss += 1;
        }
    }

    std::vector<std::array<float, 4>> stable;
    stable.reserve(2);
    for (int i = 0; i < 2; ++i) {
        if (g_eye_tracks[i].IsReady() && g_eye_tracks[i].miss <= kEyeHoldFrames) {
            stable.push_back(ClampBoxToImage(g_eye_tracks[i].ToBox(), img_w, img_h));
        }
    }
    return SortByCenterX(stable);
}

bool ShouldRedraw(const std::vector<std::array<float, 4>>& boxes, int frame_id) {
    if (g_last_draw_frame < 0) {
        return true;
    }
    if (boxes.size() != g_last_drawn_boxes.size()) {
        return true;
    }
    for (size_t i = 0; i < boxes.size(); ++i) {
        const float dx = std::fabs(CenterX(boxes[i]) - CenterX(g_last_drawn_boxes[i]));
        const float dy = std::fabs((boxes[i][1] + boxes[i][3]) * 0.5f - (g_last_drawn_boxes[i][1] + g_last_drawn_boxes[i][3]) * 0.5f);
        const float dw = std::fabs((boxes[i][2] - boxes[i][0]) - (g_last_drawn_boxes[i][2] - g_last_drawn_boxes[i][0]));
        const float dh = std::fabs((boxes[i][3] - boxes[i][1]) - (g_last_drawn_boxes[i][3] - g_last_drawn_boxes[i][1]));
        if (dx > 2.0f || dy > 2.0f || dw > 2.0f || dh > 2.0f) {
            return true;
        }
    }
    return false;
}

std::array<float, 4> BuildEyeDotBox(const std::array<float, 4>& eye_box,
                                    float img_w,
                                    float img_h) {
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

std::queue<ImagePair> image_queue;               // 图像队列
const int MAX_QUEUE_SIZE = 1;                    // 队列长度设为1，优先最新帧降低延迟

// 全局退出标志（线程安全）
bool g_exit_flag = false;
// 保护退出标志的互斥锁
std::mutex g_mtx;

/**
 * @brief 键盘监听程序，用于结束demo
 */
void keyboard_listener() {
    std::string input;
    std::cout << "键盘监听线程已启动，输入 'q' 退出程序..." << std::endl;

    while (true) {
        // 读取键盘输入（会阻塞直到有输入）
        std::cin >> input;

        // 加锁修改退出标志
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

/**
 * @brief 检查退出标志的辅助函数（线程安全）
 * @return 是否需要退出
 */
bool check_exit_flag() {
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_exit_flag;
}


/**
 * @brief 推理线程函数：从队列中获取图像并执行眼睛检测
 */
void inference_thread_func(EYEDETGRAY* eye_detector,
                           SCRFDGRAY* face_detector,
                           int dual_display_offset_y,
                           int img_width,
                           int img_height) {
    cout << "[Thread] Inference thread started!" << endl;
    
    // 眼睛检测结果初始化
    FaceDetectionResult* det_result1 = new FaceDetectionResult;
    FaceDetectionResult* face_result = new FaceDetectionResult;
    std::array<float, 4> last_face_roi = {0.0f, 0.0f, 0.0f, 0.0f};
    bool has_face_roi = false;
    int face_miss_frames = 0;
    
    while (!stop_inference) {
        ImagePair img_pair;
        bool has_image = false;
        
        // 从队列中获取图像
        {
            std::unique_lock<std::mutex> lock(mtx_image);
            
            // 等待图像就绪或停止信号
            cv_image_ready.wait(lock, [] {
                return !image_queue.empty() || stop_inference;
            });
            
            if (stop_inference && image_queue.empty()) {
                break;
            }
            
            if (!image_queue.empty()) {
                img_pair = image_queue.front();
                image_queue.pop();
                has_image = true;
            }
        }
        
        if (!has_image) {
            continue;
        }
        
        // 分时推理：先脸后眼。人脸低频刷新，眼睛每帧运行。
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

        // 执行眼睛检测（非阻塞主循环）
        // 眼睛每帧检测，脸框低频刷新并由ROI保持，兼顾稳定与速度
        eye_detector->Predict(&img_pair.img1, det_result1, kEyeInferConfThreshold);
        std::vector<std::array<float, 4>> stable_boxes =
            GetStableEyeBoxes(*det_result1,
                              static_cast<float>(img_width),
                              static_cast<float>(img_height));


        // 绘制集合：始终优先显示脸框，再叠加眼点（固定大小风格）
        std::vector<std::array<float, 4>> draw_boxes;
        if (has_face_roi) {
            draw_boxes.push_back(last_face_roi);
        }
        std::vector<std::array<float, 4>> eye_dot_boxes;
        eye_dot_boxes.reserve(stable_boxes.size());
        for (const auto& b : stable_boxes) {
            eye_dot_boxes.push_back(BuildEyeDotBox(b,
                                                   static_cast<float>(img_width),
                                                   static_cast<float>(img_height)));
        }
        draw_boxes.insert(draw_boxes.end(), eye_dot_boxes.begin(), eye_dot_boxes.end());

        // 处理检测结果 - 右侧图像
        if (draw_boxes.empty()) {
            g_consecutive_empty_frames += 1;
            if (g_visualizer != nullptr && g_has_active_overlay &&
                g_consecutive_empty_frames >= kClearAfterMissFrames) {
                std::vector<std::array<float, 4>> empty_boxes;
                g_visualizer->Draw(empty_boxes);
                g_has_active_overlay = false;
                g_last_drawn_boxes.clear();
                g_last_draw_frame = img_pair.frame_id;
                g_consecutive_empty_frames = 0;
            }
            continue;
        }

        g_consecutive_empty_frames = 0;
        if (g_visualizer != nullptr) {
            if (!eye_dot_boxes.empty()) {
                for (size_t i = 0; i < eye_dot_boxes.size(); i++) {
                    printf("[Frame %d] Stable Eye: (%.2f, %.2f, %.2f, %.2f)\n",
                           img_pair.frame_id,
                           eye_dot_boxes[i][0],
                           eye_dot_boxes[i][1],
                           eye_dot_boxes[i][2],
                           eye_dot_boxes[i][3]);
                }
            }
            // 当前策略：脸框显示矩形，眼睛显示固定大圆点（强调跟随而非框尺寸）
            std::vector<std::array<float, 4>> face_draw;
            std::vector<std::array<float, 4>> eye_draw;
            if (has_face_roi) {
                face_draw.push_back(last_face_roi);
            }
            eye_draw = eye_dot_boxes;

            g_visualizer->Draw(face_draw);
            if (g_eye_draw_mode == 0) {
                g_visualizer->DrawCircles(eye_draw);
            } else {
                g_visualizer->Draw(eye_draw);
            }
            g_last_drawn_boxes = draw_boxes;
            g_last_draw_frame = img_pair.frame_id;
            g_has_active_overlay = true;
        }
        
        // // 处理检测结果 - 左侧图像
        // if (det_result2->boxes.size() > 0) {
        //     for (size_t i = 0; i < det_result2->boxes.size(); i++) {
        //         float x3_orig = det_result2->boxes[i][0];
        //         float y3_orig = det_result2->boxes[i][1] + dual_display_offset_y;
        //         float x4_orig = det_result2->boxes[i][2];
        //         float y4_orig = det_result2->boxes[i][3] + dual_display_offset_y;
                
        //         printf("[Frame %d] Left Face detected: (%.2f, %.2f, %.2f, %.2f)\n", 
        //                img_pair.frame_id, x3_orig, y3_orig, x4_orig, y4_orig);
        //     }
        // } else {
        //     cout << "[Frame " << img_pair.frame_id << "] Left No face detected" << endl;
        // }
    }
    
    // 释放资源
    delete det_result1;
    delete face_result;
    
    cout << "[Thread] Inference thread stopped!" << endl;
}

/**
 * @brief 眼睛检测演示程序主函数
 * @return 执行结果，0表示成功
 */
int main(int argc, char* argv[]) {
    /******************************************************************************************
     * 1. 参数配置
     ******************************************************************************************/
    
    uint8_t load_flag = 0;  // 0: 当前load偶帧; 1: 当前load奇帧（初始值为0）
    // 图像尺寸配置（根据镜头参数修改）
    int img_width = 640;    // 输入图像宽度
    int img_height = 480;  // 输入图像高度
    
    // 模型配置参数
    array<int, 2> eye_det_shape = {640, 480};   // 眼睛检测模型输入尺寸
    string path_eye_det = "/app_demo/app_assets/models/eye.m1model";

    // 锁死 circle 模式，忽略环境变量和命令行参数。
    g_eye_draw_mode = 0;
    printf("[INFO] Eye draw mode: circle (forced)\n");

    // 保留后续脸部检测接口，方便分时推理时直接启用
    array<int, 2> face_det_shape = {640, 480};
    string path_face_det = "/app_demo/app_assets/models/face_640x480.m1model";
    
    /******************************************************************************************
     * 2. 系统初始化
     ******************************************************************************************/
    
    // SSNE初始化
    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
    }
    
    // 图像处理器初始化
    array<int, 2> img_shape = {img_width, img_height};  // 原始图像尺寸
    const int dual_display_offset_y = 480;  // 双目显示时第二路图像Y方向的偏移量（上下拼接显示）
    // 原图: 640×480, 模型输入图：640×480
    // 输入比例一致，无需额外裁剪偏移量
    
    // OSD可视化器初始化（用于绘制检测框）
    VISUALIZER visualizer;
    visualizer.Initialize(img_shape);  // 初始化可视化器（配置图像尺寸）
    
    // 设置全局可视化器指针供推理线程使用
    g_visualizer = &visualizer;

    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape);  // 初始化图像处理器（配置原图尺寸）
    
    // 启用脸部+眼睛：分时推理（先脸后眼）
    SCRFDGRAY face_detector;
    int face_box_len = face_det_shape[0] * face_det_shape[1];
    face_detector.Initialize(path_face_det, &img_shape, &face_det_shape, false, face_box_len);

    EYEDETGRAY eye_detector;
    int eye_box_len = eye_det_shape[0] * eye_det_shape[1];
    eye_detector.Initialize(path_eye_det, &img_shape, &eye_det_shape, eye_box_len);

    cout << "[INFO] Face+Eye Detection Models initialized!" << endl;
    // 系统稳定等待
    cout << "sleep for 0.2 second!" << endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));  // 等待系统稳定
    
    // 启动推理线程
    std::thread inference_thread(inference_thread_func,
                                 &eye_detector,
                                 &face_detector,
                                 dual_display_offset_y,
                                 img_width,
                                 img_height);
    cout << "[INFO] Inference thread started!" << endl;

    // 创建键盘监听线程
    std::thread listener_thread(keyboard_listener);
    
    uint16_t num_frames = 0;  // 帧计数器
    ssne_tensor_t img_sensor[2];  // 图像tensor定义
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
    /******************************************************************************************
     * 3. 主处理循环 - 只负责图像拷贝到ISP debug，不阻塞推理
     ******************************************************************************************/
    //循环5000帧后推出，循环次数可以修改，也可以改成while(true)
    while (!check_exit_flag()) {
        
        // 从sensor获取图像（裁剪图）
        processor.GetDualImage(&img_sensor[0], &img_sensor[1]);

        get_even_or_odd_flag(load_flag);

        if (load_flag == 0)
        {
            copy_double_tensor_buffer(img_sensor[0], img_sensor[1], output_sensor[0]);
        }
        else
        {
            copy_double_tensor_buffer(img_sensor[0], img_sensor[1], output_sensor[1]);
        }

        // 启动ISP debug load（主循环核心任务，不能被阻塞）
        res = start_isp_debug_load();
        
        // 将图像放入队列供推理线程使用（非阻塞）
        {
            std::unique_lock<std::mutex> lock(mtx_image);
            
            // 始终保留最新帧，满队列时丢弃旧帧，降低显示延迟
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

        num_frames += 1;  // 帧计数器递增
    }
    
    /******************************************************************************************
     * 4. 停止推理线程并等待其完成
     ******************************************************************************************/
    
    cout << "[INFO] Main loop finished, stopping inference thread..." << endl;

    // 等待监听线程退出，释放资源
    if (listener_thread.joinable()) {
        listener_thread.join();
    }
    
    // 设置停止标志并通知推理线程
    {
        std::unique_lock<std::mutex> lock(mtx_image);
        stop_inference = true;
        cv_image_ready.notify_one();
    }
    
    // 等待推理线程退出
    if (inference_thread.joinable()) {
        inference_thread.join();
        cout << "[INFO] Inference thread joined successfully!" << endl;
    }
    
    /******************************************************************************************
     * 5. 资源释放
     ******************************************************************************************/
    
    face_detector.Release();
    eye_detector.Release();  // 释放检测器资源
    processor.Release();  // 释放图像处理器资源
    visualizer.Release();  // 释放可视化器资源
    
    if (ssne_release()) {
        fprintf(stderr, "SSNE release failed!\n");
        return -1;
    }
    
    return 0;
}
 
