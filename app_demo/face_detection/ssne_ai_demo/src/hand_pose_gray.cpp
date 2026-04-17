/*
 * @Filename: hand_pose_gray.cpp
 * @Description: YOLOv8n-Pose 手部关键点检测 (v3 - 全面重写)
 *
 * 核心改进:
 *   1. 自适应置信度阈值: 自动计算分数分布，只保留 top-N 候选
 *   2. 关键点 raw 值钳位: 防止量化后的异常大值导致关键点乱飘
 *   3. 关键点硬约束: 强制所有关键点在检测框内
 *   4. 首帧诊断输出: 打印 tensor 统计和解码示例，便于调试
 *   5. 严格质量过滤: 最小框尺寸、面积比、宽高比、最少可见关键点数
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

#include "../include/utils.hpp"

namespace {

// ---- 诊断: 在首次检测到手时打印，而非前N帧 ----
constexpr int kDebugDetections = 3;  // 打印前N次有检测结果的帧
int g_debug_det_count = 0;

// ---- 质量过滤参数 ----
constexpr float kMinBoxPx    = 30.0f;   // 框最小边长(像素)
constexpr float kMaxAreaRatio = 0.60f;   // 框面积不超过图像60%
constexpr float kMinAR       = 0.25f;    // 最小宽高比
constexpr float kMaxAR       = 4.0f;     // 最大宽高比
constexpr float kRawKptClamp = 2.0f;     // 原始关键点值钳位范围 [-2, 2]
constexpr int   kMinVisKpts  = 3;        // 至少3个可见关键点

// ---- 自适应阈值 ----
constexpr int kAdaptiveTopN  = 50;       // 自适应: 最多保留50个候选

// ============================================================================
float Sigmoid(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

inline float Clamp(float v, float lo, float hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

// DFL 解码 (double精度, 与 eye_det_gray 一致)
static float DFL(const float* tensor, int start_ch, int spatial, int idx) {
    float mx = -1e9f;
    for (int i = 0; i < 16; ++i) {
        float v = tensor[(start_ch + i) * spatial + idx];
        if (v > mx) mx = v;
    }
    double sum = 0.0, res = 0.0;
    for (int i = 0; i < 16; ++i) {
        float r = tensor[(start_ch + i) * spatial + idx] - mx;
        if (r < -15.0f) r = -15.0f;
        double w = std::exp(static_cast<double>(r));
        sum += w;
        res += w * static_cast<double>(i);
    }
    return (sum < 1e-12) ? 0.0f : static_cast<float>(res / sum);
}

float IoU(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    float ix1 = std::max(a[0], b[0]), iy1 = std::max(a[1], b[1]);
    float ix2 = std::min(a[2], b[2]), iy2 = std::min(a[3], b[3]);
    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float ua = std::max(0.0f, a[2]-a[0]) * std::max(0.0f, a[3]-a[1]);
    float ub = std::max(0.0f, b[2]-b[0]) * std::max(0.0f, b[3]-b[1]);
    float uni = ua + ub - inter;
    return (uni <= 0.0f) ? 0.0f : inter / uni;
}

} // namespace


// ============================================================================
// Initialize
// ============================================================================
void HANDPOSEGRAY::Initialize(std::string& model_path, std::array<int, 2>* in_img_shape,
                              std::array<int, 2>* in_det_shape, int in_box_len) {
    nms_threshold = 0.30f;
    keep_top_k = 2;
    top_k = 20;

    img_shape = *in_img_shape;
    det_shape = *in_det_shape;
    box_len = in_box_len;
    w_scale = static_cast<float>(img_shape[0]) / static_cast<float>(det_shape[0]);
    h_scale = static_cast<float>(img_shape[1]) / static_cast<float>(det_shape[1]);
    steps = {8, 16, 32};

    printf("[HAND] img=%dx%d det=%dx%d scale=%.2f/%.2f nms=%.2f topk=%d/%d\n",
           img_shape[0], img_shape[1], det_shape[0], det_shape[1],
           w_scale, h_scale, nms_threshold, top_k, keep_top_k);

    char* p = const_cast<char*>(model_path.c_str());
    model_id = ssne_loadmodel(p, SSNE_STATIC_ALLOC);

    inputs[0] = create_tensor(
        static_cast<uint32_t>(det_shape[0]),
        static_cast<uint32_t>(det_shape[1]),
        SSNE_Y_8, SSNE_BUF_AI);

    int dtype = SSNE_UINT8;
    if (ssne_get_model_input_dtype(model_id, &dtype) == 0) {
        set_data_type(inputs[0], static_cast<uint8_t>(dtype));
        printf("[HAND] input dtype: %d\n", dtype);
    }

    int nr = SetNormalize(pipe_offline, model_id);
    if (nr != 0) printf("[HAND] SetNormalize ret=%d\n", nr);

    g_debug_det_count = 0;
    printf("[HAND] model loaded (21 kpts, 9 outputs)\n");
}


// ============================================================================
// DecodeBranch - 单个stride分支的解码
// ============================================================================
void HANDPOSEGRAY::DecodeBranch(const float* cls_head, const float* box_head,
                                const float* kpt_head,
                                int feat_h, int feat_w, int stride,
                                float conf_threshold,
                                std::vector<std::array<float, 4>>* boxes,
                                std::vector<float>* scores,
                                std::vector<std::vector<std::array<float, 3>>>* keypoints) const {
    const int spatial = feat_h * feat_w;
    const float dw = static_cast<float>(det_shape[0]);
    const float dh = static_cast<float>(det_shape[1]);
    const float max_area = dw * dh * kMaxAreaRatio;

    for (int gy = 0; gy < feat_h; ++gy) {
        for (int gx = 0; gx < feat_w; ++gx) {
            const int idx = gy * feat_w + gx;
            const float score = Sigmoid(cls_head[idx]);
            if (score < conf_threshold) continue;

            // ---- DFL 边框解码 ----
            float cx = (static_cast<float>(gx) + 0.5f) * stride;
            float cy = (static_cast<float>(gy) + 0.5f) * stride;
            float x1 = Clamp(cx - DFL(box_head, 0, spatial, idx) * stride, 0.0f, dw);
            float y1 = Clamp(cy - DFL(box_head, 16, spatial, idx) * stride, 0.0f, dh);
            float x2 = Clamp(cx + DFL(box_head, 32, spatial, idx) * stride, 0.0f, dw);
            float y2 = Clamp(cy + DFL(box_head, 48, spatial, idx) * stride, 0.0f, dh);
            if (x2 <= x1 || y2 <= y1) continue;

            float bw = x2 - x1, bh = y2 - y1;

            // ---- 框质量过滤 ----
            if (bw < kMinBoxPx || bh < kMinBoxPx) continue;
            if (bw * bh > max_area) continue;
            float ar = bw / std::max(1e-6f, bh);
            if (ar < kMinAR || ar > kMaxAR) continue;

            // ---- 关键点解码 (YOLOv8-Pose) ----
            std::vector<std::array<float, 3>> kpts(kNumKeypoints);
            int vis_count = 0;

            for (int k = 0; k < kNumKeypoints; ++k) {
                float rx = kpt_head[(k * 3 + 0) * spatial + idx];
                float ry = kpt_head[(k * 3 + 1) * spatial + idx];
                float rv = kpt_head[(k * 3 + 2) * spatial + idx];

                // 钳位原始值, 防止量化异常导致关键点飞出
                rx = Clamp(rx, -kRawKptClamp, kRawKptClamp);
                ry = Clamp(ry, -kRawKptClamp, kRawKptClamp);

                float kx = (rx * 2.0f + static_cast<float>(gx)) * stride;
                float ky = (ry * 2.0f + static_cast<float>(gy)) * stride;
                float kv = Sigmoid(rv);

                // 硬约束: 关键点必须在检测框内
                kx = Clamp(kx, x1, x2);
                ky = Clamp(ky, y1, y2);

                kpts[k] = {kx, ky, kv};
                if (kv > 0.3f) vis_count++;
            }

            if (vis_count < kMinVisKpts) continue;

            boxes->push_back({x1, y1, x2, y2});
            scores->push_back(score);
            keypoints->push_back(std::move(kpts));
        }
    }
}


// ============================================================================
// Postprocess - NMS + 坐标缩放
// ============================================================================
void HANDPOSEGRAY::Postprocess(std::vector<std::array<float, 4>>* boxes,
                               std::vector<float>* scores,
                               std::vector<std::vector<std::array<float, 3>>>* keypoints,
                               HandPoseResult* result, float* conf_threshold) {
    result->Clear();
    if (boxes->empty()) return;

    // 按分数降序
    std::vector<int> order(boxes->size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return scores->at(a) > scores->at(b);
    });
    if (static_cast<int>(order.size()) > top_k) order.resize(top_k);

    // NMS
    std::vector<bool> suppressed(boxes->size(), false);
    std::vector<int> kept;
    for (int i = 0; i < static_cast<int>(order.size()); ++i) {
        int ii = order[i];
        if (suppressed[ii]) continue;
        if (scores->at(ii) < *conf_threshold) continue;
        kept.push_back(ii);
        if (static_cast<int>(kept.size()) >= keep_top_k) break;
        for (int j = i + 1; j < static_cast<int>(order.size()); ++j) {
            int jj = order[j];
            if (!suppressed[jj] && IoU(boxes->at(ii), boxes->at(jj)) > nms_threshold)
                suppressed[jj] = true;
        }
    }

    // 缩放到原图坐标
    const float iw = static_cast<float>(img_shape[0]);
    const float ih = static_cast<float>(img_shape[1]);
    result->Reserve(static_cast<int>(kept.size()));

    for (int idx : kept) {
        auto& b = boxes->at(idx);
        result->boxes.push_back({
            Clamp(b[0] * w_scale, 0, iw), Clamp(b[1] * h_scale, 0, ih),
            Clamp(b[2] * w_scale, 0, iw), Clamp(b[3] * h_scale, 0, ih)
        });
        result->scores.push_back(scores->at(idx));

        std::vector<std::array<float, 3>> sk(kNumKeypoints);
        const auto& kpts = keypoints->at(idx);
        for (int k = 0; k < kNumKeypoints; ++k) {
            sk[k] = {
                Clamp(kpts[k][0] * w_scale, 0, iw),
                Clamp(kpts[k][1] * h_scale, 0, ih),
                kpts[k][2]
            };
        }
        result->keypoints.push_back(std::move(sk));
    }
}


// ============================================================================
// Predict - 主推理入口 (含自适应阈值 + 诊断输出)
// ============================================================================
void HANDPOSEGRAY::Predict(ssne_tensor_t* img_in, HandPoseResult* result,
                           float conf_threshold) {
    result->Clear();

    if (RunAiPreprocessPipe(pipe_offline, *img_in, inputs[0]) != 0) return;
    if (ssne_inference(model_id, 1, inputs) != 0) return;
    ssne_getoutput(model_id, 9, outputs);

    // 特征图尺寸
    const int fh8  = det_shape[1] / 8,  fw8  = det_shape[0] / 8;
    const int fh16 = det_shape[1] / 16, fw16 = det_shape[0] / 16;
    const int fh32 = det_shape[1] / 32, fw32 = det_shape[0] / 32;

    // 预期元素数 (9个tensor)
    const uint32_t expect[9] = {
        static_cast<uint32_t>(fh8  * fw8),                          // cls_s8
        static_cast<uint32_t>(fh16 * fw16),                         // cls_s16
        static_cast<uint32_t>(fh32 * fw32),                         // cls_s32
        static_cast<uint32_t>(fh8  * fw8  * 64),                    // box_s8
        static_cast<uint32_t>(fh16 * fw16 * 64),                    // box_s16
        static_cast<uint32_t>(fh32 * fw32 * 64),                    // box_s32
        static_cast<uint32_t>(fh8  * fw8  * kKptChannels),          // kpt_s8
        static_cast<uint32_t>(fh16 * fw16 * kKptChannels),          // kpt_s16
        static_cast<uint32_t>(fh32 * fw32 * kKptChannels),          // kpt_s32
    };

    float* ptrs[9] = {};

    for (int i = 0; i < 9; ++i) {
        uint32_t sz = get_total_size(outputs[i]);
        float* data = reinterpret_cast<float*>(get_data(outputs[i]));
        for (int e = 0; e < 9; ++e) {
            if (sz == expect[e] && ptrs[e] == nullptr) { ptrs[e] = data; break; }
        }
    }

    // 检查所有 tensor 是否匹配
    bool all_ok = true;
    for (int i = 0; i < 9; ++i) {
        if (!ptrs[i]) { all_ok = false; break; }
    }
    if (!all_ok) {
        printf("[HAND] ERROR: tensor mismatch! Expected:");
        for (int i = 0; i < 9; ++i) printf(" %u", expect[i]);
        printf("\n  Actual:");
        for (int i = 0; i < 9; ++i) printf(" %u", get_total_size(outputs[i]));
        printf("\n");
        return;
    }

    // ptrs[0..2] = cls_s8/s16/s32, ptrs[3..5] = box, ptrs[6..8] = kpt
    float* cls_ptrs[3] = {ptrs[0], ptrs[1], ptrs[2]};
    int    sp[3]       = {fh8 * fw8, fh16 * fw16, fh32 * fw32};

    // ---- 自适应阈值: 统计所有 cls sigmoid 分数，只保留 top-N ----
    std::vector<float> all_sigs;
    all_sigs.reserve(300);
    for (int s = 0; s < 3; ++s) {
        for (int j = 0; j < sp[s]; ++j) {
            float sig = Sigmoid(cls_ptrs[s][j]);
            if (sig > conf_threshold * 0.5f) all_sigs.push_back(sig);
        }
    }
    std::sort(all_sigs.rbegin(), all_sigs.rend());

    float eff_conf = conf_threshold;
    if (static_cast<int>(all_sigs.size()) > kAdaptiveTopN) {
        eff_conf = std::max(conf_threshold, all_sigs[kAdaptiveTopN]);
    }

    // ---- 解码三个stride分支 ----
    std::vector<std::array<float, 4>> bboxes;
    std::vector<float> scores;
    std::vector<std::vector<std::array<float, 3>>> all_kpts;

    DecodeBranch(ptrs[0], ptrs[3], ptrs[6], fh8,  fw8,  8,
                 eff_conf, &bboxes, &scores, &all_kpts);
    DecodeBranch(ptrs[1], ptrs[4], ptrs[7], fh16, fw16, 16,
                 eff_conf, &bboxes, &scores, &all_kpts);
    DecodeBranch(ptrs[2], ptrs[5], ptrs[8], fh32, fw32, 32,
                 eff_conf, &bboxes, &scores, &all_kpts);

    // ---- NMS后处理 ----
    Postprocess(&bboxes, &scores, &all_kpts, result, &eff_conf);

    // ---- 诊断: 只在检测到手时打印前N次 ----
    if (!result->boxes.empty() && g_debug_det_count < kDebugDetections) {
        g_debug_det_count++;
        printf("\n[HAND] ====== 检测诊断 %d/%d ======\n", g_debug_det_count, kDebugDetections);

        // tensor值范围
        const char* names[9] = {"cls8","cls16","cls32","box8","box16","box32","kpt8","kpt16","kpt32"};
        for (int i = 0; i < 9; ++i) {
            uint32_t sz = get_total_size(outputs[i]);
            float* d = ptrs[i];
            float vmin = d[0], vmax = d[0];
            for (uint32_t j = 1; j < std::min(sz, 2000u); ++j) {
                if (d[j] < vmin) vmin = d[j];
                if (d[j] > vmax) vmax = d[j];
            }
            printf("[HAND] %s size=%u range=[%.3f, %.3f]\n", names[i], sz, vmin, vmax);
        }

        // cls分数分布
        printf("[HAND] 候选=%zu 自适应阈值=%.3f top-5:", all_sigs.size(), eff_conf);
        for (int j = 0; j < std::min(5, static_cast<int>(all_sigs.size())); ++j)
            printf(" %.3f", all_sigs[j]);
        printf("\n");

        printf("[HAND] 解码候选=%zu, NMS后=%zu只手\n", bboxes.size(), result->boxes.size());
        for (size_t i = 0; i < result->boxes.size(); ++i) {
            printf("[HAND] hand[%zu] box=(%.0f,%.0f,%.0f,%.0f) score=%.3f\n",
                   i, result->boxes[i][0], result->boxes[i][1],
                   result->boxes[i][2], result->boxes[i][3], result->scores[i]);

            // 打印该手的kpt原始值 (从NMS前的数据中查找对应的原始位置)
            printf("[HAND] kpts(缩放后):");
            for (int k = 0; k < kNumKeypoints; ++k) {
                auto& kp = result->keypoints[i][k];
                printf(" [%d](%.0f,%.0f,v=%.2f)", k, kp[0], kp[1], kp[2]);
            }
            printf("\n");
        }
    }
}


void HANDPOSEGRAY::Release() {
    release_tensor(inputs[0]);
    for (int i = 0; i < 9; ++i) release_tensor(outputs[i]);
    ReleaseAIPreprocessPipe(pipe_offline);
}
