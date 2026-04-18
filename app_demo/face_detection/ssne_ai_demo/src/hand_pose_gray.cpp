/*
 * @Filename: hand_pose_gray.cpp
 * @Description: YOLOv8n-Pose 手部关键点检测 (v3 - 全面重写)
 *
 * 核心改进:
 *   1. 自适应置信度阈值: 自动计算分数分布，只保留 top-N 候选
 *   2. 放宽关键点 raw 值范围: 避免把手指相对位移硬压扁
 *   3. 关键点限制到“检测框扩展区”而非框内: 保留手指外伸形态
 *   4. Pose WBF 融合: 对重叠候选的框和关键点一起加权平均，降低抖动
 *   5. 首帧诊断输出: 打印 tensor 统计和解码示例，便于调试
 *   6. 严格质量过滤: 最小框尺寸、面积比、宽高比、最少可见关键点数
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
// YOLOv8-Pose 的关键点分支是直接回归，原始偏移值可能远大于 2。
// 之前把 raw 值硬裁到 [-2, 2] 会明显压扁手指，导致手势形态错误。
// 这里只保留一个宽松的防爆范围，用来兜底异常量化值。
constexpr float kRawKptClamp = 16.0f;
constexpr int   kMinVisKpts  = 3;        // 至少3个可见关键点
constexpr float kKptBoxMarginRatio = 0.35f; // 关键点允许落在检测框外的扩展比例
constexpr float kPoseVisScoreThreshold = 0.35f;
constexpr float kMinAcceptedHandScore = 0.55f;   // 避免 0.500(logit=0) 的量化噪声候选
constexpr float kMinInsideRatio = 0.55f;         // 可见关键点至少大部分应落在检测框附近
constexpr float kMaxEdgeRatio = 0.55f;           // 太多关键点贴图像边缘，通常是假手
constexpr float kMinKptSpreadRatio = 0.10f;      // 可见关键点至少要有基本展开，不能缩成一团
constexpr float kKptInsideTolRatio = 0.05f;
constexpr float kImageEdgeTolPx = 2.0f;

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

struct PoseWBFCluster {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float total_score = 0.0f;
    int count = 0;
    std::vector<float> kpt_x_sum;
    std::vector<float> kpt_y_sum;
    std::vector<float> kpt_xy_weight_sum;
    std::vector<float> kpt_vis_sum;
    std::vector<float> kpt_vis_weight_sum;

    PoseWBFCluster()
        : kpt_x_sum(HANDPOSEGRAY::kNumKeypoints, 0.0f),
          kpt_y_sum(HANDPOSEGRAY::kNumKeypoints, 0.0f),
          kpt_xy_weight_sum(HANDPOSEGRAY::kNumKeypoints, 0.0f),
          kpt_vis_sum(HANDPOSEGRAY::kNumKeypoints, 0.0f),
          kpt_vis_weight_sum(HANDPOSEGRAY::kNumKeypoints, 0.0f) {}

    void Add(const std::array<float, 4>& box,
             float score,
             const std::vector<std::array<float, 3>>& kpts) {
        x1 += box[0] * score;
        y1 += box[1] * score;
        x2 += box[2] * score;
        y2 += box[3] * score;
        total_score += score;
        count += 1;

        for (int k = 0; k < HANDPOSEGRAY::kNumKeypoints; ++k) {
            const float vis = Clamp(kpts[k][2], 0.0f, 1.0f);
            const float xy_weight = score * (0.25f + 0.75f * vis);
            kpt_x_sum[k] += kpts[k][0] * xy_weight;
            kpt_y_sum[k] += kpts[k][1] * xy_weight;
            kpt_xy_weight_sum[k] += xy_weight;
            kpt_vis_sum[k] += vis * score;
            kpt_vis_weight_sum[k] += score;
        }
    }

    std::array<float, 4> GetBox() const {
        if (total_score <= 1e-6f) {
            return {0.0f, 0.0f, 0.0f, 0.0f};
        }
        const float inv = 1.0f / total_score;
        return {x1 * inv, y1 * inv, x2 * inv, y2 * inv};
    }

    float GetScore() const {
        if (count <= 0) return 0.0f;
        const float avg = total_score / static_cast<float>(count);
        const float multi_bonus = std::min(0.15f, 0.05f * static_cast<float>(count - 1));
        return std::min(1.0f, avg + multi_bonus);
    }

    std::vector<std::array<float, 3>> GetKeypoints() const {
        std::vector<std::array<float, 3>> fused(HANDPOSEGRAY::kNumKeypoints);
        const auto box = GetBox();
        const float cx = 0.5f * (box[0] + box[2]);
        const float cy = 0.5f * (box[1] + box[3]);
        for (int k = 0; k < HANDPOSEGRAY::kNumKeypoints; ++k) {
            float kx = cx;
            float ky = cy;
            if (kpt_xy_weight_sum[k] > 1e-6f) {
                const float inv = 1.0f / kpt_xy_weight_sum[k];
                kx = kpt_x_sum[k] * inv;
                ky = kpt_y_sum[k] * inv;
            }
            float kv = 0.0f;
            if (kpt_vis_weight_sum[k] > 1e-6f) {
                kv = kpt_vis_sum[k] / kpt_vis_weight_sum[k];
            }
            fused[k] = {kx, ky, Clamp(kv, 0.0f, 1.0f)};
        }
        return fused;
    }
};

void WeightedPoseFusion(const std::vector<std::array<float, 4>>& boxes,
                        const std::vector<float>& scores,
                        const std::vector<std::vector<std::array<float, 3>>>& keypoints,
                        float iou_thresh,
                        std::vector<std::array<float, 4>>* out_boxes,
                        std::vector<float>* out_scores,
                        std::vector<std::vector<std::array<float, 3>>>* out_keypoints) {
    if (boxes.empty()) return;

    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });

    std::vector<PoseWBFCluster> clusters;
    std::vector<bool> used(boxes.size(), false);

    for (int i : order) {
        if (used[i]) continue;

        PoseWBFCluster cluster;
        cluster.Add(boxes[i], scores[i], keypoints[i]);
        used[i] = true;

        for (int j : order) {
            if (used[j]) continue;
            if (IoU(boxes[i], boxes[j]) > iou_thresh) {
                cluster.Add(boxes[j], scores[j], keypoints[j]);
                used[j] = true;
            }
        }

        clusters.push_back(std::move(cluster));
    }

    out_boxes->reserve(clusters.size());
    out_scores->reserve(clusters.size());
    out_keypoints->reserve(clusters.size());
    for (const auto& cluster : clusters) {
        out_boxes->push_back(cluster.GetBox());
        out_scores->push_back(cluster.GetScore());
        out_keypoints->push_back(cluster.GetKeypoints());
    }
}

struct PoseCandidateMetrics {
    int visible = 0;
    int inside = 0;
    int edge = 0;
    float inside_ratio = 0.0f;
    float edge_ratio = 0.0f;
    float spread_ratio = 0.0f;
    float quality = -1e9f;
};

PoseCandidateMetrics EvaluatePoseCandidate(const std::array<float, 4>& box,
                                           const std::vector<std::array<float, 3>>& kpts,
                                           float det_w,
                                           float det_h,
                                           float score) {
    PoseCandidateMetrics metrics;
    const float bw = std::max(1.0f, box[2] - box[0]);
    const float bh = std::max(1.0f, box[3] - box[1]);
    const float tol = std::max(2.0f, std::max(bw, bh) * kKptInsideTolRatio);

    float min_x = 1e9f;
    float min_y = 1e9f;
    float max_x = -1e9f;
    float max_y = -1e9f;

    for (const auto& kp : kpts) {
        if (kp[2] < kPoseVisScoreThreshold) continue;
        metrics.visible += 1;

        const float kx = kp[0];
        const float ky = kp[1];
        const bool inside =
            (kx >= box[0] - tol && kx <= box[2] + tol &&
             ky >= box[1] - tol && ky <= box[3] + tol);
        if (inside) metrics.inside += 1;

        const bool on_edge =
            (kx <= kImageEdgeTolPx || kx >= det_w - kImageEdgeTolPx ||
             ky <= kImageEdgeTolPx || ky >= det_h - kImageEdgeTolPx);
        if (on_edge) metrics.edge += 1;

        if (kx < min_x) min_x = kx;
        if (ky < min_y) min_y = ky;
        if (kx > max_x) max_x = kx;
        if (ky > max_y) max_y = ky;
    }

    if (metrics.visible > 0) {
        metrics.inside_ratio = static_cast<float>(metrics.inside) / static_cast<float>(metrics.visible);
        metrics.edge_ratio = static_cast<float>(metrics.edge) / static_cast<float>(metrics.visible);
        const float spread_x = std::max(0.0f, max_x - min_x) / bw;
        const float spread_y = std::max(0.0f, max_y - min_y) / bh;
        metrics.spread_ratio = std::max(spread_x, spread_y);
    }

    metrics.quality = score +
                      0.25f * metrics.inside_ratio -
                      0.20f * metrics.edge_ratio +
                      0.10f * metrics.spread_ratio +
                      0.01f * static_cast<float>(metrics.visible);
    return metrics;
}

bool IsPoseCandidateValid(const PoseCandidateMetrics& metrics, float score) {
    if (score < kMinAcceptedHandScore) return false;
    if (metrics.visible < kMinVisKpts) return false;
    if (metrics.inside_ratio < kMinInsideRatio) return false;
    if (metrics.edge_ratio > kMaxEdgeRatio) return false;
    if (metrics.spread_ratio < kMinKptSpreadRatio) return false;
    return true;
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

                // 宽松钳位，只拦截明显异常值；不要把正常的手指位移压扁。
                rx = Clamp(rx, -kRawKptClamp, kRawKptClamp);
                ry = Clamp(ry, -kRawKptClamp, kRawKptClamp);

                float kx = (rx * 2.0f + static_cast<float>(gx)) * stride;
                float ky = (ry * 2.0f + static_cast<float>(gy)) * stride;
                float kv = Sigmoid(rv);

                // 关键点允许超出检测框少量范围，避免手指被裁到框边上。
                const float kpt_margin = std::max(bw, bh) * kKptBoxMarginRatio;
                kx = Clamp(kx, std::max(0.0f, x1 - kpt_margin), std::min(dw, x2 + kpt_margin));
                ky = Clamp(ky, std::max(0.0f, y1 - kpt_margin), std::min(dh, y2 + kpt_margin));

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
// Postprocess - Pose WBF 融合 + 坐标缩放
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

    // 先收集 top_k 候选，再做框+关键点联合融合，降低相邻 anchor 抢答导致的抖动。
    std::vector<std::array<float, 4>> top_boxes;
    std::vector<float> top_scores;
    std::vector<std::vector<std::array<float, 3>>> top_kpts;
    top_boxes.reserve(order.size());
    top_scores.reserve(order.size());
    top_kpts.reserve(order.size());
    for (int idx : order) {
        if (scores->at(idx) < *conf_threshold) continue;
        top_boxes.push_back(boxes->at(idx));
        top_scores.push_back(scores->at(idx));
        top_kpts.push_back(keypoints->at(idx));
    }
    if (top_boxes.empty()) return;

    std::vector<std::array<float, 4>> fused_boxes;
    std::vector<float> fused_scores;
    std::vector<std::vector<std::array<float, 3>>> fused_kpts;
    WeightedPoseFusion(top_boxes, top_scores, top_kpts, nms_threshold,
                       &fused_boxes, &fused_scores, &fused_kpts);

    std::vector<PoseCandidateMetrics> fused_metrics(fused_boxes.size());
    std::vector<int> fused_order;
    fused_order.reserve(fused_boxes.size());
    for (int i = 0; i < static_cast<int>(fused_boxes.size()); ++i) {
        fused_metrics[i] = EvaluatePoseCandidate(fused_boxes[i], fused_kpts[i], det_shape[0], det_shape[1],
                                                 fused_scores[i]);
        if (IsPoseCandidateValid(fused_metrics[i], fused_scores[i])) {
            fused_order.push_back(i);
        }
    }
    std::sort(fused_order.begin(), fused_order.end(), [&](int a, int b) {
        if (fused_metrics[a].quality == fused_metrics[b].quality) {
            return fused_scores[a] > fused_scores[b];
        }
        return fused_metrics[a].quality > fused_metrics[b].quality;
    });
    if (static_cast<int>(fused_order.size()) > keep_top_k) {
        fused_order.resize(keep_top_k);
    }

    // 缩放到原图坐标
    const float iw = static_cast<float>(img_shape[0]);
    const float ih = static_cast<float>(img_shape[1]);
    result->Reserve(static_cast<int>(fused_order.size()));

    for (int idx : fused_order) {
        auto& b = fused_boxes[idx];
        result->boxes.push_back({
            Clamp(b[0] * w_scale, 0, iw), Clamp(b[1] * h_scale, 0, ih),
            Clamp(b[2] * w_scale, 0, iw), Clamp(b[3] * h_scale, 0, ih)
        });
        result->scores.push_back(fused_scores[idx]);

        std::vector<std::array<float, 3>> sk(kNumKeypoints);
        const auto& kpts = fused_kpts[idx];
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
