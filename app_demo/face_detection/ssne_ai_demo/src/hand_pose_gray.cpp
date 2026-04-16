/*
 * @Filename: hand_pose_gray.cpp
 * @Description: YOLOv8n-Pose hand keypoint detection
 *   - 9 output heads: 3 cls(cv3) + 3 reg(cv2) + 3 kpt(cv4)
 *   - 21 hand keypoints, each (x, y, visibility)
 *   - DFL decode (double precision, same as eye model)
 *   - Robust post-processing: NMS + min-box + aspect-ratio + kpt-in-box constraint
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

#include "../include/utils.hpp"

namespace {
// Enable to print detailed per-frame diagnostics (first N frames only)
constexpr bool kHandVerboseDebug = false;
constexpr int  kHandDebugMaxFrames = 5;  // only print first N frames
int g_hand_debug_frame_count = 0;

// Post-processing quality filters
constexpr float kMinBoxSizePx = 30.0f;    // reject boxes smaller than 30px on either dimension
constexpr float kMinAspectRatio = 0.25f;   // reject extremely narrow boxes (w/h < 0.25 or h/w < 0.25)
constexpr float kMaxAspectRatio = 4.0f;    // reject extremely tall/wide boxes
constexpr float kKptBoxMargin = 0.3f;      // allow keypoints up to 30% outside box before clamping
constexpr int   kMinValidKpts = 5;         // require at least 5 visible keypoints per hand

float Sigmoid(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

// DFL decode - double precision (same as eye_det_gray.cpp)
static float DFL(const float* tensor, int start_channel, int spatial, int idx) {
    float max_val = -1e9f;
    for (int i = 0; i < 16; ++i) {
        float val = tensor[(start_channel + i) * spatial + idx];
        if (val > max_val) {
            max_val = val;
        }
    }

    double sum = 0.0;
    double res = 0.0;
    for (int i = 0; i < 16; ++i) {
        float raw = tensor[(start_channel + i) * spatial + idx] - max_val;
        if (raw < -15.0f) raw = -15.0f;
        double weight = std::exp(static_cast<double>(raw));
        sum += weight;
        res += weight * static_cast<double>(i);
    }
    if (sum < 1e-12) return 0.0f;
    return static_cast<float>(res / sum);
}

float IoU(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]);
    const float y2 = std::min(a[3], b[3]);
    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);
    const float inter = w * h;
    const float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    const float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    const float uni = area_a + area_b - inter;
    return uni <= 0.0f ? 0.0f : inter / uni;
}

// Check box quality: minimum size and reasonable aspect ratio
bool IsValidBox(const std::array<float, 4>& box) {
    const float w = box[2] - box[0];
    const float h = box[3] - box[1];
    if (w < kMinBoxSizePx || h < kMinBoxSizePx) return false;
    const float ar = w / std::max(1e-6f, h);
    if (ar < kMinAspectRatio || ar > kMaxAspectRatio) return false;
    return true;
}
} // namespace


void HANDPOSEGRAY::Initialize(std::string& model_path, std::array<int, 2>* in_img_shape,
                              std::array<int, 2>* in_det_shape, int in_box_len) {
    nms_threshold = 0.35f;   // more aggressive NMS (was 0.45)
    keep_top_k = 3;          // max 3 hands (was 10)
    top_k = 30;              // pre-NMS candidates (was 100)

    img_shape = *in_img_shape;
    det_shape = *in_det_shape;
    box_len = in_box_len;
    w_scale = static_cast<float>(img_shape[0]) / static_cast<float>(det_shape[0]);
    h_scale = static_cast<float>(img_shape[1]) / static_cast<float>(det_shape[1]);
    steps = {8, 16, 32};

    printf("[INFO] HandPose: img=%dx%d det=%dx%d w_scale=%.3f h_scale=%.3f\n",
           img_shape[0], img_shape[1], det_shape[0], det_shape[1], w_scale, h_scale);
    printf("[INFO] HandPose: nms=%.2f keep_top_k=%d top_k=%d min_box=%.0f\n",
           nms_threshold, keep_top_k, top_k, kMinBoxSizePx);

    char* model_path_char = const_cast<char*>(model_path.c_str());
    model_id = ssne_loadmodel(model_path_char, SSNE_STATIC_ALLOC);

    uint32_t det_width = static_cast<uint32_t>(det_shape[0]);
    uint32_t det_height = static_cast<uint32_t>(det_shape[1]);
    inputs[0] = create_tensor(det_width, det_height, SSNE_Y_8, SSNE_BUF_AI);

    int dtype = SSNE_UINT8;
    if (ssne_get_model_input_dtype(model_id, &dtype) == 0) {
        set_data_type(inputs[0], static_cast<uint8_t>(dtype));
        printf("[INFO] HandPose model input dtype: %d\n", dtype);
    }

    const int normalize_ret = SetNormalize(pipe_offline, model_id);
    if (normalize_ret != 0) {
        printf("[WARN] HandPose model SetNormalize failed, ret: %d\n", normalize_ret);
    }

    printf("[INFO] HandPose model loaded, kNumKeypoints=%d kKptChannels=%d\n",
           kNumKeypoints, kKptChannels);
}


void HANDPOSEGRAY::DecodeBranch(const float* cls_head, const float* box_head,
                                const float* kpt_head,
                                int feat_h, int feat_w, int stride,
                                float conf_threshold,
                                std::vector<std::array<float, 4>>* boxes,
                                std::vector<float>* scores,
                                std::vector<std::vector<std::array<float, 3>>>* keypoints) const {
    const int spatial = feat_h * feat_w;
    const float det_w = static_cast<float>(det_shape[0]);
    const float det_h = static_cast<float>(det_shape[1]);

    for (int y = 0; y < feat_h; ++y) {
        for (int x = 0; x < feat_w; ++x) {
            const int idx = y * feat_w + x;
            const float score = Sigmoid(cls_head[idx]);
            if (score < conf_threshold) {
                continue;
            }

            // DFL box decode
            const float cx = (static_cast<float>(x) + 0.5f) * stride;
            const float cy = (static_cast<float>(y) + 0.5f) * stride;
            const float l = DFL(box_head, 0, spatial, idx) * stride;
            const float t = DFL(box_head, 16, spatial, idx) * stride;
            const float r = DFL(box_head, 32, spatial, idx) * stride;
            const float b = DFL(box_head, 48, spatial, idx) * stride;

            float x1 = std::max(0.0f, cx - l);
            float y1 = std::max(0.0f, cy - t);
            float x2 = std::min(det_w, cx + r);
            float y2 = std::min(det_h, cy + b);
            if (x2 <= x1 || y2 <= y1) {
                continue;
            }

            // Early reject: minimum box size (in det-space pixels)
            const float bw = x2 - x1;
            const float bh = y2 - y1;
            if (bw < kMinBoxSizePx || bh < kMinBoxSizePx) {
                continue;
            }

            // Keypoint decode (YOLOv8-Pose formula)
            // Clamp keypoints to box + margin to prevent wild scatter
            const float margin_x = bw * kKptBoxMargin;
            const float margin_y = bh * kKptBoxMargin;
            const float kpt_x_min = std::max(0.0f, x1 - margin_x);
            const float kpt_y_min = std::max(0.0f, y1 - margin_y);
            const float kpt_x_max = std::min(det_w, x2 + margin_x);
            const float kpt_y_max = std::min(det_h, y2 + margin_y);

            std::vector<std::array<float, 3>> kpts(kNumKeypoints);
            int valid_kpt_count = 0;
            for (int k = 0; k < kNumKeypoints; ++k) {
                const float raw_x = kpt_head[(k * 3 + 0) * spatial + idx];
                const float raw_y = kpt_head[(k * 3 + 1) * spatial + idx];
                const float raw_v = kpt_head[(k * 3 + 2) * spatial + idx];

                float kpt_x = (raw_x * 2.0f + static_cast<float>(x)) * stride;
                float kpt_y = (raw_y * 2.0f + static_cast<float>(y)) * stride;
                const float kpt_vis = Sigmoid(raw_v);

                // Clamp keypoints to box + margin
                kpt_x = std::max(kpt_x_min, std::min(kpt_x_max, kpt_x));
                kpt_y = std::max(kpt_y_min, std::min(kpt_y_max, kpt_y));

                kpts[k] = {kpt_x, kpt_y, kpt_vis};

                if (kpt_vis > 0.3f) {
                    valid_kpt_count++;
                }
            }

            // Reject detections with too few visible keypoints
            if (valid_kpt_count < kMinValidKpts) {
                continue;
            }

            boxes->push_back({x1, y1, x2, y2});
            scores->push_back(score);
            keypoints->push_back(std::move(kpts));
        }
    }
}


void HANDPOSEGRAY::Postprocess(std::vector<std::array<float, 4>>* boxes,
                               std::vector<float>* scores,
                               std::vector<std::vector<std::array<float, 3>>>* keypoints,
                               HandPoseResult* result, float* conf_threshold) {
    result->Clear();
    if (boxes->empty()) {
        return;
    }

    // Sort by score descending, truncate to top_k
    std::vector<int> order(boxes->size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return scores->at(lhs) > scores->at(rhs);
    });
    if (static_cast<int>(order.size()) > top_k) {
        order.resize(top_k);
    }

    // Standard IoU NMS
    std::vector<bool> suppressed(boxes->size(), false);
    std::vector<int> kept;
    kept.reserve(keep_top_k);

    for (int i = 0; i < static_cast<int>(order.size()); ++i) {
        const int idx_i = order[i];
        if (suppressed[idx_i]) continue;
        if (scores->at(idx_i) < *conf_threshold) continue;

        // Aspect ratio check
        if (!IsValidBox(boxes->at(idx_i))) continue;

        kept.push_back(idx_i);
        if (static_cast<int>(kept.size()) >= keep_top_k) break;

        for (int j = i + 1; j < static_cast<int>(order.size()); ++j) {
            const int idx_j = order[j];
            if (suppressed[idx_j]) continue;
            if (IoU(boxes->at(idx_i), boxes->at(idx_j)) > nms_threshold) {
                suppressed[idx_j] = true;
            }
        }
    }

    // Output: scale to original image coordinates
    const float img_w = static_cast<float>(img_shape[0]);
    const float img_h = static_cast<float>(img_shape[1]);
    result->Reserve(static_cast<int>(kept.size()));

    for (int idx : kept) {
        // Scale bounding box
        std::array<float, 4> box = boxes->at(idx);
        box[0] = std::max(0.0f, std::min(box[0] * w_scale, img_w));
        box[1] = std::max(0.0f, std::min(box[1] * h_scale, img_h));
        box[2] = std::max(0.0f, std::min(box[2] * w_scale, img_w));
        box[3] = std::max(0.0f, std::min(box[3] * h_scale, img_h));
        result->boxes.emplace_back(box);
        result->scores.emplace_back(scores->at(idx));

        // Scale keypoints, clamp to image bounds
        std::vector<std::array<float, 3>> scaled_kpts(kNumKeypoints);
        const auto& kpts = keypoints->at(idx);
        for (int k = 0; k < kNumKeypoints; ++k) {
            scaled_kpts[k] = {
                std::max(0.0f, std::min(kpts[k][0] * w_scale, img_w)),
                std::max(0.0f, std::min(kpts[k][1] * h_scale, img_h)),
                kpts[k][2]  // visibility unchanged
            };
        }
        result->keypoints.emplace_back(std::move(scaled_kpts));
    }
}


void HANDPOSEGRAY::Predict(ssne_tensor_t* img_in, HandPoseResult* result,
                           float conf_threshold) {
    result->Clear();

    int ret = RunAiPreprocessPipe(pipe_offline, *img_in, inputs[0]);
    if (ret != 0) {
        printf("[ERROR] HandPose RunAiPreprocessPipe failed, ret=%d\n", ret);
        return;
    }

    ret = ssne_inference(model_id, 1, inputs);
    if (ret != 0) {
        fprintf(stderr, "[ERROR] HandPose ssne_inference fail! ret=%d\n", ret);
        return;
    }

    ssne_getoutput(model_id, 9, outputs);

    // Feature map sizes
    const int feat_h_s8 = det_shape[1] / 8;
    const int feat_w_s8 = det_shape[0] / 8;
    const int feat_h_s16 = det_shape[1] / 16;
    const int feat_w_s16 = det_shape[0] / 16;
    const int feat_h_s32 = det_shape[1] / 32;
    const int feat_w_s32 = det_shape[0] / 32;

    // Expected tensor element counts
    const uint32_t exp_cls_s8  = feat_h_s8  * feat_w_s8  * 1;
    const uint32_t exp_cls_s16 = feat_h_s16 * feat_w_s16 * 1;
    const uint32_t exp_cls_s32 = feat_h_s32 * feat_w_s32 * 1;
    const uint32_t exp_box_s8  = feat_h_s8  * feat_w_s8  * 64;
    const uint32_t exp_box_s16 = feat_h_s16 * feat_w_s16 * 64;
    const uint32_t exp_box_s32 = feat_h_s32 * feat_w_s32 * 64;
    const uint32_t exp_kpt_s8  = feat_h_s8  * feat_w_s8  * kKptChannels;
    const uint32_t exp_kpt_s16 = feat_h_s16 * feat_w_s16 * kKptChannels;
    const uint32_t exp_kpt_s32 = feat_h_s32 * feat_w_s32 * kKptChannels;

    float* cls_s8 = nullptr;  float* cls_s16 = nullptr;  float* cls_s32 = nullptr;
    float* box_s8 = nullptr;  float* box_s16 = nullptr;  float* box_s32 = nullptr;
    float* kpt_s8 = nullptr;  float* kpt_s16 = nullptr;  float* kpt_s32 = nullptr;

    // One-time diagnostic: dump tensor sizes and raw value ranges
    const bool debug_this_frame = kHandVerboseDebug &&
                                  (g_hand_debug_frame_count < kHandDebugMaxFrames);
    if (debug_this_frame) {
        g_hand_debug_frame_count++;
        printf("\n[DEBUG] === HandPose Frame %d ===\n", g_hand_debug_frame_count);
    }

    for (int i = 0; i < 9; ++i) {
        uint32_t size = get_total_size(outputs[i]);
        float* data = reinterpret_cast<float*>(get_data(outputs[i]));

        if (debug_this_frame) {
            // Print tensor size and first few raw values
            printf("[DEBUG] output[%d] size=%u", i, size);
            if (data && size > 0) {
                float vmin = data[0], vmax = data[0];
                for (uint32_t j = 1; j < std::min(size, 1000u); ++j) {
                    if (data[j] < vmin) vmin = data[j];
                    if (data[j] > vmax) vmax = data[j];
                }
                printf(" range=[%.4f, %.4f]", vmin, vmax);
            }
            printf("\n");
        }

        if      (size == exp_cls_s8)  cls_s8  = data;
        else if (size == exp_cls_s16) cls_s16 = data;
        else if (size == exp_cls_s32) cls_s32 = data;
        else if (size == exp_box_s8)  box_s8  = data;
        else if (size == exp_box_s16) box_s16 = data;
        else if (size == exp_box_s32) box_s32 = data;
        else if (size == exp_kpt_s8)  kpt_s8  = data;
        else if (size == exp_kpt_s16) kpt_s16 = data;
        else if (size == exp_kpt_s32) kpt_s32 = data;
        else {
            printf("[WARN] HandPose output[%d] size=%u unmatched\n", i, size);
        }
    }

    if (!cls_s8 || !cls_s16 || !cls_s32 ||
        !box_s8 || !box_s16 || !box_s32 ||
        !kpt_s8 || !kpt_s16 || !kpt_s32) {
        printf("[ERROR] HandPose output tensor size mismatch!\n");
        printf("  Expected cls(%u, %u, %u) box(%u, %u, %u) kpt(%u, %u, %u)\n",
               exp_cls_s8, exp_cls_s16, exp_cls_s32,
               exp_box_s8, exp_box_s16, exp_box_s32,
               exp_kpt_s8, exp_kpt_s16, exp_kpt_s32);
        printf("  Actual sizes:");
        for (int i = 0; i < 9; ++i) {
            printf(" [%d]=%u", i, get_total_size(outputs[i]));
        }
        printf("\n");
        return;
    }

    // Log cls head stats for first few frames
    if (debug_this_frame) {
        for (int s = 0; s < 3; ++s) {
            float* cls = (s == 0) ? cls_s8 : (s == 1) ? cls_s16 : cls_s32;
            int sp = (s == 0) ? (feat_h_s8 * feat_w_s8) :
                     (s == 1) ? (feat_h_s16 * feat_w_s16) :
                                (feat_h_s32 * feat_w_s32);
            int stride_val = (s == 0) ? 8 : (s == 1) ? 16 : 32;
            float max_raw = -1e9f;
            int max_idx = 0;
            int above_thresh = 0;
            for (int j = 0; j < sp; ++j) {
                float sig = Sigmoid(cls[j]);
                if (sig > conf_threshold) above_thresh++;
                if (cls[j] > max_raw) { max_raw = cls[j]; max_idx = j; }
            }
            printf("[DEBUG] cls_s%d: spatial=%d max_raw=%.4f max_sig=%.4f above_thresh=%d\n",
                   stride_val, sp, max_raw, Sigmoid(max_raw), above_thresh);
        }
    }

    std::vector<std::array<float, 4>> bboxes;
    std::vector<float> scores;
    std::vector<std::vector<std::array<float, 3>>> all_kpts;

    DecodeBranch(cls_s8,  box_s8,  kpt_s8,  feat_h_s8,  feat_w_s8,  8,
                 conf_threshold, &bboxes, &scores, &all_kpts);
    DecodeBranch(cls_s16, box_s16, kpt_s16, feat_h_s16, feat_w_s16, 16,
                 conf_threshold, &bboxes, &scores, &all_kpts);
    DecodeBranch(cls_s32, box_s32, kpt_s32, feat_h_s32, feat_w_s32, 32,
                 conf_threshold, &bboxes, &scores, &all_kpts);

    if (debug_this_frame) {
        printf("[DEBUG] Pre-NMS candidates: %zu (threshold=%.2f)\n",
               bboxes.size(), conf_threshold);
        for (size_t i = 0; i < std::min(bboxes.size(), size_t(5)); ++i) {
            printf("  [%zu] box=[%.0f,%.0f,%.0f,%.0f] score=%.3f\n",
                   i, bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3], scores[i]);
        }
    }

    Postprocess(&bboxes, &scores, &all_kpts, result, &conf_threshold);

    if (debug_this_frame && !result->boxes.empty()) {
        printf("[DEBUG] Post-NMS results: %zu hands\n", result->boxes.size());
        for (size_t i = 0; i < result->boxes.size(); ++i) {
            printf("  hand[%zu] box=[%.1f,%.1f,%.1f,%.1f] score=%.3f\n",
                   i, result->boxes[i][0], result->boxes[i][1],
                   result->boxes[i][2], result->boxes[i][3], result->scores[i]);
            // Print first 5 keypoints
            if (i < result->keypoints.size()) {
                for (int k = 0; k < std::min(5, kNumKeypoints); ++k) {
                    const auto& kpt = result->keypoints[i][k];
                    printf("    kpt[%d] x=%.1f y=%.1f vis=%.2f\n", k, kpt[0], kpt[1], kpt[2]);
                }
            }
        }
    }
}


void HANDPOSEGRAY::Release() {
    release_tensor(inputs[0]);
    for (int i = 0; i < 9; ++i) {
        release_tensor(outputs[i]);
    }
    ReleaseAIPreprocessPipe(pipe_offline);
}
