/*
 * @Filename: eye_det_gray.cpp
 * @Description: 灰度眼睛检测实现文件，输出为3个分类头+3个bbox头
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

#include "../include/utils.hpp"

namespace {
float Sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float BoxWidth(const std::array<float, 4>& box) {
    return std::max(0.0f, box[2] - box[0]);
}

float BoxHeight(const std::array<float, 4>& box) {
    return std::max(0.0f, box[3] - box[1]);
}

float BoxCenterX(const std::array<float, 4>& box) {
    return 0.5f * (box[0] + box[2]);
}

float BoxCenterY(const std::array<float, 4>& box) {
    return 0.5f * (box[1] + box[3]);
}

std::array<float, 4> ShrinkBoxAroundCenter(const std::array<float, 4>& box,
                                           float ratio,
                                           float max_w,
                                           float max_h) {
    const float cx = 0.5f * (box[0] + box[2]);
    const float cy = 0.5f * (box[1] + box[3]);
    const float w = std::max(1.0f, box[2] - box[0]) * ratio;
    const float h = std::max(1.0f, box[3] - box[1]) * ratio;
    std::array<float, 4> out = {
        cx - 0.5f * w,
        cy - 0.5f * h,
        cx + 0.5f * w,
        cy + 0.5f * h,
    };
    out[0] = std::max(0.0f, std::min(out[0], max_w));
    out[1] = std::max(0.0f, std::min(out[1], max_h));
    out[2] = std::max(0.0f, std::min(out[2], max_w));
    out[3] = std::max(0.0f, std::min(out[3], max_h));
    return out;
}

std::vector<int> FilterSmallBoxes(const std::vector<std::array<float, 4>>& boxes,
                                  const std::vector<int>& indices,
                                  float min_box_size) {
    std::vector<int> filtered;
    filtered.reserve(indices.size());
    for (int idx : indices) {
        if (BoxWidth(boxes[idx]) >= min_box_size && BoxHeight(boxes[idx]) >= min_box_size) {
            filtered.push_back(idx);
        }
    }
    if (!filtered.empty()) {
        return filtered;
    }
    return indices;
}

std::vector<int> SelectBestEyePair(const std::vector<std::array<float, 4>>& boxes,
                                   const std::vector<float>& scores,
                                   const std::vector<int>& indices,
                                   int pair_candidates,
                                   float pair_y_thresh,
                                   float pair_size_ratio_thresh) {
    if (indices.empty()) {
        return {};
    }
    if (indices.size() <= 2) {
        std::vector<int> ordered = indices;
        std::sort(ordered.begin(), ordered.end(), [&](int lhs, int rhs) {
            return BoxCenterX(boxes[lhs]) < BoxCenterX(boxes[rhs]);
        });
        return ordered;
    }

    std::vector<int> candidates = indices;
    std::sort(candidates.begin(), candidates.end(), [&](int lhs, int rhs) {
        return scores[lhs] > scores[rhs];
    });

    const int candidate_count = std::min(static_cast<int>(candidates.size()), pair_candidates);
    candidates.resize(candidate_count);

    float best_score = -1e9f;
    int best_left = -1;
    int best_right = -1;
    for (int i = 0; i < candidate_count; ++i) {
        for (int j = i + 1; j < candidate_count; ++j) {
            int idx_a = candidates[i];
            int idx_b = candidates[j];
            int left_idx = BoxCenterX(boxes[idx_a]) <= BoxCenterX(boxes[idx_b]) ? idx_a : idx_b;
            int right_idx = left_idx == idx_a ? idx_b : idx_a;

            const float w_l = BoxWidth(boxes[left_idx]);
            const float h_l = BoxHeight(boxes[left_idx]);
            const float w_r = BoxWidth(boxes[right_idx]);
            const float h_r = BoxHeight(boxes[right_idx]);

            const float avg_w = 0.5f * (w_l + w_r);
            const float avg_h = 0.5f * (h_l + h_r);
            if (avg_w <= 0.0f || avg_h <= 0.0f) {
                continue;
            }

            const float y_diff = std::fabs(BoxCenterY(boxes[left_idx]) - BoxCenterY(boxes[right_idx])) / avg_h;
            if (y_diff > pair_y_thresh) {
                continue;
            }

            const float eps = 1e-6f;
            const float size_ratio_w = std::max(w_l, w_r) / std::max(eps, std::min(w_l, w_r));
            const float size_ratio_h = std::max(h_l, h_r) / std::max(eps, std::min(h_l, h_r));
            const float size_ratio = std::max(size_ratio_w, size_ratio_h);
            if (size_ratio > pair_size_ratio_thresh) {
                continue;
            }

            const float horizontal_gap = (BoxCenterX(boxes[right_idx]) - BoxCenterX(boxes[left_idx])) / avg_w;
            if (horizontal_gap < 0.5f || horizontal_gap > 10.0f) {
                continue;
            }

            const float pair_score = (scores[left_idx] + scores[right_idx]) - 0.25f * y_diff - 0.1f * size_ratio;
            if (pair_score > best_score) {
                best_score = pair_score;
                best_left = left_idx;
                best_right = right_idx;
            }
        }
    }

    if (best_left >= 0 && best_right >= 0) {
        return {best_left, best_right};
    }

    // fallback: top2 by score, ordered by x center
    std::vector<int> fallback = candidates;
    if (fallback.size() > 2) {
        fallback.resize(2);
    }
    std::sort(fallback.begin(), fallback.end(), [&](int lhs, int rhs) {
        return BoxCenterX(boxes[lhs]) < BoxCenterX(boxes[rhs]);
    });
    return fallback;
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
}

void EYEDETGRAY::Initialize(std::string& model_path, std::array<int, 2>* in_img_shape,
                            std::array<int, 2>* in_det_shape, int in_box_len) {
    nms_threshold = 0.25f;
    keep_top_k = 2;
    top_k = 12;
    eye_pair_only = false;
    min_box_size = 2.0f;
    pair_candidates = 12;
    pair_y_thresh = 1.5f;
    pair_size_ratio_thresh = 2.5f;
    printf("[INFO] Eye postprocess: pair_only=%d min_box=%.1f pair_candidates=%d pair_y_thresh=%.2f pair_size_ratio_thresh=%.2f\n",
           eye_pair_only ? 1 : 0,
           min_box_size,
           pair_candidates,
           pair_y_thresh,
           pair_size_ratio_thresh);
    img_shape = *in_img_shape;
    det_shape = *in_det_shape;
    box_len = in_box_len;
    w_scale = static_cast<float>(img_shape[0]) / static_cast<float>(det_shape[0]);
    h_scale = static_cast<float>(img_shape[1]) / static_cast<float>(det_shape[1]);
    steps = {8, 16, 32};

    char* model_path_char = const_cast<char*>(model_path.c_str());
    model_id = ssne_loadmodel(model_path_char, SSNE_STATIC_ALLOC);

    uint32_t det_width = static_cast<uint32_t>(det_shape[0]);
    uint32_t det_height = static_cast<uint32_t>(det_shape[1]);
    inputs[0] = create_tensor(det_width, det_height, SSNE_Y_8, SSNE_BUF_AI);

    int dtype = SSNE_UINT8;
    if (ssne_get_model_input_dtype(model_id, &dtype) == 0) {
        set_data_type(inputs[0], static_cast<uint8_t>(dtype));
        printf("[INFO] Eye model input dtype: %d\n", dtype);
    }

    const int normalize_ret = SetNormalize(pipe_offline, model_id);
    if (normalize_ret != 0) {
        printf("[WARN] Eye model SetNormalize failed, ret: %d\n", normalize_ret);
    }
}


static float DFL(const float* tensor, int start_channel, int spatial, int idx) {
    float max_val = -1e9f;
    for (int i = 0; i < 16; ++i) {
        float val = tensor[(start_channel + i) * spatial + idx];
        if (val > max_val) {
            max_val = val;
        }
    }

    float sum = 0.0f;
    float res = 0.0f;
    for (int i = 0; i < 16; ++i) {
        float weight = std::exp(tensor[(start_channel + i) * spatial + idx] - max_val);
        sum += weight;
        res += weight * static_cast<float>(i);
    }
    return res / sum;
}

void PrintClsHeadStats(const char* tag, const float* cls_head, int spatial) {
    float max_raw = -1e9f;
    float min_raw = 1e9f;
    int max_idx = -1;
    for (int i = 0; i < spatial; ++i) {
        const float v = cls_head[i];
        if (v > max_raw) {
            max_raw = v;
            max_idx = i;
        }
        if (v < min_raw) {
            min_raw = v;
        }
    }
    const float max_sigmoid = Sigmoid(max_raw);
    printf("[DEBUG] %s cls stats: min_raw=%.6f max_raw=%.6f max_sigmoid=%.6f max_idx=%d\n",
           tag, min_raw, max_raw, max_sigmoid, max_idx);
}

void EYEDETGRAY::DecodeBranch(const float* cls_head, const float* box_head,
                              int feat_h, int feat_w, int stride,
                              float conf_threshold,
                              std::vector<std::array<float, 4>>* boxes,
                              std::vector<float>* scores) const {
    const int spatial = feat_h * feat_w;
    for (int y = 0; y < feat_h; ++y) {
        for (int x = 0; x < feat_w; ++x) {
            const int idx = y * feat_w + x;
            const float score = Sigmoid(cls_head[idx]);
            if (score < conf_threshold) {
                continue;
            }

            const float cx = (static_cast<float>(x) + 0.5f) * stride;
            const float cy = (static_cast<float>(y) + 0.5f) * stride;
            const float l = DFL(box_head, 0, spatial, idx) * stride;
            const float t = DFL(box_head, 16, spatial, idx) * stride;
            const float r = DFL(box_head, 32, spatial, idx) * stride;
            const float b = DFL(box_head, 48, spatial, idx) * stride;

            float x1 = std::max(0.0f, cx - l);
            float y1 = std::max(0.0f, cy - t);
            float x2 = std::min(static_cast<float>(det_shape[0]), cx + r);
            float y2 = std::min(static_cast<float>(det_shape[1]), cy + b);
            if (x2 <= x1 || y2 <= y1) {
                continue;
            }

            boxes->push_back({x1, y1, x2, y2});
            scores->push_back(score);
        }
    }
}

void EYEDETGRAY::Postprocess(std::vector<std::array<float, 4>>* boxes,
                             std::vector<float>* scores,
                             FaceDetectionResult* result,
                             float* conf_threshold) {
    constexpr float kEyeBoxShrinkRatio = 0.72f;
    result->Clear();
    if (boxes->empty()) {
        return;
    }

    std::vector<int> order(boxes->size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return scores->at(lhs) > scores->at(rhs);
    });
    if (static_cast<int>(order.size()) > top_k) {
        order.resize(top_k);
    }

    std::vector<int> keep;
    keep.reserve(order.size());
    for (int idx : order) {
        if (scores->at(idx) < *conf_threshold) {
            continue;
        }
        bool suppressed = false;
        for (int kept_idx : keep) {
            if (IoU(boxes->at(idx), boxes->at(kept_idx)) > nms_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            keep.push_back(idx);
        }
    }

    keep = FilterSmallBoxes(*boxes, keep, min_box_size);
    if (eye_pair_only) {
        keep = SelectBestEyePair(*boxes,
                                 *scores,
                                 keep,
                                 pair_candidates,
                                 pair_y_thresh,
                                 pair_size_ratio_thresh);
    } else if (static_cast<int>(keep.size()) > keep_top_k) {
        keep.resize(keep_top_k);
    }

    result->Reserve(static_cast<int>(keep.size()));
    for (int idx : keep) {
        std::array<float, 4> box = boxes->at(idx);
        box[0] = std::max(0.0f, std::min(box[0] * w_scale, static_cast<float>(img_shape[0])));
        box[1] = std::max(0.0f, std::min(box[1] * h_scale, static_cast<float>(img_shape[1])));
        box[2] = std::max(0.0f, std::min(box[2] * w_scale, static_cast<float>(img_shape[0])));
        box[3] = std::max(0.0f, std::min(box[3] * h_scale, static_cast<float>(img_shape[1])));
        box = ShrinkBoxAroundCenter(box,
                                    kEyeBoxShrinkRatio,
                                    static_cast<float>(img_shape[0]),
                                    static_cast<float>(img_shape[1]));
        result->boxes.emplace_back(box);
        result->scores.emplace_back(scores->at(idx));
    }
}

void EYEDETGRAY::Predict(ssne_tensor_t* img_in, FaceDetectionResult* result, float conf_threshold) {
    result->Clear();

    int ret = RunAiPreprocessPipe(pipe_offline, *img_in, inputs[0]);
    if (ret != 0) {
        printf("[ERROR] Eye RunAiPreprocessPipe failed, ret=%d\n", ret);
        return;
    }

    ret = ssne_inference(model_id, 1, inputs);
    if (ret != 0) {
        fprintf(stderr, "eye ssne inference fail! ret=%d\n", ret);
        return;
    }

    ssne_getoutput(model_id, 6, outputs);
    std::vector<std::array<float, 4>> bboxes;
    std::vector<float> scores;

    const int feat_h_s8 = det_shape[1] / 8;
    const int feat_w_s8 = det_shape[0] / 8;
    const int feat_h_s16 = det_shape[1] / 16;
    const int feat_w_s16 = det_shape[0] / 16;
    const int feat_h_s32 = det_shape[1] / 32;
    const int feat_w_s32 = det_shape[0] / 32;

    float* cls_s8 = nullptr;
    float* cls_s16 = nullptr;
    float* cls_s32 = nullptr;
    float* box_s8 = nullptr;
    float* box_s16 = nullptr;
    float* box_s32 = nullptr;

    // NOTE: get_total_size() on this SDK returns element count (not byte count).
    uint32_t exp_cls_s8 = feat_h_s8 * feat_w_s8 * 1;
    uint32_t exp_cls_s16 = feat_h_s16 * feat_w_s16 * 1;
    uint32_t exp_cls_s32 = feat_h_s32 * feat_w_s32 * 1;
    uint32_t exp_box_s8 = feat_h_s8 * feat_w_s8 * 64;
    uint32_t exp_box_s16 = feat_h_s16 * feat_w_s16 * 64;
    uint32_t exp_box_s32 = feat_h_s32 * feat_w_s32 * 64;

    for (int i = 0; i < 6; ++i) {
        uint32_t size = get_total_size(outputs[i]);
        printf("\n[DEBUG] Output tensor %d total_size: %u elements\n", i, size);
        
        uint8_t* raw_u8 = reinterpret_cast<uint8_t*>(get_data(outputs[i]));
        int8_t* raw_s8 = reinterpret_cast<int8_t*>(get_data(outputs[i]));
        float* raw_f32 = reinterpret_cast<float*>(get_data(outputs[i]));

        printf("  -> First 8 bytes (HEX):  ");
        for (int j = 0; j < 8 && j < size; ++j) {
            printf("%02X ", raw_u8[j]);
        }
        printf("\n");

        printf("  -> First 8 values (INT8): ");
        for (int j = 0; j < 8 && j < size; ++j) {
            printf("%d ", raw_s8[j]);
        }
        printf("\n");

        printf("  -> First 2 values (F32):  ");
        for (int j = 0; j < 2 && (j*4) < size; ++j) {
            printf("%e ", raw_f32[j]);
        }
        printf("\n");
        
        if (size == exp_cls_s8) cls_s8 = reinterpret_cast<float*>(get_data(outputs[i]));
        else if (size == exp_cls_s16) cls_s16 = reinterpret_cast<float*>(get_data(outputs[i]));
        else if (size == exp_cls_s32) cls_s32 = reinterpret_cast<float*>(get_data(outputs[i]));
        else if (size == exp_box_s8) box_s8 = reinterpret_cast<float*>(get_data(outputs[i]));
        else if (size == exp_box_s16) box_s16 = reinterpret_cast<float*>(get_data(outputs[i]));
        else if (size == exp_box_s32) box_s32 = reinterpret_cast<float*>(get_data(outputs[i]));
    }
    
    if (!cls_s8 || !cls_s16 || !cls_s32 || !box_s8 || !box_s16 || !box_s32) {
        printf("[ERROR] Output tensor size mismatch!\n");
        printf("Expected total_size (elements): cls(%u, %u, %u), box(%u, %u, %u)\n", 
               exp_cls_s8, exp_cls_s16, exp_cls_s32, exp_box_s8, exp_box_s16, exp_box_s32);
        return;
    }

    PrintClsHeadStats("S8", cls_s8, feat_h_s8 * feat_w_s8);
    PrintClsHeadStats("S16", cls_s16, feat_h_s16 * feat_w_s16);
    PrintClsHeadStats("S32", cls_s32, feat_h_s32 * feat_w_s32);

    DecodeBranch(cls_s8, box_s8, feat_h_s8, feat_w_s8, 8, conf_threshold, &bboxes, &scores);
    DecodeBranch(cls_s16, box_s16, feat_h_s16, feat_w_s16, 16, conf_threshold, &bboxes, &scores);
    DecodeBranch(cls_s32, box_s32, feat_h_s32, feat_w_s32, 32, conf_threshold, &bboxes, &scores);

    Postprocess(&bboxes, &scores, result, &conf_threshold);
}

void EYEDETGRAY::Release() {
    release_tensor(inputs[0]);
    for (int i = 0; i < 6; ++i) {
        release_tensor(outputs[i]);
    }
    ReleaseAIPreprocessPipe(pipe_offline);
}
