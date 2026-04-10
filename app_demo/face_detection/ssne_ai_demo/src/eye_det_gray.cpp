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
    nms_threshold = 0.30f;
    keep_top_k = 4;
    top_k = 20;
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
            const float l = box_head[idx];
            const float t = box_head[spatial + idx];
            const float r = box_head[2 * spatial + idx];
            const float b = box_head[3 * spatial + idx];

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
        if (static_cast<int>(keep.size()) >= keep_top_k) {
            break;
        }
    }

    result->Reserve(static_cast<int>(keep.size()));
    for (int idx : keep) {
        std::array<float, 4> box = boxes->at(idx);
        box[0] = std::max(0.0f, std::min(box[0] * w_scale, static_cast<float>(img_shape[0])));
        box[1] = std::max(0.0f, std::min(box[1] * h_scale, static_cast<float>(img_shape[1])));
        box[2] = std::max(0.0f, std::min(box[2] * w_scale, static_cast<float>(img_shape[0])));
        box[3] = std::max(0.0f, std::min(box[3] * h_scale, static_cast<float>(img_shape[1])));
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

    float* cls_s8 = reinterpret_cast<float*>(get_data(outputs[0]));
    float* cls_s16 = reinterpret_cast<float*>(get_data(outputs[1]));
    float* cls_s32 = reinterpret_cast<float*>(get_data(outputs[2]));
    float* box_s8 = reinterpret_cast<float*>(get_data(outputs[3]));
    float* box_s16 = reinterpret_cast<float*>(get_data(outputs[4]));
    float* box_s32 = reinterpret_cast<float*>(get_data(outputs[5]));

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
