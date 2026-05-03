/*
 * @Filename: pose_det_gray.cpp
 * @Description: 单通道 YOLOv8 手势/pose 检测实现文件，输出为3个分类头+3个bbox头
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <numeric>
#include <vector>

#include "../include/utils.hpp"

namespace {
constexpr bool kPoseVerboseTensorDebug = false;
constexpr int kPoseDebugPredictPrintCount = 5;
constexpr float kPoseClassMargin = 0.08f;
constexpr float kPoseCrossClassSuppressIoU = 0.60f;
constexpr float kPoseMaxBoxAreaRatio = 0.45f;
constexpr float kPoseMinAspectRatio = 0.30f;
constexpr float kPoseMaxAspectRatio = 2.80f;
constexpr float kPosePreferredAspectMin = 0.55f;
constexpr float kPosePreferredAspectMax = 1.85f;
constexpr float kPoseAreaRejectLowRatio = 0.0035f;
constexpr float kPoseAreaRejectHighRatio = 0.42f;
constexpr float kPoseAreaPreferredLowRatio = 0.015f;
constexpr float kPoseAreaPreferredHighRatio = 0.20f;
constexpr float kPoseEdgeMarginRatio = 0.08f;
constexpr float kPoseGeometryRejectThreshold = 0.12f;
constexpr float kPoseLowScoreGeometryGate = 0.36f;
constexpr int kPoseOkClassId = 1;
constexpr float kPoseOkMetricBoost = 0.08f;
constexpr float kPoseOkQualityBoost = 0.04f;
constexpr float kPoseOkScoreRelax = 0.04f;

struct PoseCandidate {
    std::array<float, 4> box;
    float score = 0.0f;
    int cls = -1;
    float quality = 0.0f;
    float metric = 0.0f;
};

float Sigmoid(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

float BoxWidth(const std::array<float, 4>& box) {
    return std::max(0.0f, box[2] - box[0]);
}

float BoxHeight(const std::array<float, 4>& box) {
    return std::max(0.0f, box[3] - box[1]);
}

float BoxArea(const std::array<float, 4>& box) {
    return BoxWidth(box) * BoxHeight(box);
}

float Clamp01(float v) {
    return std::max(0.0f, std::min(1.0f, v));
}

float CenterX(const std::array<float, 4>& box) {
    return 0.5f * (box[0] + box[2]);
}

float CenterY(const std::array<float, 4>& box) {
    return 0.5f * (box[1] + box[3]);
}

float RangeQuality(float value,
                   float reject_low,
                   float preferred_low,
                   float preferred_high,
                   float reject_high) {
    if (value <= reject_low || value >= reject_high) {
        return 0.0f;
    }
    if (value >= preferred_low && value <= preferred_high) {
        return 1.0f;
    }
    if (value < preferred_low) {
        return Clamp01((value - reject_low) / std::max(1e-6f, preferred_low - reject_low));
    }
    return Clamp01((reject_high - value) / std::max(1e-6f, reject_high - preferred_high));
}

float ComputePoseGeometryQuality(const std::array<float, 4>& box,
                                 int det_w,
                                 int det_h) {
    const float w = BoxWidth(box);
    const float h = BoxHeight(box);
    if (w < 1.0f || h < 1.0f) {
        return 0.0f;
    }

    const float aspect = w / h;
    const float area_ratio = BoxArea(box) / std::max(1.0f, static_cast<float>(det_w * det_h));
    const float cx = CenterX(box);
    const float cy = CenterY(box);
    const float nx = std::fabs(cx - det_w * 0.5f) / std::max(1.0f, det_w * 0.5f);
    const float ny = std::fabs(cy - det_h * 0.5f) / std::max(1.0f, det_h * 0.5f);
    const float center_quality = Clamp01(1.0f - (0.60f * nx + 0.40f * ny));

    const float edge_margin_x = det_w * kPoseEdgeMarginRatio;
    const float edge_margin_y = det_h * kPoseEdgeMarginRatio;
    const float edge_dx = std::min(box[0], std::max(0.0f, static_cast<float>(det_w) - box[2]));
    const float edge_dy = std::min(box[1], std::max(0.0f, static_cast<float>(det_h) - box[3]));
    const float edge_quality_x = Clamp01(edge_dx / std::max(1.0f, edge_margin_x));
    const float edge_quality_y = Clamp01(edge_dy / std::max(1.0f, edge_margin_y));
    const float edge_quality = 0.5f * edge_quality_x + 0.5f * edge_quality_y;

    const float area_quality = RangeQuality(area_ratio,
                                            kPoseAreaRejectLowRatio,
                                            kPoseAreaPreferredLowRatio,
                                            kPoseAreaPreferredHighRatio,
                                            kPoseAreaRejectHighRatio);
    const float aspect_quality = RangeQuality(aspect,
                                              kPoseMinAspectRatio,
                                              kPosePreferredAspectMin,
                                              kPosePreferredAspectMax,
                                              kPoseMaxAspectRatio);
    return 0.34f * area_quality +
           0.28f * aspect_quality +
           0.20f * center_quality +
           0.18f * edge_quality;
}

float IoU(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]);
    const float y2 = std::min(a[3], b[3]);
    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);
    const float inter = w * h;
    const float area_a = BoxWidth(a) * BoxHeight(a);
    const float area_b = BoxWidth(b) * BoxHeight(b);
    const float uni = area_a + area_b - inter;
    return uni <= 0.0f ? 0.0f : inter / uni;
}

float DFL(const float* tensor, int start_channel, int spatial, int idx) {
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

void PrintTensorBytePreview(const char* tag, ssne_tensor_t tensor, int max_bytes = 16) {
    if (!kPoseVerboseTensorDebug) {
        return;
    }
    uint32_t total_size = get_total_size(tensor);
    uint8_t* data = reinterpret_cast<uint8_t*>(get_data(tensor));
    printf("[DEBUG] %s total_size=%u preview=", tag, total_size);
    const int n = std::min<int>(max_bytes, static_cast<int>(total_size));
    for (int i = 0; i < n; ++i) {
        printf("%02X ", data[i]);
    }
    printf("\n");
}

struct WBFCluster {
    int cls = -1;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float total_score = 0.0f;
    float total_weight = 0.0f;
    float total_quality = 0.0f;
    float best_score = 0.0f;
    int count = 0;

    void Add(const PoseCandidate& det) {
        const float weight = std::max(1e-4f, det.metric);
        x1 += det.box[0] * weight;
        y1 += det.box[1] * weight;
        x2 += det.box[2] * weight;
        y2 += det.box[3] * weight;
        total_score += det.score;
        total_weight += weight;
        total_quality += det.quality;
        best_score = std::max(best_score, det.score);
        count += 1;
        cls = det.cls;
    }

    std::array<float, 4> GetBox() const {
        if (total_weight < 1e-6f) return {0.0f, 0.0f, 0.0f, 0.0f};
        const float inv = 1.0f / total_weight;
        return {x1 * inv, y1 * inv, x2 * inv, y2 * inv};
    }

    float GetScore() const {
        if (count <= 0) return 0.0f;
        const float avg = total_score / static_cast<float>(count);
        const float quality = total_quality / static_cast<float>(count);
        const float multi_bonus = std::min(0.10f, 0.03f * static_cast<float>(count - 1));
        const float quality_bonus = 0.05f * Clamp01((quality - 0.45f) / 0.55f);
        return std::min(1.0f, 0.72f * best_score + 0.28f * avg + multi_bonus + quality_bonus);
    }

    float GetQuality() const {
        if (count <= 0) {
            return 0.0f;
        }
        return total_quality / static_cast<float>(count);
    }
};

void WeightedBoxFusionByClass(const std::vector<PoseCandidate>& in,
                              float iou_thresh,
                              std::vector<PoseCandidate>* out) {
    std::map<int, std::vector<PoseCandidate>> grouped;
    for (const auto& det : in) {
        grouped[det.cls].push_back(det);
    }

    for (auto& kv : grouped) {
        auto& dets = kv.second;
        std::sort(dets.begin(), dets.end(), [](const PoseCandidate& lhs, const PoseCandidate& rhs) {
            return lhs.metric > rhs.metric;
        });

        std::vector<WBFCluster> clusters;
        for (const auto& det : dets) {
            bool merged = false;
            for (auto& cluster : clusters) {
                if (IoU(det.box, cluster.GetBox()) > iou_thresh) {
                    cluster.Add(det);
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                WBFCluster cluster;
                cluster.Add(det);
                clusters.push_back(cluster);
            }
        }

        for (const auto& cluster : clusters) {
            PoseCandidate fused;
            fused.box = cluster.GetBox();
            fused.score = cluster.GetScore();
            fused.cls = cluster.cls;
            fused.quality = cluster.GetQuality();
            fused.metric = fused.score + 0.10f * fused.quality;
            out->push_back(fused);
        }
    }
}

void SuppressOverlapsClassAgnostic(std::vector<PoseCandidate>* dets,
                                   float iou_thresh) {
    if (dets == nullptr || dets->empty()) {
        return;
    }

    std::sort(dets->begin(), dets->end(), [](const PoseCandidate& lhs, const PoseCandidate& rhs) {
        return lhs.metric > rhs.metric;
    });

    std::vector<PoseCandidate> kept;
    kept.reserve(dets->size());
    for (const auto& det : *dets) {
        bool suppressed = false;
        for (const auto& keep : kept) {
            if (IoU(det.box, keep.box) > iou_thresh) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(det);
        }
    }
    dets->swap(kept);
}
}

void POSEDETGRAY::Initialize(std::string& model_path, std::array<int, 2>* in_img_shape,
                             std::array<int, 2>* in_det_shape, int in_box_len, int in_num_classes) {
    nms_threshold = 0.35f;
    keep_top_k = 6;
    top_k = 80;
    min_box_size = 18.0f;
    img_shape = *in_img_shape;
    det_shape = *in_det_shape;
    box_len = in_box_len;
    num_classes = std::max(1, in_num_classes);
    debug_model_path = model_path;
    debug_input_dtype = SSNE_UINT8;
    debug_predict_count = 0;
    w_scale = static_cast<float>(img_shape[0]) / static_cast<float>(det_shape[0]);
    h_scale = static_cast<float>(img_shape[1]) / static_cast<float>(det_shape[1]);
    steps = {8, 16, 32};

    printf("[INFO] Pose postprocess: classes=%d WBF=%.2f min_box=%.1f top_k=%d keep=%d\n",
           num_classes, nms_threshold, min_box_size, top_k, keep_top_k);
    printf("[INFO] Pose model path: %s\n", debug_model_path.c_str());

    FILE* fp = fopen(debug_model_path.c_str(), "rb");
    if (fp != nullptr) {
        fseek(fp, 0, SEEK_END);
        long model_bytes = ftell(fp);
        fclose(fp);
        printf("[INFO] Pose model file opened successfully, size=%ld bytes\n", model_bytes);
    } else {
        printf("[ERROR] Pose model file open failed: %s\n", debug_model_path.c_str());
    }

    char* model_path_char = const_cast<char*>(model_path.c_str());
    model_id = ssne_loadmodel(model_path_char, SSNE_STATIC_ALLOC);
    printf("[INFO] Pose model loaded, model_id=%u\n", static_cast<unsigned int>(model_id));

    uint32_t det_width = static_cast<uint32_t>(det_shape[0]);
    uint32_t det_height = static_cast<uint32_t>(det_shape[1]);
    inputs[0] = create_tensor(det_width, det_height, SSNE_Y_8, SSNE_BUF_AI);
    printf("[INFO] Pose input tensor created: width=%u height=%u format=SSNE_Y_8\n",
           det_width, det_height);

    int dtype = SSNE_UINT8;
    const int dtype_ret = ssne_get_model_input_dtype(model_id, &dtype);
    if (dtype_ret == 0) {
        set_data_type(inputs[0], static_cast<uint8_t>(dtype));
        debug_input_dtype = dtype;
        printf("[INFO] Pose model input dtype: %d\n", dtype);
    } else {
        printf("[WARN] Pose model input dtype query failed, ret=%d\n", dtype_ret);
    }

    const int normalize_ret = SetNormalize(pipe_offline, model_id);
    if (normalize_ret != 0) {
        printf("[WARN] Pose model SetNormalize failed, ret: %d\n", normalize_ret);
    } else {
        printf("[INFO] Pose model SetNormalize success\n");
    }
}

void POSEDETGRAY::DecodeBranch(const float* cls_head, const float* box_head,
                               int feat_h, int feat_w, int stride,
                               float conf_threshold,
                               FaceDetectionResult* result) const {
    const int spatial = feat_h * feat_w;
    for (int y = 0; y < feat_h; ++y) {
        for (int x = 0; x < feat_w; ++x) {
            const int idx = y * feat_w + x;

            float best_score = 0.0f;
            int best_cls = -1;
            float second_score = 0.0f;
            for (int c = 0; c < num_classes; ++c) {
                const float score = Sigmoid(cls_head[c * spatial + idx]);
                if (score > best_score) {
                    second_score = best_score;
                    best_score = score;
                    best_cls = c;
                } else if (score > second_score) {
                    second_score = score;
                }
            }
            if (best_score < conf_threshold || best_cls < 0) {
                continue;
            }
            if ((best_score - second_score) < kPoseClassMargin && best_score < 0.85f) {
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

            result->boxes.push_back({x1, y1, x2, y2});
            result->scores.push_back(best_score);
            result->class_ids.push_back(best_cls);
        }
    }
}

void POSEDETGRAY::Postprocess(FaceDetectionResult* result, float* conf_threshold) {
    if (result->boxes.empty()) {
        return;
    }

    std::vector<PoseCandidate> candidates;
    candidates.reserve(result->boxes.size());
    const size_t n = std::min(result->boxes.size(),
                              std::min(result->scores.size(), result->class_ids.size()));
    const float max_box_area = static_cast<float>(det_shape[0] * det_shape[1]) * kPoseMaxBoxAreaRatio;
    for (size_t i = 0; i < n; ++i) {
        const auto& box = result->boxes[i];
        if (result->scores[i] < *conf_threshold) continue;
        if (BoxWidth(box) < min_box_size || BoxHeight(box) < min_box_size) continue;
        if (BoxArea(box) > max_box_area) continue;
        const float quality = ComputePoseGeometryQuality(box, det_shape[0], det_shape[1]);
        if (quality < kPoseGeometryRejectThreshold) continue;
        const bool is_ok = (result->class_ids[i] == kPoseOkClassId);
        const float effective_conf = is_ok ? std::max(0.0f, *conf_threshold - kPoseOkScoreRelax)
                                           : *conf_threshold;
        if (result->scores[i] < effective_conf) continue;
        const float assisted_score =
            result->scores[i] + 0.10f * quality +
            (is_ok ? (kPoseOkMetricBoost + kPoseOkQualityBoost * quality) : 0.0f);
        if (result->scores[i] < (effective_conf + 0.04f) && quality < kPoseLowScoreGeometryGate) {
            continue;
        }

        PoseCandidate det;
        det.box = box;
        det.score = result->scores[i];
        det.cls = result->class_ids[i];
        det.quality = quality;
        det.metric = assisted_score;
        candidates.push_back(det);
    }
    if (candidates.empty()) {
        result->Clear();
        return;
    }

    std::sort(candidates.begin(), candidates.end(), [](const PoseCandidate& lhs, const PoseCandidate& rhs) {
        return lhs.metric > rhs.metric;
    });
    if (static_cast<int>(candidates.size()) > top_k) {
        candidates.resize(top_k);
    }

    std::vector<PoseCandidate> fused;
    fused.reserve(candidates.size());
    WeightedBoxFusionByClass(candidates, nms_threshold, &fused);
    SuppressOverlapsClassAgnostic(&fused, kPoseCrossClassSuppressIoU);
    std::sort(fused.begin(), fused.end(), [](const PoseCandidate& lhs, const PoseCandidate& rhs) {
        return lhs.metric > rhs.metric;
    });
    if (static_cast<int>(fused.size()) > keep_top_k) {
        fused.resize(keep_top_k);
    }

    result->Clear();
    result->Reserve(static_cast<int>(fused.size()));
    for (const auto& det : fused) {
        std::array<float, 4> box = det.box;
        box[0] = std::max(0.0f, std::min(box[0] * w_scale, static_cast<float>(img_shape[0])));
        box[1] = std::max(0.0f, std::min(box[1] * h_scale, static_cast<float>(img_shape[1])));
        box[2] = std::max(0.0f, std::min(box[2] * w_scale, static_cast<float>(img_shape[0])));
        box[3] = std::max(0.0f, std::min(box[3] * h_scale, static_cast<float>(img_shape[1])));
        result->boxes.push_back(box);
        result->scores.push_back(det.score);
        result->class_ids.push_back(det.cls);
    }
}

void POSEDETGRAY::Predict(ssne_tensor_t* img_in, FaceDetectionResult* result, float conf_threshold) {
    result->Clear();
    debug_predict_count += 1;
    const bool print_debug = kPoseVerboseTensorDebug && (debug_predict_count <= kPoseDebugPredictPrintCount);
    if (print_debug) {
        printf("[DEBUG] Pose Predict #%d start, model_id=%u conf=%.3f input_dtype=%d\n",
               debug_predict_count, static_cast<unsigned int>(model_id), conf_threshold, debug_input_dtype);
        PrintTensorBytePreview("Pose raw input(img_in)", *img_in);
    }

    int ret = RunAiPreprocessPipe(pipe_offline, *img_in, inputs[0]);
    if (ret != 0) {
        printf("[ERROR] Pose RunAiPreprocessPipe failed, ret=%d\n", ret);
        return;
    }
    if (print_debug) {
        printf("[DEBUG] Pose preprocess ok\n");
        PrintTensorBytePreview("Pose preprocessed input", inputs[0]);
    }

    ret = ssne_inference(model_id, 1, inputs);
    if (ret != 0) {
        fprintf(stderr,
                "pose ssne inference fail! ret=%d model_id=%u path=%s predict_count=%d dtype=%d\n",
                ret,
                static_cast<unsigned int>(model_id),
                debug_model_path.c_str(),
                debug_predict_count,
                debug_input_dtype);
        return;
    }
    if (print_debug) {
        printf("[DEBUG] Pose inference ok, fetching 6 outputs\n");
    }

    ssne_getoutput(model_id, 6, outputs);

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

    uint32_t exp_cls_s8 = feat_h_s8 * feat_w_s8 * num_classes;
    uint32_t exp_cls_s16 = feat_h_s16 * feat_w_s16 * num_classes;
    uint32_t exp_cls_s32 = feat_h_s32 * feat_w_s32 * num_classes;
    uint32_t exp_box_s8 = feat_h_s8 * feat_w_s8 * 64;
    uint32_t exp_box_s16 = feat_h_s16 * feat_w_s16 * 64;
    uint32_t exp_box_s32 = feat_h_s32 * feat_w_s32 * 64;

    for (int i = 0; i < 6; ++i) {
        uint32_t size = get_total_size(outputs[i]);
        if (print_debug) {
            printf("[DEBUG] Pose output tensor %d total_size: %u elements\n", i, size);
            PrintTensorBytePreview("Pose output tensor", outputs[i], 12);
        }

        if (size == exp_cls_s8) cls_s8 = reinterpret_cast<float*>(get_data(outputs[i]));
        else if (size == exp_cls_s16) cls_s16 = reinterpret_cast<float*>(get_data(outputs[i]));
        else if (size == exp_cls_s32) cls_s32 = reinterpret_cast<float*>(get_data(outputs[i]));
        else if (size == exp_box_s8) box_s8 = reinterpret_cast<float*>(get_data(outputs[i]));
        else if (size == exp_box_s16) box_s16 = reinterpret_cast<float*>(get_data(outputs[i]));
        else if (size == exp_box_s32) box_s32 = reinterpret_cast<float*>(get_data(outputs[i]));
    }

    if (!cls_s8 || !cls_s16 || !cls_s32 || !box_s8 || !box_s16 || !box_s32) {
        printf("[ERROR] Pose output tensor size mismatch!\n");
        printf("Expected total_size (elements): cls(%u, %u, %u), box(%u, %u, %u)\n",
               exp_cls_s8, exp_cls_s16, exp_cls_s32, exp_box_s8, exp_box_s16, exp_box_s32);
        return;
    }

    DecodeBranch(cls_s8, box_s8, feat_h_s8, feat_w_s8, 8, conf_threshold, result);
    DecodeBranch(cls_s16, box_s16, feat_h_s16, feat_w_s16, 16, conf_threshold, result);
    DecodeBranch(cls_s32, box_s32, feat_h_s32, feat_w_s32, 32, conf_threshold, result);
    Postprocess(result, &conf_threshold);
}

void POSEDETGRAY::Release() {
    release_tensor(inputs[0]);
    for (int i = 0; i < 6; ++i) {
        release_tensor(outputs[i]);
    }
    ReleaseAIPreprocessPipe(pipe_offline);
}
