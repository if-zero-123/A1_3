import sys
import re

file_path = r"D:\jichuang_docker\data\A1_SDK_SC035HGS\smartsens_sdk\smart_software\src\app_demo\face_detection\ssne_ai_demo\src\eye_det_gray.cpp"

with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

# Replace the dynamic mapping logic to use get_total_size instead of get_tensor_shape
old_mapping = r"""    float* cls_s8 = nullptr;
    float* cls_s16 = nullptr;
    float* cls_s32 = nullptr;
    float* box_s8 = nullptr;
    float* box_s16 = nullptr;
    float* box_s32 = nullptr;

    for (int i = 0; i < 6; ++i) {
        uint32_t n, c, h, w;
        get_tensor_shape(outputs[i], &n, &c, &h, &w);
        
        if (c == 1) { // cls heads
            if (w == feat_w_s8) cls_s8 = reinterpret_cast<float*>(get_data(outputs[i]));
            else if (w == feat_w_s16) cls_s16 = reinterpret_cast<float*>(get_data(outputs[i]));
            else if (w == feat_w_s32) cls_s32 = reinterpret_cast<float*>(get_data(outputs[i]));
        } else if (c == 64) { // box heads
            if (w == feat_w_s8) box_s8 = reinterpret_cast<float*>(get_data(outputs[i]));
            else if (w == feat_w_s16) box_s16 = reinterpret_cast<float*>(get_data(outputs[i]));
            else if (w == feat_w_s32) box_s32 = reinterpret_cast<float*>(get_data(outputs[i]));
        }
    }
    
    if (!cls_s8 || !cls_s16 || !cls_s32 || !box_s8 || !box_s16 || !box_s32) {
        printf("[ERROR] Output tensor shape mismatch! Ensure model outputs 3x cls (c=1) and 3x box (c=64)\n");
        return;
    }"""

new_mapping = r"""    float* cls_s8 = nullptr;
    float* cls_s16 = nullptr;
    float* cls_s32 = nullptr;
    float* box_s8 = nullptr;
    float* box_s16 = nullptr;
    float* box_s32 = nullptr;

    // Calculate expected sizes in bytes (float32 = 4 bytes)
    uint32_t exp_cls_s8 = feat_h_s8 * feat_w_s8 * 1 * 4;
    uint32_t exp_cls_s16 = feat_h_s16 * feat_w_s16 * 1 * 4;
    uint32_t exp_cls_s32 = feat_h_s32 * feat_w_s32 * 1 * 4;
    uint32_t exp_box_s8 = feat_h_s8 * feat_w_s8 * 64 * 4;
    uint32_t exp_box_s16 = feat_h_s16 * feat_w_s16 * 64 * 4;
    uint32_t exp_box_s32 = feat_h_s32 * feat_w_s32 * 64 * 4;

    for (int i = 0; i < 6; ++i) {
        uint32_t size = get_total_size(outputs[i]);
        
        if (size == exp_cls_s8) cls_s8 = reinterpret_cast<float*>(get_data(outputs[i]));
        else if (size == exp_cls_s16) cls_s16 = reinterpret_cast<float*>(get_data(outputs[i]));
        else if (size == exp_cls_s32) cls_s32 = reinterpret_cast<float*>(get_data(outputs[i]));
        else if (size == exp_box_s8) box_s8 = reinterpret_cast<float*>(get_data(outputs[i]));
        else if (size == exp_box_s16) box_s16 = reinterpret_cast<float*>(get_data(outputs[i]));
        else if (size == exp_box_s32) box_s32 = reinterpret_cast<float*>(get_data(outputs[i]));
    }
    
    if (!cls_s8 || !cls_s16 || !cls_s32 || !box_s8 || !box_s16 || !box_s32) {
        printf("[ERROR] Output tensor size mismatch! Check if the outputs match 6 YOLOv8 heads.\n");
        return;
    }"""

content = content.replace(old_mapping, new_mapping)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Updated mapping to use get_total_size.")
