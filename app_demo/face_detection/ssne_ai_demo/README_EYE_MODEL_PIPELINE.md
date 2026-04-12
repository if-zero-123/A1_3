# A1 眼睛模型部署与运行全流程说明（含输入/接口/输出/后处理/OSD）

本文档基于当前工程代码整理，目标是把你这套「脸+眼」链路从部署到显示完整讲清楚。

适用目录：

- `smart_software/src/app_demo/face_detection/ssne_ai_demo/`

核心源码：

- 主流程：`demo_face.cpp`
- 图像输入：`src/pipeline_image.cpp`
- 眼睛模型：`src/eye_det_gray.cpp`
- 人脸模型：`src/scrfd_gray.cpp`
- OSD 可视化：`src/utils.cpp`
- 数据结构定义：`include/common.hpp`
- OSD 设备实现：`src/osd-device.cpp`, `include/osd-device.hpp`

---

## 1. 你现在这套系统在做什么（总览）

运行时是双线程结构：

1. **主线程（采图 + ISP显示）**
   - 从在线 pipeline 拉取双路图像（`GetDualImageData`）
   - 把图像送 ISP debug load
   - 把最新一帧放入推理队列（队列长度=1，旧帧会被丢弃）

2. **推理线程（脸+眼）**
   - 低频做人脸（每 `kFaceInferInterval=3` 帧）
   - 每帧做眼睛推理
   - 对眼睛检测结果做卡尔曼稳定
   - 转换成固定半径“眼点框”
   - OSD 绘制：脸框 + 眼点（强制 circle 模式）

---

## 2. 模型部署到程序里的入口与接口

### 2.1 模型路径与输入尺寸

在 `demo_face.cpp`：

- 眼睛模型路径：`/app_demo/app_assets/models/eye.m1model`
- 人脸模型路径：`/app_demo/app_assets/models/face_640x480.m1model`
- 两个模型输入尺寸都配置为 `640x480`

### 2.2 类接口（来自 `include/common.hpp`）

#### 眼睛模型接口：`EYEDETGRAY`

- 初始化：
  - `Initialize(std::string& model_path, std::array<int,2>* in_img_shape, std::array<int,2>* in_det_shape, int in_box_len)`
- 推理：
  - `Predict(ssne_tensor_t* img_in, FaceDetectionResult* result, float conf_threshold=0.25f)`
- 释放：
  - `Release()`

#### 人脸模型接口：`SCRFDGRAY`

- 初始化：
  - `Initialize(...)`
- 推理：
  - `Predict(...)`
- 释放：
  - `Release()`

#### 通用结果结构：`FaceDetectionResult`

- `boxes: vector<array<float,4>>`（每个框 `[x1,y1,x2,y2]`）
- `scores: vector<float>`（置信度）
- `landmarks`（人脸关键点可选）

---

## 3. 输入数据是什么？从哪来？是什么格式？

### 3.1 图像来源

`src/pipeline_image.cpp`：

- `OnlineSetOutputImage(kPipeline0, SSNE_Y_8, 640, 480)`
- `GetDualImageData(...)` 获取双路图（灰度）

所以输入给模型前的源数据是：

- **格式**：`SSNE_Y_8`（8-bit 单通道灰度）
- **尺寸**：640x480
- **容器类型**：`ssne_tensor_t`

### 3.2 推理前预处理接口

眼睛/人脸模型都走：

- `RunAiPreprocessPipe(pipe_offline, *img_in, inputs[0])`

这是 SDK 的离线前处理流水线，负责把在线图转成模型输入张量（尺寸/数据类型/归一化等按模型配置执行）。

---

## 4. 推理完成后获得什么数据？有哪些输出？

## 4.1 眼睛模型输出（`eye_det_gray.cpp`）

调用：

- `ssne_getoutput(model_id, 6, outputs)`

即：**6 个输出头**（YOLOv8 风格三尺度，分类3个 + 回归3个）。

当前代码按 `get_total_size()` 动态识别每个输出头归属：

- 分类头元素数：
  - S8: `80*60*1 = 4800`
  - S16: `40*30*1 = 1200`
  - S32: `20*15*1 = 300`
- 回归头元素数（DFL，64通道）：
  - S8: `80*60*64 = 307200`
  - S16: `40*30*64 = 76800`
  - S32: `20*15*64 = 19200`

> 关键注意：在此 SDK 上，`get_total_size()` 返回的是**元素个数**，不是字节数。

## 4.2 人脸模型输出（`scrfd_gray.cpp`）

也是 `ssne_getoutput(..., 6, outputs)`，按固定顺序读取：

- `outputs[0..2]` 分数
- `outputs[3..5]` 框

---

## 5. 推理后处理怎么做的？

## 5.1 眼睛后处理（`eye_det_gray.cpp`）

### 第一步：Decode（YOLOv8 + DFL）

在 `DecodeBranch`：

- 分类得分：`score = sigmoid(cls_head[idx])`
- 回归使用 DFL：每个边界（l/t/r/b）从 16-bin 分布做 softmax 加权求期望
- 由网格中心 `(x+0.5, y+0.5)*stride` + `l/t/r/b` 解码出 `x1,y1,x2,y2`

### 第二步：筛选 + NMS

- 置信度阈值过滤
- 按分数排序
- IoU NMS（`nms_threshold=0.25`）
- 小框过滤（`min_box_size`）
- 可选双眼配对筛选（当前 `eye_pair_only=false`）

### 第三步：尺度恢复 + 几何微调

- 由模型输入尺度映射回原图尺度（`w_scale/h_scale`）
- 再做 `ShrinkBoxAroundCenter`（宽高收缩 + 中心上移）

最后输出到 `FaceDetectionResult`：

- `result->boxes`
- `result->scores`

## 5.2 稳定跟踪（`demo_face.cpp`）

`GetStableEyeBoxes` 使用 2 组卡尔曼（左右眼）对中心和宽高做平滑：

- 状态：`x, v`（位置+速度）
- 对象：`cx, cy, w, h`
- 参数（当前版本）：
  - `kKalmanQPos=16`
  - `kKalmanQVel=4`
  - `kKalmanR=4`

并带短时丢检保持：

- `kEyeHoldFrames=1`
- `kClearAfterMissFrames=2`

---

## 6. OSD 绘制依据是什么？怎么画出来？

## 6.1 绘制输入数据来源

推理线程每帧得到：

- 脸框（可选）`last_face_roi`
- 稳定眼框 `stable_boxes`

然后将稳定眼框转换为**固定半径眼点框**（`BuildEyeDotBox`）：

- 中心 = 稳定框中心
- 半径 = `kEyeDotFixedRadius`（当前12像素）
- 得到 `eye_dot_boxes`

## 6.2 绘制接口

### 脸框

- `g_visualizer->Draw(face_draw)`
- 在 `utils.cpp` 里，使用空心矩形 `TYPE_HOLLOW`

### 眼点（当前强制 circle）

- `g_visualizer->DrawCircles(eye_draw)`

`DrawCircles` 当前实现是：

- **每只眼一个 `TYPE_SOLID` 实心四边形**（不是多点拼圆）
- 这样是为规避 OSD `ret=-1`（对象过多/约束超限）问题

## 6.3 为什么之前“圆块方案”失败？

从 OSD 头文件可知（`osd_lib_api.h`）：

- OSD 是 quadrangle 引擎
- 每层每行凸四边形数量有约束

之前用很多小块拼“圆环”，每帧会提交大量小四边形，容易触发：

- `osd_add_quad_rangle_layer ret = -1`

所以当前改成每眼 1 个实心块，是更稳的落地方案。

---

## 7. 当前实时策略（帧率/延迟）

### 7.1 降延迟

- 队列长度 `MAX_QUEUE_SIZE=1`
- 队列满时丢弃旧帧，永远保留最新帧

### 7.2 分时推理

- 人脸每 3 帧一次（`kFaceInferInterval=3`）
- 眼睛每帧推理

### 7.3 调试日志开关

`eye_det_gray.cpp`:

- `kEyeVerboseTensorDebug=false`

关闭后可大幅减轻串口/控制台 IO 对帧率影响。

---

## 8. 部署模型到板端的标准步骤（工程侧）

> 注：以下是工程接入流程，模型转换命令以你本地 A1-AI-Tool 实际版本为准。

1. 训练并导出 ONNX（灰度单通道眼睛模型）
2. 使用 A1 工具链转换为 `eye.m1model`
3. 拷贝到板端：
   - `/app_demo/app_assets/models/eye.m1model`
4. 编译 demo：
   - 重新编译 `ssne_ai_demo`
5. 运行 demo，观察：
   - 眼点是否稳定显示
   - 串口是否存在 `ret=-1`

---

## 9. 你最关心的“数据流一句话版”

`Sensor灰度帧(ssne_tensor_t)`
→ `RunAiPreprocessPipe`
→ `ssne_inference`
→ `ssne_getoutput(6头)`
→ `DFL解码 + 置信度过滤 + NMS + 尺度恢复`
→ `卡尔曼稳定`
→ `固定半径眼点框`
→ `OSD Draw(脸框) + DrawCircles(眼点)`

---

## 10. 建议的现场调参顺序（避免反复试错）

1. **先稳显示**：确认没有 `ret=-1`
2. **调眼点大小**：`kEyeDotFixedRadius`（推荐 9~12）
3. **调跟随灵敏度**：`kKalmanQPos/QVel/R`
4. **调丢检观感**：`kEyeHoldFrames` 与 `kClearAfterMissFrames`
5. **再调检测阈值**：`kEyeInferConfThreshold` / `kEyeDisplayScoreThreshold`

---

## 11. 本文档对应当前代码状态说明

本文档描述的是当前分支下已改动后的实现（包括：

- 眼模型 `get_total_size` 按元素匹配
- 固定半径眼点
- 强制 circle 模式
- 低延迟队列策略
- OSD circle 走实心块代理方案）

如果后续你改了参数或切回 box 模式，请同步更新本 README 的参数表。
