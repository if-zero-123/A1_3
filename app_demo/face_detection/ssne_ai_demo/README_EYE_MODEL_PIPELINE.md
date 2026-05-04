# A1 眼动 / 手势 demo 说明

这份说明只讲当前这套 `ssne_ai_demo` 实际在做什么，方便后面继续改代码或者联调上位机。

适用目录：

- `smart_software/src/app_demo/face_detection/ssne_ai_demo/`

主要文件：

- 主流程：`demo_face.cpp`
- 图像输入：`src/pipeline_image.cpp`
- 眼睛检测：`src/eye_det_gray.cpp`
- 人脸检测：`src/scrfd_gray.cpp`
- 手势检测：`src/pose_det_gray.cpp`
- OSD：`src/utils.cpp`
- OSD 设备：`src/osd-device.cpp`
- 公共结构：`include/common.hpp`
- 运行脚本：`scripts/run.sh`

---

## 1. 当前功能概览

程序现在有三条识别链路：

1. 人脸检测
2. 眼睛检测 + 跟踪稳定
3. 手势检测

另外还多了一条串口遥测链路：

- 通过 UART 发眼动位置和手势识别结果给上位机

运行时是双线程：

- 主线程负责取图、ISP debug load、把最新帧塞进队列
- 推理线程负责人脸 / 眼睛 / 手势推理、OSD 绘制、串口发包

队列长度固定是 `1`，旧帧会直接丢掉，目的是优先保实时性。

---

## 2. 图像输入

图像输入在 `src/pipeline_image.cpp`。

当前配置：

- 图像格式：`SSNE_Y_8`
- 输出尺寸：`640 x 480`
- 接口：`GetDualImageData(...)`

也就是说，后面的眼睛、人脸、手势三路模型，拿到的都是双路灰度图。

主线程每轮做的事情基本是：

1. `processor.GetDualImage(&img_sensor[0], &img_sensor[1])`
2. 把两路图拷到 `output_sensor`
3. `start_isp_debug_load()`
4. 把当前帧塞进推理队列

---

## 3. 模型与输入尺寸

在 `demo_face.cpp` 里当前用到这几个模型：

- 眼睛模型：`/app_demo/app_assets/models/eye.m1model`
- 手势模型：`/app_demo/app_assets/models/pose.m1model`
- 人脸模型：`/app_demo/app_assets/models/face_640x480.m1model`

输入尺寸都按 `640 x 480` 初始化。

### 相关类

- 眼睛：`EYEDETGRAY`
- 手势：`POSEDETGRAY`
- 人脸：`SCRFDGRAY`

统一输出结构体：

- `FaceDetectionResult`
  - `boxes`
  - `scores`
  - `class_ids`
  - `landmarks`（人脸路径可用）

---

## 4. 推理与后处理

### 4.1 人脸

人脸使用 `SCRFDGRAY`。

人脸这一路的后处理可以拆成两段看：

#### 单帧后处理（`src/scrfd_gray.cpp`）

1. 根据 anchor 解码出检测框
2. 按置信度阈值过滤低分框
3. 做 NMS 去重
4. 把检测框从模型输入尺度恢复到原图尺度

这里的输出是一个普通的人脸框列表，结构还是 `FaceDetectionResult`。

#### 时序后处理（`demo_face.cpp`）

主流程里没有直接把每一帧人脸框都拿来用，而是只取“当前主人脸 ROI”：

1. 从这一帧人脸结果里选最高分框
2. 记成 `last_face_roi`
3. 后续几帧如果没检出，也暂时保留这个 ROI
4. 保留超时后，再认为人脸丢失

这样做的目的不是让人脸框更稳，而是给眼睛路径提供一个比较稳定的约束区域。

当前策略不是每帧跑，而是自适应降频：

- 丢失 ROI 或刚进入：`kFaceInferIntervalFast = 2`
- ROI 稳定后：`kFaceInferIntervalStable = 6`

作用：

- 有脸时尽量省一点算力
- 丢脸时尽快恢复 ROI

### 4.2 眼睛

眼睛检测在 `src/eye_det_gray.cpp`。

当前后处理特征：

- 6 个输出头
- DFL 解码
- WBF 融合
- 小框过滤
- 眼睛配对筛选

可以分成两层看：

#### 单帧后处理（`src/eye_det_gray.cpp`）

1. 三个尺度头分别做 DFL 解码
2. 按分数收集候选框
3. 做 WBF 融合，减少同一只眼多框抖动
4. 过滤小框
5. 做双眼配对筛选，只保留更像“一对眼睛”的候选

这一步结束以后，得到的是单帧意义下相对干净的眼睛检测框。

检测后不是直接拿来画，而是在 `demo_face.cpp` 里继续做：

- 人脸 ROI 约束
- CPU 瞳孔暗斑精定位
- 卡尔曼跟踪
- 显示层平滑

#### 时序后处理（`demo_face.cpp`）

这一层才是眼动观感的关键，主要做了下面几件事：

1. **人脸 ROI 约束**
   - 先用人脸 ROI 把明显不在眼区的候选框筛掉
   - 这样能减少误检点跑到脸外面

2. **CPU 瞳孔暗斑精定位**
   - 在眼框内部再找一遍更暗的区域
   - 把检测框中心往更接近瞳孔的位置拉
   - 如果暗斑质量不够或者位移太离谱，就放弃这一步

3. **左右眼匹配**
   - 不是简单按分数，而是按 IoU、中心距离、面积比例和左右半区去匹配
   - 这一步主要是解决左右眼串位和跳眼问题

4. **卡尔曼跟踪**
   - 对 `cx / cy / w / h` 分别做滤波
   - 小抖动压掉，真实运动尽量跟上

5. **显示层平滑**
   - 画到屏幕前再做一层轻量平滑
   - 目的是把肉眼最容易看到的 1px 级别抖动再压一遍

6. **稳态隔帧检测**
   - 眼睛稳定时，不一定每帧都重新跑模型
   - 中间帧直接走跟踪预测，换一点帧率

当前为了提帧，眼睛也不是永远每帧跑：

- 稳定时：`kEyeInferIntervalStable = 2`
- 如果运动速度起来了，再恢复每帧检测

### 4.3 手势

手势检测在 `src/pose_det_gray.cpp`。

当前有这些特点：

- 三分类：`up / ok / down`
- 单帧后处理里带几何质量评分
- 时序跟踪里有候选确认、切类门槛、类别保持
- `ok` 类别有额外偏置

同样可以分成两层：

#### 单帧后处理（`src/pose_det_gray.cpp`）

1. 三个尺度头分别解码
2. 每个候选先做类别置信选择
3. 过滤掉太小、太大、比例异常、边缘质量差的框
4. 做按类别的 WBF
5. 再做一次跨类别重叠抑制
6. 按质量分和分数排序，保留最终候选

这里的重点是：手势这一路不是只看原始分类分数，还会结合几何质量一起决定是否保留。

#### 时序后处理（`demo_face.cpp`）

手势的时序状态机比脸和眼都复杂，主要做了：

1. **候选确认**
   - 新目标不会单帧直接上屏
   - 需要连续确认，避免误触发

2. **跟踪保持**
   - 已锁定目标后，短时间丢检也不会马上清掉
   - 框位置做平滑，类别做保持

3. **切类门槛**
   - 从一个手势切到另一个手势，需要更高分或者连续多帧确认
   - 这样能减少 `up / ok / down` 来回跳

4. **`ok` 偏置**
   - `ok` 类别在选择和切换上做了额外偏置
   - 这是为了让当前演示更容易稳定识别 `ok`

5. **稳态降频**
   - 手势稳定后，pose 模型不会每帧都跑
   - 这样能给帧率留一点空间

串口日志当前只保留 `ok` 的识别打印。

---

## 5. 眼动稳定策略

眼动稳定逻辑主要在 `demo_face.cpp` 的 `EyeTrack` 和 `GetStableEyeBoxes(...)`。

现在实际是多层叠加：

1. 眼睛检测框筛选
2. 人脸 ROI 约束
3. CPU 瞳孔精定位
4. 左右眼匹配
5. 卡尔曼跟踪
6. 输出层平滑

### 左右眼匹配

不是简单按分数，而是结合：

- IoU
- 中心距离
- 面积比例
- 左右半区偏置

左眼这边额外给了一点稳定偏置，主要是为了减少串眼和跳框。

### CPU 瞳孔精定位

当前是从眼框内部找暗斑中心。

主要限制：

- ROI 太小直接跳过
- 低置信度暗斑不采纳
- 超过最大位移不采纳
- 最后跟原始框中心做混合

### 卡尔曼参数

当前版本偏“跟手优先”：

- `kKalmanQPos = 3.2`
- `kKalmanQVel = 0.9`
- `kKalmanRCalm = 3.2`
- `kKalmanRActive = 0.22`
- `kResidualThresh = 1.8`

如果后面要继续调眼动，通常就从这里和 `kEyeOutput...` 这一组开始。

---

## 6. OSD 绘制

OSD 绘制在 `src/utils.cpp`，底层设备在 `src/osd-device.cpp`。

当前分层大致是：

- 脸框 / 手势框：画空心框
- 眼点：画实心块

眼点没有画真正的圆，而是用一个小的 `TYPE_SOLID` 实心框当作代理点。

这么做的原因很实际：

- OSD quadrangle 引擎有数量限制
- 用很多小块拼圆容易 `ret=-1`
- 实心小块稳定得多

### 当前闪烁控制

`demo_face.cpp` 里有几层抑制：

- `kRedrawMinDelta = 3.2`
- `kOsdMinRedrawFrameGap = 3`
- `kClearAfterMissFrames = 5`

也就是说，不是每一帧都刷，只在变化够明显或者该清层时才刷。

---

## 7. UART 遥测

这是当前版本和早期版本差异最大的一块。

### 7.1 工程接入

需要的东西已经接到工程里了：

- 头文件：`uart_api.h`
- 链接库：`libuart.so`
- 运行脚本会尝试：
  - `insmod /lib/modules/$(uname -r)/extra/uart_kmod.ko`

### 7.2 初始化

`main()` 里会在 `ssne_initial()` 后做：

- `uart_init()`
- `uart_set_baudrate(..., 115200)`
- `uart_set_parity(..., UART_PARITY_NONE)`

当前默认：

- `kEnableUartTelemetry = true`
- `kUartBaudrate = 115200`

### 7.3 发送频率

不是每帧都发，而是：

- `kUartTelemetryInterval = 2`

也就是隔帧发，避免串口本身拖慢主流程。

### 7.4 发包长度限制

考虑了 UART FIFO 只有 32 字节的限制。

当前发送时会按：

- `kUartSendMaxChunk = 32`

自动分包发送。

### 7.5 当前协议

帧格式：

```text
AA 55 | version | msg_type | payload_len(LE) | payload | crc16(LE)
```

固定字段：

- `magic0 = 0xAA`
- `magic1 = 0x55`
- `version = 0x01`
- `msg_type = 0x01`

当前 payload 是 `EyeGestureTelemetry`：

1. `frame_id` `uint32 LE`
2. `timestamp_ms` `uint32 LE`
3. `left_valid` `uint8`
4. `right_valid` `uint8`
5. `gesture_valid` `uint8`
6. `gesture_cls` `uint8`
7. `left_cx_q8` `uint16 LE`
8. `left_cy_q8` `uint16 LE`
9. `right_cx_q8` `uint16 LE`
10. `right_cy_q8` `uint16 LE`
11. `gesture_score_q10` `uint16 LE`

换算规则：

- 眼动坐标：`q8 / 8.0`
- 手势分数：`q10 / 1024.0`

---

## 8. 上位机对应关系

当前这个串口协议已经对应了：

- `D:\jichuangsai\vision_a1`

那边是一个 `Node + 本地网页` 的可视化小工具。

如果你后面改了这里的协议字段顺序、长度或者缩放系数，上位机的 `shared/protocol.js` 也要一起改。

---

## 9. 当前实时策略

当前为了平衡识别效果和帧率，策略大致是：

- 主线程只保最新帧
- 人脸稳态低频
- 眼睛稳态隔帧
- 手势稳定后降频
- OSD 不连续重刷
- 串口隔帧发包

从代码角度看，瓶颈现在主要还是：

1. 双路取图
2. `copy_double_tensor_buffer`
3. `start_isp_debug_load`
4. 眼睛检测 + 瞳孔精定位

OSD 现在已经不是最大的开销来源。

---

## 10. 运行方式

### 板端运行

在 `scripts/run.sh` 里会做：

1. 尝试加载 UART 驱动
2. 启动 `ssne_ai_demo`

所以正常板端启动还是走原来的脚本。

### 编译注意

当前工程已经依赖：

- `libssne.so`
- `libcmabuffer.so`
- `libosd.so`
- `libuart.so`
- `libsszlog.so`
- `libzlog.so`
- `libemb.so`

如果后面有人裁库或者拷工程，一定别漏了 `libuart.so`。

---

## 11. 后面常改的地方

如果只是要继续调效果，通常会改下面这些：

### 调眼动

看 `demo_face.cpp` 里这些：

- `kKalmanQPos`
- `kKalmanQVel`
- `kKalmanRCalm`
- `kKalmanRActive`
- `kResidualThresh`
- `kEyeOutputDeadzonePx`
- `kEyeOutputMotionScalePx`
- `kEyeOutputSmoothAlphaCalm`
- `kEyeOutputSmoothAlphaActive`

### 调手势

看这些：

- `kPoseDisplayScoreThreshold`
- `kPoseDisplayScoreThresholdTracked`
- `kPoseDisplayScoreThresholdSwitch`
- `kPoseOkMetricBias`
- `kPoseOkThresholdRelax`

### 调串口

看这些：

- `kEnableUartTelemetry`
- `kUartBaudrate`
- `kUartTelemetryInterval`

---
