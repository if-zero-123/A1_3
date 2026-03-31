# A1 SmartSens AI Demo Source

这是 `smart_software/src` 子仓库的入口文档（GitHub 默认会显示本文件）。

## 项目内容

当前仓库主要包含：

- `app_demo/`：AI 演示应用源码（当前重点是 `ssne_ai_demo`）
- `linux-5.15.24/`：内核源码目录（已在 `.gitignore` 中排除）

## 重点演示应用

- 双目人脸检测 Demo：
  - 路径：`app_demo/face_detection/ssne_ai_demo/`
  - 详细文档：[`app_demo/face_detection/ssne_ai_demo/README.md`](app_demo/face_detection/ssne_ai_demo/README.md)

## 构建与运行（在上层 SDK 仓库执行）

请在上层 `smartsens_sdk` 目录执行：

```bash
bash scripts/build_release_sdk.sh
bash scripts/build_app.sh
```

## 默认启动说明

系统上电后会通过启动脚本进入 `app_demo`，并执行：

```bash
./ssne_ai_demo
```

即默认启动当前 AI 演示程序。
