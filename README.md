# WeChat OCR

基于气泡颜色分割 + PaddleOCR 的微信 PC 端聊天记录识别导出工具。支持 GUI 调参与命令行两种用法。

## 功能

- 框选聊天区域截图，或加载本地图片
- 按气泡颜色区分说话人（对方 / 自己）
- 采色器自定义颜色，适配不同主题与显示器
- 未采样时使用默认 HSV，可直接识别
- 支持滚动多屏采集与导出（txt / json）
- 可选输出调试掩码、气泡框、裁剪图

## 项目结构

```text
wechatOCR/
├── main.py           # 入口：无参启动 GUI，带参走 CLI
├── gui_app.py        # GUI 界面与交互
├── ocr_core.py       # OCR、颜色掩码、窗口与导出核心逻辑
├── requirements.txt
└── .gitignore
```

## 环境要求

- Windows
- Python 3.10+（推荐 3.11）
- 微信 PC 客户端已打开

## 安装

```bash
# 建议使用虚拟环境
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

首次运行会下载 PaddleOCR 模型，可能稍慢。若想跳过联网检查，可设置：

```bash
set PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
```

有 NVIDIA GPU 时可自行替换为对应 CUDA 版 `paddlepaddle`。

## 快速开始（GUI）

```bash
python main.py
# 或
python main.py --gui
```

推荐流程：

1. 填写「微信窗口标题」（默认 `微信`）
2. 点击「框选截图」，程序会尝试激活微信并进入框选
3. （可选）点击「选取对方色 / 自己色」，在预览图上取样
4. 点击「预览掩码/气泡框」确认框是否正确
5. 设置导出路径、是否保存调试输出
6. 点击「执行 OCR 并导出」

说明：

- 未采样时使用默认颜色阈值（对方默认 HSV：`120, 2, 240`）
- Esc 取消框选后，状态栏会显示「已取消框选」
- 框选截图时会先隐藏 GUI，再截图，避免窗口自身被截进画面

## 命令行用法（CLI）

```bash
# 手动框选当前屏
python main.py --region manual --rounds 1 --out out.txt

# 自动推算聊天区域，并保存调试信息
python main.py --region auto --rounds 1 --debug-dir .\debug_run_1 --keep-image .\debug_run_1\source.png

# 滚动采集多屏
python main.py --region manual --rounds 5 --pause 0.8 --out chat.json
```

### 常用参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--gui` | 启动 GUI | 无 |
| `--title` | 微信窗口标题关键字 | `微信` |
| `--region` | `manual` 框选 / `auto` 自动推算 | `manual` |
| `--rounds` | 滚动采集次数，`1` 为当前屏 | `1` |
| `--pause` | 每屏滚动后等待秒数 | `0.8` |
| `--out` | 导出路径（`.txt` / `.json`） | 按时间自动生成 |
| `--keep-image` | 保存截图到指定路径 | 空 |
| `--debug-dir` | 调试输出目录 | 空 |

## 输出说明

### 文本导出示例

```text
# 微信聊天OCR导出记录
# 生成时间(UTC): ...
# 采集屏数: 1

[对方]: 你好
[自己]: 在的
```

### 调试目录（开启后）

常见文件：

- `00_source.png`：原始截图
- `01_white_mask.png`：对方颜色掩码
- `02_green_mask.png`：自己颜色掩码
- `03_bubbles.png`：气泡框可视化
- `debug_log.json`：每个气泡的检测/OCR/过滤状态
- `crop_*.png`：气泡裁剪图

## 常见问题

### 1. 找不到微信窗口

确认微信已打开，并检查窗口标题关键字。测试版可尝试：

```bash
python main.py --title "微信测试版"
```

GUI 中也可直接修改「微信窗口标题」。

### 2. 对方消息识别不到

优先在 GUI 中对对方气泡采样，再调节 H/S/V 容差。  
也可查看 `01_white_mask.png` 与 `03_bubbles.png` 判断是颜色阈值问题还是气泡过滤问题。

### 3. 截图发白 / 被窗口覆盖

请使用最新代码重新启动。框选截图时会先隐藏 GUI，再截图。

### 4. 首次启动很慢

PaddleOCR 首次会下载并加载模型，属于正常现象。

## 不建议提交到 Git 的内容

本地环境与调试产物请保持忽略（已在 `.gitignore`）：

- `venv/`
- `__pycache__/`
- `.idea/`
- `debug_*`、`debug_log.json`
- `wechat_export_*.txt` / `wechat_export_*.json`

## License

仅供个人学习与自用，请遵守微信及相关服务条款，勿用于未授权场景。
