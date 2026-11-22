# EchoCard

**本地优先的个人全天候记忆采集与 AI 整理工作台**（Windows 独立桌面应用）。

DJI Mic 2 自动吸取 · 手机照片双通道 · 本地语音流水线（转写 / 说话人聚类 / 音乐检测）· AI 投喂包导出与总结回流归档。

## 功能总览（已全部落地）

- **采集（M1）**
  - DJI Mic 2 发射器 USB 热插拔自动吸取：四重认亲（`.echocard_source` 标记 ∨ VID 0x2CA3/0x10D6 ∨ USBSTOR 产品串 ∨ 序列号记忆）→ 单文件「拷贝 → 大小+SHA-256 校验 → 落库 → 删单个源文件」原子状态机，**校验不过绝不删除**，断连/零字节/非目标 U 盘全保护
  - 手机扫码传图（局域网二维码页）+ 本地目录监控（Syncthing/网盘）双通道；`POST /api/v1/ingest/audio|photo` 标准接口可直接接入自研 ESP32-S3 硬件
  - 照片自动按 EXIF 归档 + 生成面向大模型的压缩图（长边 1024 / q82 / `HH-MM-SS_序号.jpg`）
- **本地语音引擎（M2，全程离线）**
  - ffmpeg 转码 16kHz 单声道 + 温和响度归一化（loudnorm，适配 DJI 32-bit float 低电平）
  - Silero VAD v5 切分 → **SenseVoice-Small int8** 转写（中/英/日/韩/粤 + 语言/情绪/事件标签），PyPI 轮子 CPU int8 即可流畅运行；CUDA 需 GPU 编译版 sherpa-onnx（配置 `pipeline_provider=cuda`，未装时内部自动回退）
  - **3D-Speaker CAM++** 声纹 Embedding → 当日无监督聚类（`说话人 1/2/…`）→ 与长期声纹库软匹配（≥0.75 人名 / 0.55~0.75 疑似-XXX）
  - **改名即学声纹**：界面一键改名自动抽取优质切片更新多原型声纹库，越用越准；支持合并被拆散的聚类
  - 音乐检测：SenseVoice 事件标签 + YAMNet（onnxruntime）非语音区间兜底 → 带挂钟起止的音乐卡片
  - **听歌识曲**：ShazamIO（免 Key，走代理，滑动窗口重采，结果落库缓存）+ 按任意时段查歌
  - 瘦身：转写完成后 WAV → Opus（约 15MB/小时）并删除原始 WAV；星标切片永久保留
- **投喂与回流（M3）**
  - 单日/日历多日合并导出：`EchoCard_日期_投喂包.zip` + 同级解压文件夹（`投喂AI.md` + `photos/`，不嵌 Base64）
  - 内置可编辑中文 Prompt：反偏见纠错声明 / 照片隐含意义推理 / 半结构化输出与打包回流协议
  - 三通道回流（拖 zip / 拖 md / 粘贴文本）容错解析 YAML Frontmatter → 标签/待办/长期事实/歌曲入库
  - SQLite **FTS5 全文检索**（中文双字分词）+ 可勾选待办
  - 归档后按配置/手动清理未 ⭐ 星标压缩音频（永久保留星标原声 + 文字稿 + 压缩照片 + 总结）
  - 可选轻量 API 一键总结（约 70 行 OpenAI 兼容多模态客户端，智谱 GLM 占位；留空 Key 完全隐藏）
- **桌面体验（M4）**
  - PyWebView 独立窗口（无浏览器地址栏感）+ pystray 托盘常驻（关闭窗口缩进托盘）+ win11toast 通知
  - 深色/浅色一键切换的中文 UI：月历状态点（灰=处理中 / 蓝=待投喂 / 绿=已归档）+ 多日多选合并导出 + 全局搜索 + 三 Tab（AI 复盘 / 时间轴与说话人校准 / 流水线日志）+ 设置页
  - 注册表 Run 键开机自启（当前用户，无需管理员）

## 快速开始

```powershell
# 1) 环境要求：Windows 11 + Python 3.12 + ffmpeg（winget install Gyan.FFmpeg）
cd D:\Projects\EchoCard
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt

# 2) 下载本地模型（约 300MB，ModelScope 优先 / hf-mirror / GitHub 兜底）
.venv\Scripts\python scripts\setup_models.py

# 3) 运行测试（模拟 U 盘安全状态机等全套用例）
.venv\Scripts\pytest

# 4) 启动
.venv\Scripts\python -m echocard            # 桌面模式：独立窗口 + 托盘 + Toast
.venv\Scripts\python -m echocard --dev      # 开发/验收：浏览器访问 http://127.0.0.1:8765
```

首次使用：插入 DJI Mic 2 发射器 → 自动认亲导入 → 若日期 ≥ 设置里的「历史转写分界线」（默认 `2026-11-22`）则自动转写/聚类/压缩，早于分界线的仅归档保留原 WAV（可在界面手动转写）。

## 数据与隐私红线

1. 所有个人数据收敛于数据根目录（默认 `D:\EchoCardData`：`audio/ photos/ db/ exports/ models/ logs/`），**绝不进入本仓库**（`.gitignore` 整目录 + 后缀双重拦截）
2. 设备清空开关**默认关闭**；开启后也只会在「拷贝 + 大小/SHA-256 校验 + 落库」全部成功后删除设备上的**单个**源文件，绝不格式化整盘、绝不触碰隐藏/系统文件
3. 未认亲的普通 U 盘 / 相机卡绝不被触碰
4. 反偏见：转写片段只增不删，仅附加客观特征标签（`bgm`/`low_volume`…），禁止主观定性
5. 代理地址与 API Key 仅存 `%APPDATA%\EchoCard\config.json`（仓库外）

## 架构一览

```
echocard/
├─ core/        配置(隐私不入库) · SQLite(WAL+FTS5) · 事件总线 · 日志 · CJK 检索
├─ ingest/      USB 四重认亲 · DJI 原子导入状态机 · 照片管道(EXIF/压缩/HEIC) · 局域网 API+二维码
├─ audio/       ffmpeg 转码/响度归一/Opus · VAD+SenseVoice · 聚类软匹配 · 声纹库(改名即学) · 音乐检测 · ShazamIO 识曲
├─ pipeline.py  日级流水线编排（历史豁免 + 倒序自动补全 + Opus 瘦身 + 声纹 npz）
├─ export/      投喂包(zip+目录+Prompt) · 三通道回流(容错YAML) · 星标清理
├─ llm_client.py 轻量 OpenAI 兼容多模态客户端（~70 行）
├─ desktop/     FastAPI 服务 · PyWebView 窗口 · pystray 托盘 · Toast · 注册表自启
├─ webui/       中文单页 UI（月历/三 Tab/说话人校准/查歌/设置，深浅色）
└─ tests/       70 用例：模拟U盘安全状态机 · 音频算法 · 流水线规则 · 导出回流检索 · 桌面端点
```

## 里程碑

- ✅ M1 骨架与安全采集
- ✅ M2 本地语音、零门槛声纹与音乐检测引擎
- ✅ M3 投喂导出、回流解析与轻量 API 总结
- ✅ M4 独立桌面 UI、托盘与打磨

## 许可证

MIT
