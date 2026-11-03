# SAM3-I Colab 单图部署与验证记录

更新时间：2026-11-03。唯一源码工作区：WSL TG 的 `/home/kmg/SAM3-I`。

新版 notebook 将模型一次加载到显存，由常驻子进程串行处理多次单图查询。换提示词、提示类别、阈值、精度或图片时复用同一模型；查询结束后继续保留模型，直到显式释放或运行时结束。源码继续从 GitHub 获取，权重从 Google Drive 复制到本地，结果展示和自动保存保持原有方式。

此前用户提供的 Colab 运行记录确认：BF16 简单指令单图推理成功，输出 1 个实例，PyTorch 峰值已分配显存为 7.6356 GiB。这是旧版每次加载模型的运行记录，不是本次常驻模型改造的 GPU 验证结果。

## 1. GitHub 自动部署方案

仓库为 [xheihex/SAM3-I](https://github.com/xheihex/SAM3-I)。2026-11-03 核对时，远端 `main` 为 `35d4550346eab2a1613f15f9143c572580b097dd`。

| 文件 | 用途 |
|---|---|
| `notebooks/SAM3I_single_image_colab.ipynb` | 上传 Colab，自动获取 GitHub 源码并执行单图推理 |
| `scripts/colab_single_image.py` | 原有单图模型加载、三种提示路径与结果导出，保留单次 CLI |
| `scripts/colab_inference_worker.py` | 常驻 GPU 模型，串行处理多次单图请求 |
| `scripts/audit_single_image_checkpoint.py` | CPU/meta 权重名称与形状审查 |
| `scripts/colab_notebook_helpers.py` | 获取源码、控制常驻进程、记录运行信息、日志与归档 |
| `scripts/single_image_visualization.py` | 掩码、框、编号与置信度面板展示 |
| `scripts/validate_colab_single_image.py` | CPU 接口与合成输出验证 |
| `scripts/validate_colab_github.py` | GitHub 部署、日志、编号与自动保存验证 |
| `scripts/validate_colab_resident.py` | 常驻模型复用、请求隔离及进程生命周期的 CPU 验证 |
| `scripts/sync_colab_notebook.py` | 将部署、展示和常驻进程代码同步到 notebook |
| `scripts/build_colab_bundle.py` | 保留为离线源码 ZIP 打包工具，常规 Colab 部署不需要使用 |

只需上传新版 notebook，不再上传源码 ZIP。notebook 内嵌部署辅助函数、展示模块和常驻进程源码；运行时将进程源码写入 Colab 本地部署目录，再导入下载仓库中的原有单图适配脚本。因此能够配合已推送的旧版单图推理脚本运行，不要求先将本次新增脚本推送到远端。修改这三部分源码后，需运行 `scripts/sync_colab_notebook.py` 更新 notebook；内嵌代码的哈希随结果保存。

源码下载使用 `https://github.com/xheihex/SAM3-I.git`。可选择 `main` 或明确的提交 SHA，获取后放到独立 checkout，并记录实际解析出的提交。使用 `main` 会随远端更新；需要重复同一次实验时，应填写上次记录的提交 SHA。Git 提交标识下载的模型推理源码，notebook 内嵌适配层另有内容记录，二者共同标识实际运行版本。

项目环境仍由 uv 和仓库锁文件管理，使用独立 Python 3.12 环境。Colab 原生内核负责 Drive、图片上传、进程控制和显示；uv 同步环境后直接启动该环境的 `.venv/bin/python`，模型在此常驻子进程中执行。这样释放模型时可直接终止实际持有 CUDA 上下文的 Python 进程。无需准备评估数据集、COCO JSON 或另行下载 SAM3 基础权重。

## 2. 对作者模型代码的影响

没有修改 `sam3/`、原批量推理脚本、模型配置、`pyproject.toml` 或 `uv.lock`。模型调用仍使用作者的：

- `build_sam3_image_model(..., inst_stage="3", load_from_HF=False)`；
- `Sam3Processor.set_image`、`set_text_prompt` 和模型 `forward_grounding`；
- 默认 1008×1008 图像预处理及概念、简单、复杂三条显式路径。

已有推理适配保持两项调整：

1. 仅在模型构建期间局部替换 checkpoint 读取函数，用 `torch.load(path, mmap=True, map_location="cpu", weights_only=False)` 读取本地权重。保留作者的权重匹配、加载和阶段逻辑，构建结束或抛错后恢复原函数。
2. 继承 `Sam3Processor`，将筛选后的掩码转到 CPU，以 FP32 逐实例插值恢复原图尺寸，不保留未使用的大幅 `masks_logits` 概率图。FP32 合成输出已与作者后处理对照；BF16/FP16 的边界像素可能与 GPU 低精度插值不同。

常驻进程仅长期持有同一个 eval 模型，逐次调用已有 `predict()` 和 `save_results()`。每次查询重新创建 processor、图像状态和文字状态，查询结束后释放这些状态；本次未缓存图像特征。修改提示类别或精度时仍调用原有三条推理路径，避免上一条查询的状态混入下一条查询。

展示层只读取预测结果绘图，不改变匹配分数、阈值筛选或模型前向。保留全部超过阈值的实例，不另加 NMS 或 top-1。

## 3. Colab 操作顺序

1. 上传并打开新版 `SAM3I_single_image_colab.ipynb`，选择 GPU 运行时。
2. 运行源码部署单元格。默认从上述 GitHub 仓库获取代码，可选择分支或提交；核对打印的实际提交。
3. 运行环境准备单元格。uv 按 `uv.lock` 建立 Python 3.12 项目环境，记录 torch、实际 GPU、显存和原生 BF16 支持等环境信息。
4. 挂载 Google Drive。默认权重路径采用本次已成功运行的 `/content/drive/MyDrive/models/sam3-i/checkpoint.pt`，可按实际位置修改。
5. 将权重复制到 Colab 本地磁盘后使用 mmap。已有副本的源路径、大小及修改时间等元数据不变时可复用，避免每次重新复制约 9.13 GiB。更换权重后应确保副本与来源一致。
6. 执行 CPU/meta 权重审查。该子进程隐藏 GPU，检查名称与形状，不读取权重数值。
7. 运行“加载模型到显存（运行一次）”单元格，等待模型加载完成，查看加载耗时与 `worker_id`。相同配置下重跑此单元格会复用已有模型。
8. 上传一张图片，选择提示类型、文字、阈值和精度，运行推理。
9. 同一个推理单元格在展示完成后自动打包结果并保存到 Drive。默认目录为 `/content/drive/MyDrive/results/SAM3-I`；下载单元格仅用于额外下载到浏览器。
10. 继续处理同一图片时，修改参数后重跑推理单元格；处理另一张图片时，重跑上传单元格和推理单元格。两种情况都复用显存中的模型。
11. 不再推理时，运行最后的释放模型单元格。下次推理前重新运行加载模型单元格。

每次推理使用新的结果目录。模型子进程的 stdout/stderr 流式显示并保存为日志；权重结构审查的完整输出写入日志，notebook 只显示摘要。成功运行的结果包包含环境 JSON、checkpoint 审查 JSON、实际源码提交与适配层记录。推理或绘图失败时，日志暂留 Colab 的部署日志目录；看到 Drive 保存路径后再结束运行时。

### 提示路径

| category | text_type | 示例 |
|---|---:|---|
| `concept` | 1 | `shoe` |
| `simple` | 2 | `the red shoe in the top row` |
| `complex` | 4 | `the red item in the top row worn on feet while running` |

路径由调用方显式选择，句子长度和目标数量不自动决定路径。完整阶段构建使用 `inst_stage="3"`，单次查询用 `text_type=1/2/4` 选路，避免作者 `inst_stage="1_2"` 时按 checkpoint 文件名触发阶段迁移、复制或重置适配器。

### 模型复用、耗时和释放

加载完成后，`worker_id` 和 `worker_pid` 标识常驻模型；多次推理中两者应保持相同，`query_index` 随查询递增。重跑加载单元格时，仓库、权重文件元数据、BPE 和适配代码一致且进程仍存活，才复用已有模型；需要替换时先停止旧进程，再加载新模型，避免两个模型同时驻留显存。重新部署源码或准备权重会停止旧进程，更换权重后需重新审查并加载模型。

`model_load_seconds` 记录一次性模型构建与权重加载耗时。`inference_seconds` 记录本次 `predict()` 耗时，包含图像预处理、模型前向和 CPU 掩码插值，已同步 CUDA；不包含模型加载、结果文件导出、绘图、ZIP 打包和 Drive 保存。`elapsed_seconds` 在常驻模式下与 `inference_seconds` 相同；旧版单次 CLI 的该字段包含加载耗时，比较结果时应留意 `timing_scope`。第一条查询还可能包含 CUDA 初始化后的首次计算开销，不能直接视为稳定速度。每条查询重新统计 PyTorch 峰值已分配显存，统计中包括常驻模型。

释放单元格调用 `MODEL_WORKER.stop()`，终止并回收持有模型的 Python 子进程，释放其 CUDA 上下文。中断查询或请求超时也会停止进程，避免后台继续运行；再次推理需重跑加载单元格。非法输入等可修正错误会返回日志，模型前向运行时错误或显存不足会使进程退出。Colab 运行时重置后，仍需从部署、环境和权重准备开始执行。

## 4. 编号与置信度展示

图中用一致的实例颜色显示半透明掩码、框及从 1 开始的编号。置信度在图下方逐编号显示，保留四位小数；图中编号、下方置信度和 JSON 中的 `instance_id` 一一对应。置信度是模型匹配分数，不是本次实验的准确率。

主要输出包括：

- `overlay.png`：原图尺寸的掩码图层、框及编号；
- `result_figure.png`：带编号图与下方置信度面板，适合直接查看或分享；
- `mask_000.png` 等：各实例原图尺寸二值掩码，白色为目标；
- `union_mask.png`：全部实例并集；
- `cutout.png`：以并集作为 alpha 的透明 PNG；
- `result.json`：提示、路径、精度、阈值、原图尺寸及每个实例的 `instance_id`、分数、框、面积、掩码文件名和 COCO 压缩 RLE；
- 环境信息、checkpoint 审查、源码版本记录与运行日志。

框为 `[x0,y0,x1,y1]`，RLE 尺寸为 `[H,W]`。JSON 保留原始框坐标，展示时才裁剪到图像范围。空预测仍保存原图展示、空实例列表、全零并集与透明抠图，并明确显示没有实例通过阈值；这不等于证明图中不存在目标。

## 5. 用户提供的旧版运行记录

检查对象为用户此前提供的 `SAM3I_single_image_colab.ipynb` 运行副本，原文件保持不变。此记录发生于常驻模型改造之前。

| 记录项 | 可确认的内容 |
|---|---|
| notebook 执行 | 7 个代码单元格中 6 个有执行计数，没有 error 类型输出或 traceback |
| 运行时 | metadata 标记 `gpuType=A100`；CUDA 环境检查子进程返回 0 |
| 权重 | 来自 `MyDrive/models/sam3-i/checkpoint.pt`，本地副本约 9.13 GiB |
| 权重审查 | 前向结构 2226/2226、作者加载拓扑 3385/3385，均通过；审查 CUDA 未初始化 |
| 图片 | RGB 鞋墙 JPG，宽 4080、高 3060 |
| 提示 | `the red shoe in the top row`，`simple`，阈值 0.6，BF16 |
| 推理结果 | 1 个实例，嵌入了结果展示图 |
| 显存 | PyTorch `max_memory_allocated` 为 7.635556697845459 GiB |
| 结果持久化 | 最后保存到 Drive/下载的单元格未执行 |

这份记录证明单图简单指令推理已经成功，但还存在记录缺口：

- 环境单元格只持久保存了 `CompletedProcess(returncode=0)`，没有实际 GPU 全名、torch 版本、空闲显存及 BF16 探测结果的 stdout；A100 信息来自 notebook metadata。
- 推理单元格只打印实例数和显存峰值，没有内嵌 `result.json`、实例分数、耗时或完整权重加载日志。
- 最后的保存单元格未执行，因此不能从该 notebook 确认结果已经保存到 Drive。
- PyTorch allocated 峰值不包含全部 CUDA 驱动、工作区和缓存开销，不能作为完整物理显存下限。
- 此记录只验证 `simple` / BF16 路径，没有验证概念、复杂指令或 FP16；没有 GT 对照和数据集评估，不能据此确认论文指标。

GitHub 部署与自动保存改造增加了流式日志、显式环境记录、编号与置信度面板及推理后自动保存。本次进一步增加模型加载和单次推理的独立计时，以及常驻进程和查询编号。

## 6. 资源与精度

作者实际加载拓扑的 FP32 参数为 7,280,061,656 字节，约 6.78 GiB，还需要 buffer、激活和 CUDA 工作区。作者加载器另外注册了 `backbone.text` 编码器，而实际文字前向使用 `backbone.language_backbone`；当前适配保留该行为。

`auto` 在原生 BF16 可用时选择 BF16，否则选 FP16。检测使用 `torch.cuda.is_bf16_supported(including_emulation=False)`。autocast 改变计算精度，模型参数仍为 FP32；BF16 的此次成功记录不能证明 FP16 稳定性。设置 `fp32` 可用于数值对照，但可能需要更多激活显存。

批量固定为 1，未启用视频、交互预测器或 `torch.compile`。逐实例 CPU 插值控制临时内存，但全部原尺寸 bool 掩码仍随实例数增长；大量 4K 实例可能占用数 GiB 主机内存。

Colab 中源码、环境和权重副本位于 `/content`，持久权重和自动保存的结果位于 Drive。运行时重置后需重新获取源码、安装环境与准备权重。可用资源依据 [Colab FAQ](https://research.google.com/colaboratory/faq.html)；Drive 上传下载参照 [Colab 官方 I/O 示例](https://colab.research.google.com/notebooks/io.ipynb)。独立项目环境依据 [uv Python 管理](https://docs.astral.sh/uv/guides/install-python/)和[锁文件同步](https://docs.astral.sh/uv/concepts/projects/sync/)，计算精度依据 [PyTorch AMP](https://docs.pytorch.org/docs/2.7/amp.html)。

## 7. 本地 WSL 验证与离线打包

所有命令在 WSL TG 源码根目录执行，使用根 uv 环境。真实权重默认仍为 `/home/kmg/models/sam3-i/checkpoint.pt`；验证输出显式放材料目录。

```bash
cd /home/kmg/SAM3-I
CHECK_DIR=/mnt/d/heihe/Documents/ChatGPT/sam3i试验区/.cache/colab-single-image-check

CUDA_VISIBLE_DEVICES= uv run --locked --no-dev python scripts/validate_colab_single_image.py \
  --image /home/kmg/SAM3-I/MVIMG_20260107_201842.jpg --output-dir "$CHECK_DIR"

CUDA_VISIBLE_DEVICES= uv run --locked --no-dev python scripts/audit_single_image_checkpoint.py \
  --checkpoint /home/kmg/models/sam3-i/checkpoint.pt --output "$CHECK_DIR/checkpoint-audit.json"

# 更新部署、展示或常驻进程源码后，同步到可独立使用的 notebook。
uv run --locked --no-dev python scripts/sync_colab_notebook.py

# GitHub 部署与 notebook 自动保存检查；--github-smoke 额外获取远端源码并重建环境。
CUDA_VISIBLE_DEVICES= uv run --locked --no-dev python scripts/validate_colab_github.py \
  --output-dir /mnt/d/heihe/Documents/ChatGPT/sam3i试验区/.cache/colab-github-check --github-smoke

# 常驻模型与进程控制检查，仅使用 CPU 桩和合成输出。
CUDA_VISIBLE_DEVICES= uv run --locked --no-dev python scripts/validate_colab_resident.py \
  --output-dir /mnt/d/heihe/Documents/ChatGPT/sam3i试验区/.cache/colab-resident-check

# 可选：仅在需要离线源码 ZIP 时运行。
uv run --locked --no-dev python scripts/build_colab_bundle.py
```

此前已完成的本地检查：

| 检查 | 结果 |
|---|---|
| 11 项 CPU 检查 | 通过 |
| 实际图片预处理 | RGB 4080×3060，转为 `[1,3,1008,1008]` |
| 实际分词 | `[1,32]` |
| 三种路径 | CPU 桩检查显式路由 1/2/4，未执行真实模型数值前向 |
| FP32 后处理对照 | 三条路径及阈值 0/0.5/1 的合成分数、框和掩码一致 |
| 导出与空结果 | RLE 解码、PNG、并集、alpha、叠加和空结果通过 |
| mmap 与作用域 | 训练 checkpoint / 原始 state_dict 小样本读取，成功/异常后恢复通过 |
| 真实 checkpoint | 前向 2226/2226、加载拓扑 3385/3385，缺失和形状冲突为 0 |
| 额外 checkpoint 键 | 18 个训练投影键，已在审查报告列出 |
| CUDA | 检查前后未初始化，CPU 验证拦截了 CUDA 延迟初始化 |
| 旧离线部署包 | 独立 WSL 目录按锁文件从缓存重建环境成功 |
| 旧 notebook | Python 语法检查通过 |

此前 GitHub 部署改造的 6 项 CPU 检查全部通过：notebook 内嵌模块与源码一致且没有执行输出、stdout/stderr 与失败日志保存、权重复用及来源变化、旧版结果格式的编号与掩码对应关系、非法编号/分数拒绝、执行 notebook 的实际推理单元格后自动归档到模拟 Drive（仅替换 GPU 模型计算）。原有 11 项 CPU 检查也再次通过，检查前后 CUDA 均未初始化。

另已实际从 GitHub 下载 main 的 `35d4550346eab2a1613f15f9143c572580b097dd`，在独立 WSL 目录从 uv 缓存重建锁定环境，并检查单图推理 CLI；临时源码及环境已清理。证据位于材料目录 `.cache/colab-github-check/github-notebook-check.json`、`github-smoke.json`、`github-smoke-logs/` 及 `legacy-contract/cpu-contract-check.json`。合成图含单实例/多实例、空结果与小图展示检查，不是新的真实分割结果。

本次常驻模型改造已通过 8 项模型复用与进程控制 CPU 检查、6 项 notebook 集成检查和原有 11 项 CPU 检查，共 25 项。实际推理单元格已在一个模拟常驻 worker 上连续执行两次，各自产生独立 Drive 归档；真实子进程 IPC、非法请求恢复、超时、进程退出、配置变化重载及请求张量释放均有覆盖，检查前后 CUDA 未初始化。证据位于材料目录 `.cache/colab-resident-check/resident-worker-check.json`、`notebook/github-notebook-check.json` 及 `legacy-contract/cpu-contract-check.json`。

常驻模型版 notebook 尚未在 Colab 执行完整 GPU 前向和连续查询测速。本地 CPU 检查用于验证模型加载次数、请求隔离、进程控制、展示和保存，不替代 GPU 数值验证；此前 GitHub 获取与环境重建的证据也不证明本次常驻模式已完成 GPU 验证。

历史证据位于材料目录 `.cache/colab-single-image-check/`。其中 `synthetic-*` 为合成测试输出，不是鞋墙图片的模型预测。meta 审查只构建参数形状，对审查范围内的 CUDA 缓存和少量配置标量作局部替换，checkpoint 张量映射到 meta，不读取权重数值；这些替换不进入真实推理路径。
