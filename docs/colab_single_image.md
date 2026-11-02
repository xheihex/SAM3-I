# SAM3-I Colab 单图部署与验证记录

更新时间：2026-11-03。唯一源码工作区：WSL TG 的 `/home/kmg/SAM3-I`。

已核对用户提供的 Colab 运行记录：单图推理成功，BF16 简单指令输出 1 个实例，记录的 PyTorch 峰值已分配显存为 7.6356 GiB。新版 notebook 从 GitHub 自动获取代码，权重继续从 Google Drive 加载；展示改为彩色半透明掩码、框、实例编号及下方的逐编号置信度。

## 1. GitHub 自动部署方案

仓库为 [xheihex/SAM3-I](https://github.com/xheihex/SAM3-I)。2026-11-03 核对时，远端 `main` 为 `35d4550346eab2a1613f15f9143c572580b097dd`。

| 文件 | 用途 |
|---|---|
| `notebooks/SAM3I_single_image_colab.ipynb` | 上传 Colab，自动获取 GitHub 源码并执行单图推理 |
| `scripts/colab_single_image.py` | 单图模型加载、三种提示路径与结果导出 |
| `scripts/audit_single_image_checkpoint.py` | CPU/meta 权重名称与形状审查 |
| `scripts/colab_notebook_helpers.py` | 获取源码、记录运行信息、日志及部署辅助函数 |
| `scripts/single_image_visualization.py` | 掩码、框、编号与置信度面板展示 |
| `scripts/validate_colab_single_image.py` | CPU 接口与合成输出验证 |
| `scripts/validate_colab_github.py` | GitHub 部署、日志、编号与自动保存验证 |
| `scripts/sync_colab_notebook.py` | 将部署和展示模块同步到 notebook 内嵌单元格 |
| `scripts/build_colab_bundle.py` | 保留为离线源码 ZIP 打包工具，常规 Colab 部署不需要使用 |

只需上传新版 notebook，不再上传源码 ZIP。notebook 内嵌部署辅助函数与展示模块，它们分别是上述两个新增脚本的镜像，因此能够配合已推送的旧版单图推理脚本运行；使用这份 notebook 不要求先将本次新增辅助脚本推送到远端。

源码下载使用 `https://github.com/xheihex/SAM3-I.git`。可选择 `main` 或明确的提交 SHA，获取后放到独立 checkout，并记录实际解析出的提交。使用 `main` 会随远端更新；需要重复同一次实验时，应填写上次记录的提交 SHA。Git 提交标识下载的模型推理源码，notebook 内嵌适配层另有内容记录，二者共同标识实际运行版本。

项目环境仍由 uv 和仓库锁文件管理，使用独立 Python 3.12 环境。Colab 原生内核仅负责 Drive、图片上传和显示；模型在独立子进程中执行。无需准备评估数据集、COCO JSON 或另行下载 SAM3 基础权重。

## 2. 对作者模型代码的影响

没有修改 `sam3/`、原批量推理脚本、模型配置、`pyproject.toml` 或 `uv.lock`。模型调用仍使用作者的：

- `build_sam3_image_model(..., inst_stage="3", load_from_HF=False)`；
- `Sam3Processor.set_image`、`set_text_prompt` 和模型 `forward_grounding`；
- 默认 1008×1008 图像预处理及概念、简单、复杂三条显式路径。

已有推理适配保持两项调整：

1. 仅在模型构建期间局部替换 checkpoint 读取函数，用 `torch.load(path, mmap=True, map_location="cpu", weights_only=False)` 读取本地权重。保留作者的权重匹配、加载和阶段逻辑，构建结束或抛错后恢复原函数。
2. 继承 `Sam3Processor`，将筛选后的掩码转到 CPU，以 FP32 逐实例插值恢复原图尺寸，不保留未使用的大幅 `masks_logits` 概率图。FP32 合成输出已与作者后处理对照；BF16/FP16 的边界像素可能与 GPU 低精度插值不同。

新的展示层只读取预测结果绘图，不改变匹配分数、阈值筛选或模型前向。保留全部超过阈值的实例，不另加 NMS 或 top-1。

## 3. Colab 操作顺序

1. 上传并打开新版 `SAM3I_single_image_colab.ipynb`，选择 GPU 运行时。
2. 运行源码部署单元格。默认从上述 GitHub 仓库获取代码，可选择分支或提交；核对打印的实际提交。
3. 运行环境准备单元格。uv 按 `uv.lock` 建立 Python 3.12 项目环境，记录 torch、实际 GPU、显存和原生 BF16 支持等环境信息。
4. 挂载 Google Drive。默认权重路径采用本次已成功运行的 `/content/drive/MyDrive/models/sam3-i/checkpoint.pt`，可按实际位置修改。
5. 将权重复制到 Colab 本地磁盘后使用 mmap。已有副本的源路径、大小及修改时间等元数据不变时可复用，避免每次重新复制约 9.13 GiB。更换权重后应确保副本与来源一致。
6. 执行 CPU/meta 权重审查。该子进程隐藏 GPU，检查名称与形状，不读取权重数值。
7. 上传一张图片，选择提示类型、文字、阈值和精度，运行推理。
8. 同一个推理单元格在展示完成后自动打包结果并保存到 Drive。默认目录为 `/content/drive/MyDrive/SAM3-I/results`；最后的下载单元格仅用于额外下载到浏览器。

每次推理使用新的结果目录。模型子进程的 stdout/stderr 流式显示并保存为日志；权重结构审查的完整输出写入日志，notebook 只显示摘要。成功运行的结果包包含环境 JSON、checkpoint 审查 JSON、实际源码提交与适配层记录。推理或绘图失败时，日志暂留 Colab 的部署日志目录；看到 Drive 保存路径后再结束运行时。

### 提示路径

| category | text_type | 示例 |
|---|---:|---|
| `concept` | 1 | `shoe` |
| `simple` | 2 | `the red shoe in the top row` |
| `complex` | 4 | `the red item in the top row worn on feet while running` |

路径由调用方显式选择，句子长度和目标数量不自动决定路径。完整阶段构建使用 `inst_stage="3"`，单次查询用 `text_type=1/2/4` 选路，避免作者 `inst_stage="1_2"` 时按 checkpoint 文件名触发阶段迁移、复制或重置适配器。

切换提示后重新执行推理单元格即可；当前每次在新子进程中加载模型，退出后释放该进程的模型显存。

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

## 5. 用户运行记录检查

检查对象为用户提供的 `SAM3I_single_image_colab.ipynb` 运行副本，原文件保持不变。

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

新版通过流式日志、显式环境记录、编号与置信度面板、推理后自动保存结果补足上述运行记录问题。

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

# 更新辅助模块后，将它们同步到可独立使用的 notebook。
uv run --locked --no-dev python scripts/sync_colab_notebook.py

# 6 项新版检查，并实际下载 GitHub main、从本地缓存重建锁定环境。
CUDA_VISIBLE_DEVICES= uv run --locked --no-dev python scripts/validate_colab_github.py \
  --output-dir /mnt/d/heihe/Documents/ChatGPT/sam3i试验区/.cache/colab-github-check --github-smoke

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

此次新增的 6 项 CPU 检查全部通过：notebook 内嵌模块与源码一致且没有执行输出、stdout/stderr 与失败日志保存、权重复用及来源变化、旧版结果格式的编号与掩码对应关系、非法编号/分数拒绝、执行 notebook 的实际推理单元格后自动归档到模拟 Drive（仅替换 GPU 模型计算）。原有 11 项 CPU 检查也再次通过，检查前后 CUDA 均未初始化。

另已实际从 GitHub 下载 main 的 `35d4550346eab2a1613f15f9143c572580b097dd`，在独立 WSL 目录从 uv 缓存重建锁定环境，并检查单图推理 CLI；临时源码及环境已清理。证据位于材料目录 `.cache/colab-github-check/github-notebook-check.json`、`github-smoke.json`、`github-smoke-logs/` 及 `legacy-contract/cpu-contract-check.json`。合成图含单实例/多实例、空结果与小图展示检查，不是新的真实分割结果。

新版 notebook 尚未在 Colab 重新执行完整 GPU 前向；此次本地验证覆盖 GitHub 获取、环境重建、展示和保存，不替代 GPU 数值验证。

历史证据位于材料目录 `.cache/colab-single-image-check/`。其中 `synthetic-*` 为合成测试输出，不是鞋墙图片的模型预测。meta 审查只构建参数形状，对审查范围内的 CUDA 缓存和少量配置标量作局部替换，checkpoint 张量映射到 meta，不读取权重数值；这些替换不进入真实推理路径。
