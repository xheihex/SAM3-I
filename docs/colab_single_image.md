# SAM3-I Colab 单图部署与本地验证

更新时间：2026-11-03。源码工作区：WSL TG 的 `/home/kmg/SAM3-I`。

已完成部署代码与不占用模型显存的本地验证。**用户于 2026-11-03 确认 Colab 单图推理部署成功。**具体 GPU 型号、运行精度、耗时和推理日志尚未记录；此前 WSL 检查仍仅覆盖 CPU 接口与 meta 结构，不能据此推断 FP16 稳定性或论文指标。

## 1. 方案与交付文件

| 文件 | 用途 |
|---|---|
| `notebooks/SAM3I_single_image_colab.ipynb` | 上传 Colab 后按顺序执行 |
| `dist/colab/sam3i-colab-source.zip` | 与本地检查一致的源码部署包，约 2 MB，不含权重和数据 |
| `scripts/colab_single_image.py` | 单图 GPU 推理、三种文字路径与结果导出 |
| `scripts/audit_single_image_checkpoint.py` | CPU/meta 权重名称与形状检查 |
| `scripts/validate_colab_single_image.py` | CPU 预处理、路由、合成输出和文件导出验证 |
| `scripts/build_colab_bundle.py` | 重建部署包与逐文件 SHA256 清单 |

项目能力说明在材料目录的 `SAM3-I_能力与输入输出及数据处理说明.md`。本方案支持图像＋语言提示的实例分割，输出掩码、框和匹配分数，提供概念、简单指令、复杂指令三条显式路径。第一轮建议使用英文提示。无需准备 COCO JSON、评估数据集或另行下载 SAM3 基础权重。

2026-11-02 验证时的本地 Git HEAD 为 `d96abbed9d2b75d52b3beca27d310057154e1ac1`；远端当次核对为 `5656d47bef4f48f9b1952b2a30cde9594b3f7d39`，当时本地 uv 环境提交尚未推送。采用源码 ZIP 可确保 Colab 使用已检查的本地配置。ZIP 内的 `SOURCE_MANIFEST.json` 用逐文件哈希标识实际快照，Git HEAD 本身不包含本次新增文件。

## 2. 对原作者代码的影响

没有修改 `sam3/`、原批量推理脚本、模型配置、`pyproject.toml` 或 `uv.lock`。新增适配脚本调用作者的：

- `build_sam3_image_model(..., inst_stage="3", load_from_HF=False)`；
- `Sam3Processor.set_image`、`set_text_prompt` 和模型 `forward_grounding`；
- 默认 1008×1008 图像预处理及概念/简单/复杂路径。

独立适配层有两处运行时调整：

1. 构建模型期间局部替换 checkpoint 读取函数，用 `torch.load(path, mmap=True, map_location="cpu", weights_only=False)` 读取本地文件。保留作者的权重匹配、加载和阶段逻辑；构建结束或抛错后恢复原函数。训练 checkpoint 大于纯模型权重，mmap 避免一次性复制其全部存储到主机内存。
2. 继承 `Sam3Processor`，仅将筛选后的掩码转到 CPU，以 FP32 逐实例插值恢复原图大小，不保留未使用的大幅概率图 `masks_logits`。分数、阈值、框坐标语义保持一致；FP32 合成输出已与原后处理精确对比。BF16/FP16 的边界像素可能与原 GPU 低精度插值不同。

保留全部超过阈值的实例，不额外做 NMS 或 top-1。逐实例插值控制临时内存；全部原尺寸 bool 掩码仍随实例数占用 CPU 内存。

## 3. Colab 操作顺序

1. 在 Colab 上传并打开 `notebooks/SAM3I_single_image_colab.ipynb`，把运行时改为 GPU。
2. 按 notebook 第 1 步上传 `sam3i-colab-source.zip`。核对打印的 ZIP SHA256 与本地打包输出一致。
3. 第 2 步安装 uv，并用锁文件创建独立的 Python 3.12 环境。项目通过子进程执行，Colab 原生内核负责 Drive、上传和展示。不要在原生内核里混装整套项目依赖。
4. 将作者 Stage 3 权重放自己的 Drive，例如 `MyDrive/SAM3-I/checkpoint.pt`。第 3 步修改 `DRIVE_CHECKPOINT`，挂载 Drive，再复制到 `/content/sam3i-checkpoints/checkpoint.pt`；mmap 使用该本地副本。
5. 第 4 步审查 checkpoint 结构。这个子进程单独设置 `CUDA_VISIBLE_DEVICES=""`，不影响随后推理进程使用 GPU。
6. 第 5 步上传一张图片。第 6 步设置英文提示、category、阈值和精度并运行。
7. 第 7 步查看/下载结果 ZIP，同时保存到 `MyDrive/SAM3-I/results/`。每次推理使用新的结果目录。

| category | text_type | 示例 |
|---|---:|---|
| `concept` | 1 | `shoe` |
| `simple` | 2 | `the red shoe in the top row` |
| `complex` | 4 | `the red item in the top row worn on feet while running` |

句子长度与目标数量不自动决定路径。完整 Stage 3 构建使用 `inst_stage="3"`，单次文字查询再用 `text_type=1/2/4` 选路。这样也避免作者 `inst_stage="1_2"` 时按 checkpoint 文件名判断阶段迁移、复制或重置适配器的行为。

切换提示后重新执行推理单元格即可。当前 notebook 每次都在新子进程中加载模型，退出后释放该进程的模型显存；这是便于首次部署验证的方式，交互延迟尚未优化。

## 4. 资源与精度

现有 checkpoint 文件约 9.2 GiB。作者实际加载拓扑的 FP32 参数为 7,280,061,656 字节，约 **6.78 GiB**，还要加 buffer、激活、CUDA 工作区等。不能用参数大小当作完整显存需求。

作者加载器还注册了一个冗余的 `backbone.text` 编码器，实际文字前向使用 `backbone.language_backbone`。本方案保留作者行为；没有删除该模块或改变模型权重精度。

- 优先在可获得的更大显存、更多主机内存的 GPU 运行时完成第一轮；A100 是可选目标，L4/T4 视实际分配情况试运行。
- `auto` 在原生 BF16 可用时选 BF16，否则选 FP16。使用 PyTorch 2.7.1 的 `is_bf16_supported(including_emulation=False)` 检测原生支持。
- autocast 降低计算精度，**模型参数仍保留 FP32**，不会把上述权重显存减半。FP16 数值与效果需要实机验证；非有限结果会明确报错。
- 若需要对比精度，可以设置 `fp32`，但其激活显存可能更高。
- 批量已经固定为 1，未启用视频、交互预测器或 `torch.compile`。
- GPU 掩码插值已移到 CPU，但低阈值和大量 4K 实例仍可能占用数 GiB 主机内存。标准 Colab 主机内存是否足够需要实测。
- Colab 中源码、环境、权重副本和临时结果位于 `/content`；持久权重和最终结果位于 Drive。运行时重置后需要重新执行安装与复制。

Colab 不保证 GPU 类型、资源可用性或运行时长，见 [Colab FAQ](https://research.google.com/colaboratory/faq.html)。Drive 挂载、上传下载参照 [Colab 官方 I/O 示例](https://colab.research.google.com/notebooks/io.ipynb)。环境方案依据 [uv Python 管理](https://docs.astral.sh/uv/guides/install-python/)和[锁文件同步](https://docs.astral.sh/uv/concepts/projects/sync/)；计算精度依据 [PyTorch AMP](https://docs.pytorch.org/docs/2.7/amp.html)。

## 5. 推理输出

每次新目录中包含：

- `overlay.png`：彩色掩码、框和分数；
- `mask_000.png` 等：各实例原图尺寸二值掩码，白色为目标；
- `union_mask.png`：所有实例的并集；
- `cutout.png`：以并集作 alpha 的透明 PNG；
- `result.json`：提示、路径、精度、阈值、原图尺寸、实例分数、原始像素框、面积和 COCO 压缩 RLE；
- notebook 最后另附 checkpoint 审查报告和源码清单。

框为 `[x0,y0,x1,y1]`，RLE 的尺寸为 `[H,W]`。JSON 保留作者原始框，展示时才裁剪到图像范围。空预测会输出空实例数组、全零并集和透明抠图，不等于证明图中不存在目标。

## 6. 本地 WSL 验证与重建

全部命令在 WSL TG 的源码根目录执行，使用根 uv 环境；输出显式放材料目录。默认 checkpoint 为用户指定的 `/home/kmg/models/sam3-i/checkpoint.pt`。

```bash
cd /home/kmg/SAM3-I
CHECK_DIR=/mnt/d/heihe/Documents/ChatGPT/sam3i试验区/.cache/colab-single-image-check

CUDA_VISIBLE_DEVICES= uv run --locked --no-dev python scripts/validate_colab_single_image.py \
  --image /home/kmg/SAM3-I/MVIMG_20260107_201842.jpg --output-dir "$CHECK_DIR"

CUDA_VISIBLE_DEVICES= uv run --locked --no-dev python scripts/audit_single_image_checkpoint.py \
  --checkpoint /home/kmg/models/sam3-i/checkpoint.pt --output "$CHECK_DIR/checkpoint-audit.json"

uv run --locked --no-dev python scripts/build_colab_bundle.py
```

当次检查结果：

| 检查 | 结果 |
|---|---|
| 11 项 CPU 检查 | 通过 |
| 实际 JPG | RGB，宽 4080、高 3060；处理后 `[1,3,1008,1008]` |
| 实际分词 | `[1,32]` |
| 概念、简单、复杂路径 | 显式路由 1/2/4，模型数值计算由合成桩替代 |
| 作者后处理对照 | FP32 合成输入，三路径及阈值 0/0.5/1，分数、框和掩码一致 |
| 导出与空预测 | RLE 解码与 PNG 一致；并集、alpha、叠加和空结果通过 |
| mmap 与作用域 | 训练 checkpoint / 原始 state_dict 的小样本读取、构建成功/异常后恢复通过 |
| 真实 checkpoint | 前向结构 2226/2226、作者加载拓扑 3385/3385 键形状匹配；缺失/冲突为 0 |
| checkpoint 额外键 | 18 个训练投影键，审查报告中列明 |
| CUDA | 验证前后均未初始化；CPU 验证还拦截了 CUDA 延迟初始化 |
| 部署包重建环境 | 独立 WSL 目录，从 uv 缓存按锁文件离线安装成功 |
| notebook | 代码单元格通过 Python 语法检查；未执行 Colab 交互单元格 |

证据保存于材料目录 `.cache/colab-single-image-check/`：`cpu-contract-check.json`、`checkpoint-audit.json`、`bundle-sync.log` 等。目录内 `synthetic-*` 为合成测试输出，不能作为鞋墙图片的模型预测。

meta 审查仅构建参数形状，关闭审查范围内的 CUDA 位置缓存，把少量配置标量放 CPU；所有 checkpoint 张量映射到 meta，不读取模型权重数值。这些审查专用替换不会进入真实推理路径。

2026-11-03，用户确认 Colab 单图推理部署成功。后续可补充具体 GPU、提示路径、运行精度、显存峰值、耗时和样例结果，再继续验证概念与复杂指令效果及论文评估指标。
