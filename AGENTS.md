# SAM3-I 项目约定

以下约定由用户明确指定，适用于本项目的所有聊天和后续任务，无论聊天从 WSL 源码目录还是 Windows 材料目录进入，除非用户另行更改。

## 项目目标

- 复现 SAM3-I 模型。
- 验证模型的准确率及论文报告的评估结果。
- 后续将合适的模型组件或方法集成到用户自己的项目中。

## 工作环境与文件存放

- 唯一源码工作区是 WSL 发行版 `TG` 中的 `/home/kmg/SAM3-I`。
- Windows 访问地址为 `\\wsl.localhost\TG\home\kmg\SAM3-I`，它指向同一份 WSL 源码，不是第二份源码。
- 所有项目操作均通过 WSL 执行，包括代码修改、依赖安装、环境配置、训练、推理、评估及材料处理；使用 WSL 的 Linux 工具链和运行环境。
- 从 Windows 启动终端命令时，使用 `wsl.exe -d TG --cd /home/kmg/SAM3-I --exec <Linux 命令>`；不要因聊天的初始工作目录在 Windows 而改用 Windows 的 Python、Conda 或其他运行环境。
- WSL 用于保存源码、脚本、构建产物、可执行文件及运行所需的依赖和环境；源码配套的配置、项目说明和本文件随源码保留。
- `D:\heihe\Documents\ChatGPT\sam3i试验区` 是 Windows 材料工作区，用于保存论文、笔记、参考资料、数据集、模型权重、实验日志、评估结果等非代码材料；不在其中创建源码副本或运行环境。
- 在 WSL 中通过 `/mnt/d/heihe/Documents/ChatGPT/sam3i试验区` 访问 Windows 材料目录。需要读写非代码材料时，将输入、输出及相关缓存路径显式指向此目录下的相应位置，计算过程仍在 WSL 中执行。
- 本约定适用于后续工作；不要仅因更新约定而批量移动或删除现有文件。
- 开始后续任务前，先在 WSL 源码目录检查适用的项目说明及当前状态，再开展用户要求的工作。

## 项目记忆（2026-10-30）

- 已在源码根目录建立 uv 工作区：`pyproject.toml`、`uv.lock`、`.python-version`；统一使用根目录 `.venv`，`sam3/` 为本地 editable 成员。后续通过 `uv sync --locked`、`uv add/remove`、`uv run` 管理依赖及运行。
- 已验证的环境：Python 3.12、PyTorch 2.7.1+cu126、torchvision 0.22.1+cu126、NumPy 1.26.4；PyTorch 显式使用 CUDA 12.6 官方索引，具体版本由锁文件记录。
- `run.sh` 已接入 uv；已修复从仓库根目录导入 `sam3` 的打包问题。默认安装图像推理、评估、训练及开发依赖；Decord 改为视频分支按需导入，可选视频/笔记本组仍有上游 wheel 标签错误，详见 README。
- 已通过依赖兼容检查、CUDA 运算及 torchvision GPU NMS、图像数据加载和推理/评估/训练脚本帮助入口验证。尚未在本次任务中下载模型权重、运行完整模型或验证论文评估指标。
- 用户指定的 SAM3-I 权重文件为 `/home/kmg/models/sam3-i/checkpoint.pt`；后续推理和评估默认使用此路径。

## 项目记忆（2026-11-03）

- 已新增 Colab 单图推理 notebook、独立推理适配脚本、checkpoint 审查、CPU 验证及源码打包脚本，部署说明见 `docs/colab_single_image.md`；未修改原作者模型源码。
- 本地 WSL 已通过 11 项 CPU 检查和真实 checkpoint 的 meta 名称/形状审查，验证过程中 CUDA 未初始化。
- 已检查用户 2026-11-03 提供的 Colab 运行记录：metadata 为 A100，simple / BF16 / 阈值 0.6，输出 1 个实例，PyTorch 峰值已分配显存约 7.6356 GiB；最后保存单元格未执行，实例分数、耗时和完整子进程日志未持久保留。成功推理不代表已复现论文评估指标。
- notebook 已改为从 GitHub 自动获取代码，默认 `https://github.com/xheihex/SAM3-I.git` 的 main，记录实际提交号；权重仍由 Drive 的 `/content/drive/MyDrive/models/sam3-i/checkpoint.pt` 复制到 Colab 本地磁盘加载。
- notebook 内嵌部署及展示辅助函数，兼容远端已有的单图推理脚本；图上显示掩码、框和从 1 开始的编号，图下显示对应置信度，成功运行同单元格自动保存结果和日志到 Drive。
- 新增 6 项 GitHub/展示/自动保存 CPU 检查及原有 11 项检查均通过，实际下载 GitHub main 并重建锁定环境、检查推理 CLI 成功；新版尚未在 Colab 重新运行 GPU 前向。修改辅助模块后需运行 `scripts/sync_colab_notebook.py` 同步内嵌代码。
