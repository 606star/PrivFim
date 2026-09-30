# 整理版验证记录

验证日期：2026-09-30。使用独立创建的 Python 3.12.3 环境，
通过 `pip install -e '.[dev,data,crypto]'` 安装，不依赖历史工程的虚拟环境。
具体 Python 包版本保存在根目录 `requirements-lock.txt`。

## 已完成的验证

| 检查 | 结果 |
|---|---|
| 核心算法、工作流、密码学正确性测试 | 257 项通过，包含已构建的原生 TFHE 适配器 |
| 11 类实验完整参数轴 | Toy、seed=2026、m=32，共 211 个方法/配置运行完成 |
| 全局评价 | 上述 211 个结果均为 `evaluation_scope=global_exact`，重新计算完整合法空间中的 Top-k |
| 默认配置入口 | `privfim configured --config config/default.json` 完成，Toy、m=2048、四种主方法 |
| 真实数据完整默认配置 | DefaultCredit 全部 30,000 行，m=2048、k=15、K=4、ε=1、seed=2026，四种主方法完成 |
| 官方数据准备 | CensusIncomeKDD、MiniBooNE、PokerHand、DefaultCredit、LetterRecognition 下载/缓存读取与整数转换成功 |
| 历史数据一致性 | 上述五个数据集及 Bank 的整数矩阵与历史实验输入逐元素一致 |
| 真实密码学运行 | Toy 显式取 4 行、K=2、k=3，两种原生加密后端均完成，每个支持数与明文核验一致 |
| 构建 | Python 源码分发包和 wheel 构建成功。完整工作流推荐从 Git 源码安装 |
| 源码 ZIP 迁移 | 解压到独立临时目录，通过文件校验，离线准备 Toy/Bank、运行八种方法和独立实验 7 入口，生成全局指标 CSV 与 PDF |
| 迁移后的测试 | 256 项通过，1 项原生 TFHE 测试因源码包不带编译产物而按预期跳过 |

密码学验证只检查小样本的真实执行和正确性，不代表全量数据的运行时间。
普通 11 类实验的上述验证是流程检查，不是重新执行所有数据集的正式实验。
这些本地产物位于 `results/`，不进入源码包或 Git。

## 可重复执行的检查

```bash
pytest -q
privfim prepare --datasets Toy
privfim suite --datasets Toy --seeds 2026 --m 32 --output results/check_all11
privfim configured --config config/default.json
python scripts/setup_crypto.py --jobs 2
privfim crypto --datasets Toy --clients 2 --k 3 --max-rows 4 --output results/check_crypto
```

没有编译 TFHE 时，相应的原生测试会显示 skip，其余测试仍然执行。
编译后再运行即可覆盖原生后端。源码包内的 `RELEASE_MANIFEST.json` 列出每个文件的 SHA-256，
不包含原始数据下载、虚拟环境、密钥、旧实验结果或第三方仓库。
