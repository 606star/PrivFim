# PrivFim

两轮纵向联邦频繁项集挖掘的可复现实验实现。数据行表示已对齐的用户，属性按列分给客户端。
第一轮发布加噪单项计数和 noisy N，筛选候选项集。第二轮发布正向 DP-FM Alpha，
服务端通过 MAP 估计交集支持数，输出 Top-k。

完整流程：**准备整数数据 → 纵向划分 → 运行协议 → 独立全局真值复核 → 指标汇总 → PDF 绘图**。
迁移后不需要原工程、旧结果目录或论文目录。默认使用 CPU，不要求 GPU。

## 安装与第一次运行

推荐 Linux、Python 3.11 或更新版本。在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,data,crypto]'
privfim prepare --datasets Toy
privfim run --dataset Toy --seeds 2026 --m 64 --k 5 --output results/quickstart
pytest -q
```

Toy 是固定种子生成的 1,200 行、8 个二元属性的数据，用于验证流程。
上面的 m=64、k=5 是快速验证设置。正常默认设置为 ε=1、δ=10^-5、m=2048、k=15、K=4，
项集长度 1–4，两阶段预算各占一半。运行完整默认配置：

```bash
privfim configured --config config/default.json
```

配置中的数据和输出路径相对于 JSON 所在目录解析。`python main.py ...` 与 `privfim ...` 等价。
验证环境使用 Python 3.12.3，完整依赖快照见 `requirements-lock.txt`，
已完成的检查及其范围见 [VALIDATION.md](docs/VALIDATION.md)。

## 数据准备

```bash
privfim prepare --datasets Bank DefaultCredit CensusIncomeKDD MiniBooNE PokerHand LetterRecognition
privfim run --dataset DefaultCredit --seeds 2026 2027 2028 2029 2030 --output results/credit
```

原始文件保存于 `data/raw/`，整数 CSV 与编码记录保存于 `data/real/`。
已处理文件通过 SHA-256 检查后复用。处理规则和 Bank 的历史来源边界见 [DATA.md](docs/DATA.md)。
使用 `--data-dir /your/data` 改变目录，或直接评估自己的整数表：

```bash
privfim run --csv /path/to/table.csv --seeds 2026 --output results/custom
```

CSV 首行为 `0,1,...,M-1`，其余行是整数类别码。同一列内编码须一致，缺失值应作为预先定义的类别。
不同列的相同数字是不同 item。公开值域和预处理规则是实验输入。

## 方法

| CLI 名称 | 报告与估计 |
|---|---|
| MAP-M（图中 PriVFim） | 目标单项与本地联合投影共同排序，每方最多 k 个键，正向 Alpha + MAP |
| MAP-S | 候选涉及的目标单项 Alpha，直接交集 MAP |
| MAP-L | 候选的所有唯一非空本地投影 Alpha，无 k 个键上限 |
| IE-Full（内部 fm_full） | 相关属性完整值域 Alpha，通过 noisy N 减补集并集估计 |
| IE-Other（内部 fm_other） | 未点名的属性值合并到 OTHER 桶，再通过并补估计 |
| FO | OUE 对齐报告和去偏交集估计 |
| First-round | 单轮加噪单项计数，频率乘积猜测候选支持数 |
| Second-round | 单轮报告公开本地值域中所有单项、二项、三项 Alpha，再用 MAP 估计 |
| Items-only | 单轮只报告公开值域单项 Alpha，用于额外消融 |

```bash
privfim run --dataset Toy --methods MAP-S MAP-L MAP-M --seeds 2026 --output results/map_compare
```

`mixed_joint_budget_weight>0` 在当前实现中启用属性投影测量组预算复用，不是简单的数值权重。
设为 0 表示关闭复用、逐键均分。流程和预算说明见 [METHOD.md](docs/METHOD.md)。

## 11 类实验

每个实验有独立文件，调用同一个协议与评价入口。

| 编号 | 变化参数 | 默认数据集 |
|---|---|---|
| 1 | ε = 0.25, 0.5, 1, 2, 4 | CensusIncomeKDD, MiniBooNE |
| 2 | k = 5, 10, 15, 20, 25 | CensusIncomeKDD, MiniBooNE |
| 3 | K = 2, 4, 8 | CensusIncomeKDD, MiniBooNE |
| 4 | 样本比例 0.2, 0.4, 0.6, 0.8, 1 | Bank, DefaultCredit |
| 5 | 属性合并比例，同上 | Bank, DefaultCredit |
| 6 | 值域缩减比例，同上 | Bank, DefaultCredit |
| 7 | 第一轮报告上限/k = 0.5, 0.75, 1, 1.5, 2 | DefaultCredit, MiniBooNE |
| 8 | 第二轮报告上限/k，同上 | Bank, MiniBooNE |
| 9 | 候选数/k = 1, 1.5, 2, 2.5, 5 | CensusIncomeKDD, PokerHand |
| 10 | 第一轮预算比例 0.1 至 0.9 | PokerHand, MiniBooNE |
| 11 | First-round / Second-round / MAP-M | Bank, DefaultCredit |

```bash
# 单独运行实验 4
python new-Experiments/experiment04_sample_size.py --datasets Bank DefaultCredit --seeds 2026 --output results/exp4

# 全部实验，默认 5 个 seed：2026–2030
privfim suite --experiments 1 2 3 4 5 6 7 8 9 10 11 --output results/suite

# 离线检查全部 11 个入口，不会启动大规模密码学实验
privfim suite --datasets Toy --seeds 2026 --m 32 --output results/all11_smoke

# 指定单个轴的值
privfim suite --experiments 2 --datasets Bank --values 5 15 25 --seeds 2026 --output results/k_custom
```

同一输出目录可恢复已完成且代码、数据、配置校验和一致的任务。
改动代码或数据后使用新结果目录。扩展 seed 时保留旧 seed 并增加新 seed 即可。
`--max-rows` 用于显式的小规模验证。实验 4 请使用样本比例，二者不能混用。

## 输出与评价

每个实验输出 `runs.csv`、`aggregated.csv`、`figures/*.pdf`。每个任务另存
`config.json`、`result.json`、`status.json`、`estimates.csv/json` 和 `global_truth.json`。

- F1：预测 Top-k 与完整合法项集空间中的真实 Top-k 的集合匹配。
- NCR：真实名次由 0 开始，权重为 k−rank，归一化分母为 k(k+1)/2。
- 相同真实支持数时按 canonical itemset 字典序处理并列，各方法共用规则。
- 支持数 MSE 与频率 MSE 分开保存，频率 MSE 为支持数 MSE/N²。
- 时间包含第一轮、共享秩构建、第二轮报告和服务器估计。数据读取、离线真值和绘图不计入。
- Communication 是紧凑消息格式的计算量，不是实际网络抓包值，不含传输协议头。

`protocol/summary.json` 保留核心原有候选范围评价供诊断。
**正式比较读取外层 result.json 和 runs.csv 的 evaluation_scope=global_exact 指标。**
全局真值在协议完成后独立计算，不反馈到候选生成或估计。

```bash
privfim plot --input results/credit/aggregated.csv --output results/credit/redraw
```

## 密码学

密码学按数据集独立评估，没有默认时间窗口，也不预设 F1/NCR 为 1。
真实 TFHE、Paillier 的安装运行和计时口径见 [CRYPTO.md](docs/CRYPTO.md)。
该入口不参加普通 suite 的自动调度。

## 目录与实现边界

```text
privfim/                 核心客户端、服务端、DP-FM/MAP、并补、预算、指标
workflows/               数据准备、统一运行、全局复核、汇总绘图、密码学入口
new-Experiments/         实验 1–11，各一个文件
experiments/             复用模块和原生密码学适配层
scripts/setup_crypto.py  拉取固定版本并编译 TFHE
config/                 默认 JSON 配置
data/reference/         Bank 历史整数参考副本与校验和
tests/                  核心算法、全局真值、工作流、密码学正确性测试
docs/                   数据、协议、密码学、验证记录
```

这是集中进程内的联邦协议实验实现，假设用户已对齐。可复现种子与 SplitMix64 共享秩用于实验，
没有实现生产环境的秘密 PRF 密钥分发、私有实体对齐和网络服务。条件 RDP 核算记录保留在结果中，
不能将公开实验 seed 当作服务端未知的秘密 K。

## 上传 Git

`.gitignore` 已排除虚拟环境、原始下载、大型结果、构建产物和第三方仓库。
根项目许可证尚未指定，第三方来源说明见 [NOTICE.md](NOTICE.md)。

```bash
git init -b main  # 下载源码 ZIP 时执行，已有 Git 仓库可跳过
git add .
git diff --cached --stat
git commit -m "Release reproducible PrivFim implementation"
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

需要另存一个不含环境和结果的源码包时：

```bash
python scripts/package_release.py --output ../PrivFim-release.zip
```
