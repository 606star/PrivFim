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

CSV 首行为 `0,1,...,M-1`，其余行是整数类别码。同一列内编码须一致，缺失值应作为预先定义的类别。
不同列的相同数字是不同 item。公开值域和预处理规则是实验输入。

## 方法

| CLI 名称 | 报告与估计 |
|---|---|
| MAP-M（图中 PriVFim） | 目标单项与本地联合投影共同排序，每方最多 k 个键，正向 Alpha + MAP |
| IE-Full | 相关属性完整值域 Alpha，通过 noisy N 减补集并集估计 |
| IE-Other | 未点名的属性值合并到 OTHER 桶，再通过并补估计 |
| FO | OUE 对齐报告和去偏交集估计 |
| First-round | 单轮加噪单项计数，频率乘积猜测候选支持数 |
| Second-round | 单轮报告公开本地值域中所有单项、二项、三项 Alpha，再用 MAP 估计 |
| Items-only | 单轮只报告公开值域单项 Alpha，用于额外消融 |

```bash
privfim run --dataset Toy --methods MAP-S MAP-L MAP-M --seeds 2026 --output results/map_compare
```

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

## 目录

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

