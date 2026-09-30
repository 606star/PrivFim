# 密码学按数据集评估

密码学使用独立入口，默认不设运行时间窗口，不做运行时间外推，不预设完美准确率。
每个数据集、配置和方法分别保存真实时间、支持数和全局评价。普通 11 类实验不会隐式启动加密。

## 编译

需要 Git、CMake、支持 OpenMP 的 C++ 编译器。当前原生 TFHE 配置为 x86-64 SPQLIOS FMA。

```bash
python -m pip install -e '.[crypto]'
python scripts/setup_crypto.py --jobs 2
```

脚本拉取固定 commit，构建本地共享库和适配器，不需要系统级安装权限。
VPP-OFIM 的 Paillier 实现不需要 C++ 构建，可单独运行。

## 运行

```bash
# 显式小样本验证，不是全量性能结果
privfim prepare --datasets Toy
privfim crypto --datasets Toy --clients 2 --k 3 --max-rows 4 --output results/crypto_smoke

# 按数据集评估，省略 --max-rows 使用完整数据
privfim crypto --datasets Bank DefaultCredit --seed 2026 --output results/crypto_by_dataset

# 单独运行某个方法
privfim crypto --datasets DefaultCredit --methods VPP-OFIM --output results/credit_vpp
```

默认 `--candidate-source two-round`：共享 PriVFim 第一轮加噪计数和 top-2k 候选，
第二轮分别采用原生 TFHE 查询或 Paillier ERV 支持计数。这是两轮框架中的加密替换对照，
不等于两篇原论文的完整独立候选发现算法。

`--candidate-source public` 使用完整公开合法项集域作为候选。
`--candidate-limit` 是公开枚举复杂度保护，超过上限明确报错，不按真实频率预筛，不返回部分结果。
没有默认超时，也没有默认密文 GiB 限制。完整数据的成本可能很高，执行前可查看
`python -m experiments.crypto_reproductions.preflight --datasets ... --output results/preflight.json`。

## 文件与计时

```text
results/crypto_by_dataset/
  DatasetName/
    seed2026_k15_K4_two-round/
      config.json
      candidates.json
      comparison.csv
      NIPP-FIM/{result.json,status.json,estimates.csv,global_truth.json,nipp_native.json}
      VPP-OFIM/{result.json,status.json,estimates.csv,global_truth.json}
```

时间包含本地编码、密钥生成、加密、云端计算、解密、验证，以及共享第一轮。
读取数据、离线全局真值、逐项正确性检查、写文件和绘图不计入协议计时。
失败或中断单独记录，不填入汇总为成功。成功任务按相同配置和源码校验和恢复。

F1/NCR 由完整全局真值计算。在 two-round 模式，即使加密支持数完全正确，
第一轮候选遗漏仍可能导致 F1/NCR 小于 1。epsilon 改变会改变候选，不能自动将其时间复制为水平线。

## 来源与边界

- NIPP-FIM：匹配论文标题的仓库 `https://github.com/Airscope/non-interactive-ppfim`，
  commit `d862f5c0d9dd4fa8cd155abdd8bbaa3cc6e63a81`。直接编译上游 `cloud.cpp`、`user.cpp`。
  仓库身份与论文作者的关联尚未确认，不能据此声称完整逐行复现论文。
- TFHE：`https://github.com/tfhe/tfhe`，commit `9373b5d8a5b022ca5b4e112ae1c2440bc18e160c`。
- VPP-OFIM：Zhao et al., Electronics 12(8), 1952 (2023)，DOI `10.3390/electronics12081952`。
  未找到原源码，当前实现依据论文流程加入 ERV、盲化比较和 AFI/AII 核验。

NIPP 的全局 top-k 支持数输出是适配扩展，原始接口用于阈值判定。拥有方共享 TFHE 密钥，
没有实现门限/多密钥 FHE。VPP 的比较正确性与人工验证模式有测试，未重新证明完整 transcript 的模拟安全性。
验证模式并不保证检测任意篡改。所有角色本地运行，假设用户预先对齐。
Paillier 默认 2048 bit。单元测试中的较短密钥只用于正确性验证。
