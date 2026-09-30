# PrivFim 协议设计

> 历史设计笔记：预算和入口在后续迭代中有所调整。当前默认值、noisy N 比例和全局评价口径以
> [METHOD.md](METHOD.md) 与 [README](../README.md) 为准。本文件保留用于追溯设计过程。

MAP 的联合似然、适用范围和理论边界见 [map_theory.md](map_theory.md)。

## 1. 方案定义

| 方案 | 候选域 | 客户端第二轮报告 | 服务端估计 |
|---|---|---|---|
| MAP-S-All | 第一轮频繁单项形成的全部合法 1--4 项集 | 目标单项的正向私有 Alpha | 单项边缘 MAP / 多项直接交集 MAP |
| MAP-S | SVSM 统一 Top-2k（包括 1-项集） | 目标单项的正向私有 Alpha | 单项边缘 MAP / 多项直接交集 MAP |
| MAP-L | 与 MAP-S 相同 | 每个候选在本方的唯一非空投影 Alpha，不限键数 | 单项边缘 MAP / 多项直接交集 MAP |
| MAP-M | 与 MAP-S 相同 | 目标单项与本地投影混合排序，每方最多 `k` 个 Alpha | 单项边缘 MAP / 多项直接交集 MAP |
| MAP-M-Bin | 与 MAP-S 相同 | MAP-M 的第一轮私有等质量桶 Alpha，每方最多 `k` 个桶键 | 桶级 MAP 后私有桶内恢复 |
| FM-Full | 与 MAP-S 相同 | 相关属性完整值域的正向桶 Alpha | 分桶并补 |
| FM-Other | 与 MAP-S 相同 | 目标值桶及每属性一个 OTHER 桶 | 分桶并补 |

代码标识 `direct_union_complement` 为兼容已有配置而保留，现在对应
`MAP-S-All`，不再表示 MAP 使用补集。协议版本为
`positive-alpha-map-v21-noisy-n-mean`，并进入实验缓存键。

## 2. 数据模型

数据按属性纵向划分，各客户端共享对齐的用户行。一个项表示为
`(attribute_id, value)`；一个合法项集不能包含同一属性的两个不同取值。

属性可以均匀划分，也可以由 `attribute_ratios` 指定每方属性数量比例。该比例只
改变属性归属，不改变用户行对齐关系。

## 3. Noisy N、第一轮与候选 S

对齐后每个客户端的本地行数均为 `N`。客户端 `j` 使用独立随机性上传

```text
noisy_N_j = N + Laplace(1 / epsilon_N_client)
```

服务端计算所有报告的均值，再截断到至少 1，得到后续协议统一使用的
`noisy_N`。真实 `N` 不作为协议输入，只用于离线真值评价。

每个客户端对本地属性形成直方图，为计数加入 Laplace 噪声后上传。第一轮客户端
预算先按本地属性数均分。服务端聚合计数并选取 Top-`k` 频繁单项。

对频繁单项的加噪计数 `c_i` 和 `noisy_N`，SVSM 风格的猜测计数为

```text
guess_count(X) = noisy_N * product(c_i / noisy_N)
```

服务端枚举 Top-`k` 单项形成的合法 1--4 项集，过滤同属性冲突后统一按猜测分数
选择 Top-`2k`。1-项集与多项集在同一个排序中竞争，因此最终 `|S|=2k`。若合法
组合总数不足 `2k`，运行会显式失败而不是悄悄缩小 `S`。该猜测只用于候选筛选，不作为
第二轮 MAP 的先验。最终 Top-k 在 1--4 项候选的第二轮估计频率上统一排序。

## 4. 两轮预算

总预算先为 noisy N 预留比例 `noisy_n_ratio`，再按阶段和客户端划分剩余预算：

```text
epsilon_N              = epsilon * noisy_n_ratio
epsilon_N_client       = epsilon_N / client_count
epsilon_statistics     = epsilon * (1 - noisy_n_ratio)
epsilon_phase1_client  = epsilon_statistics * phase1_ratio / client_count
epsilon_phase2_client  = epsilon_statistics * (1 - phase1_ratio) / client_count
delta_phase2_client   = delta / client_count
```

同一对齐用户可能同时影响全部客户端的行数报告，因此 `epsilon_N_client` 跨客户端
顺序组合为 `epsilon_N`；求均值和正值截断不再增加隐私损失。

客户端第二轮协议先根据公开候选 `S` 确定固定的唯一报告键集合 `R_j`，再按实际键数均分：

```text
epsilon_key = epsilon_phase2_client / |R_j|
delta_key   = delta_phase2_client / |R_j|
```

因此四个协议在实验中的客户端第二轮名义预算相同，但键数、单键预算、噪声和通信量
不同。单键均分只负责噪声校准；MAP 键复用共享哈希时，完整输出是否落在目标预算内
由第 10 节的重叠 tuple RDP 会计另行审计。

MAP-M 例外地按测量组分配：相同属性投影下的不同值键互斥，故先把预算均分给投影组，
再让组内每个值桶复用该组预算。此规则与 MAP-S/MAP-L 的逐键均分分开记录，避免在
内部比较中改变后两者的定义。

## 5. MAP-S、MAP-L 与 MAP-M

客户端 `j` 对候选 `X` 的本地投影记为 `X_j`。

MAP-S 的报告键为所有候选涉及的唯一单项：

```text
R_MAP-S(j) = unique({{x} | x in X_j, X in S})
```

MAP-L 上传每个候选在本方的实际非空投影并去重，不附带候选没有要求的组成单项：

```text
R_MAP-L(j) = unique({X_j | X in S, X_j != empty})
```

MAP-M 保留限键权衡，但优先保护 MAP-S 的单项基础信息。当相关单项不超过 `k`
时先全部保留，再从剩余槽位选择高价值局部联合键；若单项已经超过 `k`，
则退化为公开频率 Top-k 单项：

```text
P_MAP-M(j) = R_MAP-S(j) union R_MAP-L(j)
R_MAP-M(j) = R_MAP-S(j) union Top-B(P_MAP-L(j) minus R_MAP-S(j))
\quad |R_MAP-M(j)| \le k
```

MAP-L-Top 是另一条保守的局部联合键改进路径。它固定保留所有 MAP-S 单项回退键，
再从长度大于一的本地投影中选取最多 `B` 个：

R_MAP-L-Top(j) = R_MAP-S(j) union Top-B({Y in R_MAP-L(j): |Y| > 1}).
```

排序只使用第一轮公开候选猜测：投影 `Y` 的收益为其覆盖候选的猜测频率之和乘以
`(|Y|-1)`。可为联合键设置大于一的公开预算权重。服务端会融合 `{a}`、`{b}` 与
`{a,b}` 三份 Alpha 的嵌套联合似然；缺失联合块时，因组成单项都已报告，按 MAP-S
单项块回退。因此它不会像 MAP-M 那样因某个所需单项被截断而跳过候选。

MAP-M 在触发限键时使用所有剩余键槽位加入排序后的联合键，避免再引入一个独立的槽位比例超参数。
其第二轮预算按公开的属性投影测量组分配，而不是按单个键的类型加权分配：同一投影的不同
取值键相互互斥，因而并行复用一份组预算；不同投影组可能同时命中，故在组间均分总预算并
以组数进行 RDP 审计。`mixed_joint_budget_weight` 的正值仅为兼容旧配置保留，具体大小不影响
MAP-M；设为 0 时启用完全关闭分组复用的消融，对所有上传键逐键均分预算。
`public_guess_frequency` 只由下发的 `S` 和第一轮 DP 频数构成。若本地键
`Y={(a_1,v_1),..., (a_r,v_r)}` 的单项猜测频数为 `c_i`，排序分数为
`noisy_N * product(c_i/noisy_N)`；缺少单项分数时退化为生成该键的候选猜测频数。因此选键
集合不依赖未私有化的本地支持度，且只有 MAP-M 满足 `|R_MAP-M(j)| <= k`。

### 可选的 MAP-M-Bin 高维分箱

当配置把 `mixed_binned_itemset_alpha`（MAP-M-Bin）加入 `modes` 且
`map_m_bin_count=b` 非空时，该消融保持精确候选的公开 Top-k 选择，但在生成第二轮
Alpha 前将每个属性的公开完整值域按值序切成至多 `b` 个连续的近似等质量桶。桶边界由
第一轮已经私有化的完整值域直方图确定：依次选择接近剩余平均私有质量的前缀，并保证每桶
至少保留一个值；若该属性的私有质量全为零，则退回公开等宽分桶。等价的精确键合并为一个
桶键，因此报告数可能低于 `k`。服务端对桶键执行同一套正向 MAP，再以同一份第一轮
私有直方图估计

```text
P(a=v | bin(a,v)) = noisy_count(a=v) / sum_{w in bin(a,v)} noisy_count(a=w)
```

全桶退化为零时回退为公开均匀分布。对于精确候选 `X`，当前恢复器使用
`estimated_count(X) = estimated_count(bin(X)) * product_{(a,v) in X} P(a=v | bin(a,v))`。
因此它是桶内跨属性条件独立近似，而不是精确 MAP，也不应与默认 MAP-M 混称。分箱边界
的值域范围是公开的；具体桶边界和桶内概率只读取第一轮 DP 输出，故不会增加隐私访问
或隐私预算。
由于边界本身是第一轮 DP 输出的函数，服务端会把实际报告键涉及属性的连续桶切点定向下发
给所属客户端；公开完整值域已在客户端可得，所以无需重复发送完整桶表或无关属性的边界。
实验通信量已将这部分第二轮准备下行计入 `binning_downlink_bytes`。

对每个键 `Y`，客户端上传的是满足 `Y` 的正向集合：

```text
reported_membership(Y) = intersection(membership(x), x in Y)
```

例如客户端持有性别和年龄，若 `S` 只诱导 `{男,15岁}` 这个本地投影，则 MAP-L
只上传 `{男,15岁}`。若 `S` 还包含会在本方投影为 `{男}` 和 `{15岁}` 的其他候选，
MAP-L 才额外上传这两个单项。MAP-M 则把这些投影与组成单项统一排序并限为 Top-k。女性、其他
年龄和 `OTHER` 都不属于 MAP 的报告键，也不分走 MAP 的预算。

服务端对 MAP-S 把候选拆成单项块；对 MAP-L/MAP-M 优先使用每方的本地联合块。它先用
每个完整私有 Alpha 向量估计块的边缘基数，再由集合论得到交集搜索区间，最后对
正向 Alpha 建立联合似然直接估计交集。MAP 不执行并补，也不使用
`N - estimated_union`。

## 6. FM-Full

对与 `S` 重合的属性 `a`，FM-Full 上传其完整值域 `D_a` 的正向分桶 Alpha：

```text
R_FM-Full(j) = {{a=v} | a is relevant to S, v in D_a}
```

以目标 `{性别=男, 年龄=15}` 为例，服务端取性别下除男以外的全部桶，以及年龄
下除 15 以外的全部桶，逐坐标取最大值。完整值域桶是互斥完备划分，所以这些桶
的并集精确等于目标交集的补集。服务端估计该并集基数后返回 `N - union_count`。

FM-Full 的所有上传 Alpha 都已经分别注入 DPFM 噪声。每一个值域桶都是一个报告
键，都会分配一份单键预算。

## 7. FM-Other

设 `V_a(S)` 是候选集合点名的属性值。FM-Other 保留这些目标值，并把其余值合成
一个固定占位桶：

```text
OTHER_a = {rows whose value is not in V_a(S)}
R_FM-Other(j) = {{a=v} | v in V_a(S)} union {{a=OTHER}}
```

对某个目标值 `v`，服务端合并其他目标值桶和 `OTHER` 桶。这仍然等于
`D_a \ {v}`，因此并补集合逻辑不变。与 FM-Full 相比，原本大量与 `S` 无关的
值只占一个报告键和一份预算。

如果 `S` 已点名属性下多个值，这些值仍分别保留，因为它们也是其他候选的目标；
只有没有被 `S` 点名的值进入共同 `OTHER` 桶。

## 8. 固定通信模式

第二轮键只能由第一轮 DP 输出得到的 `S`、公开属性归属和所选协议决定，不能由
未私有化的本地支持度决定。即使某个固定键本地支持度为 0，也要生成带幻影元素
和强制下限的私有 Alpha，否则“是否上传”本身会泄漏信息。

实验原型的 FM-Full 值域由载入数据的离散域构造。严格部署时应预先公开属性值域
并补齐零桶，不能在第二轮临时用私有数据发现值域。

## 9. 评估口径

协议过程不使用真实支持度。完整数据只在离线评估阶段计算真实 Top-k 和候选误差。
主要指标包括 Precision、Recall、F1、Jaccard、NCR、MAE、RMSE、唯一报告键数、
单键预算、Alpha 载荷、完整紧凑通信字节数和客户端/服务端耗时。

多个方案在实验中是独立模拟。真实系统若同时发布多个方案输出，必须组合计算各次
发布的隐私损失，不能把它们视为只消耗一次预算。

## 10. 重叠 tuple 的联合隐私会计

本协议可以借鉴 VertiMRF 的共享哈希设计：客户端协商同一组 keyed PRF 哈希秩，
但服务器不知道密钥；服务器只对已经私有化的 Alpha 做边缘 MAP、联合 MAP、并补、
排序和 Top-k，这些步骤都是后处理。VertiMRF 的隐私证明还说明，共享哈希下应将
同一客户端乃至全部纵向方的一组 Sketch 作为联合 tuple 分析，而不能把单键结论
直接当成完整输出结论。

对于 MAP-S、MAP-L 与 FM，令公开报告键为 `R_j`，记录 `x` 命中的键为
`A_j(x)={r in R_j: x satisfies r}`，并令 `q_j=max_x |A_j(x)|`。实现只根据第一轮
DP 输出、公开值域和属性归属构造 `R_j`，再用 `q_j <= 单项键涉及的属性数 + 多项键数`
给出不扫描私有行的保守上界。设每个键的 `m` 维 DPFM Alpha 得到预算
`(epsilon_jr, delta_jr)`，其单哈希坐标参数为

```text
eta_jr = epsilon_jr / (4 * sqrt(m * log(1 / delta_jr))).
```

MAP-M/MAP-M-Bin 则以属性投影组 `g` 作为报告原语。相同投影的不同值键必在至少一个
属性上冲突，一条记录至多命中组内一个桶。若 `G_j` 是公开投影组数，则每组复用
`(epsilon_2j/G_j, delta_2j/G_j)`，并令其单坐标参数为 `eta_jg`。组数是保守命中上界。
在 add/remove 相邻关系、服务器未知的共享 PRF 和各原语独立新鲜幻影随机性下，
VertiMRF 的条件链证明可从“每属性命中一个桶”推广到“记录至多命中公开数目的键或组”。
对任意 RDP 阶数 `lambda >= 2`，全部客户端第二轮 tuple 的保守界为

```text
B <= sum_j q_j * max_r(eta_jr^2)     (逐键模式)
B <= sum_j G_j * eta_jg^2            (MAP-M/MAP-M-Bin)
RDP_lambda(M2(D) || M2(D')) <= 2 * m * lambda * B
epsilon2(lambda) = 2 * m * lambda * B
                   + log(1 / delta2) / (lambda - 1).
```

会计器取上述表达式的最小值，并检查 `epsilon2 <= epsilon_phase2`。第一轮完整公开
值域直方图是 `epsilon_phase1`-DP；第二轮键集合只通过第一轮 DP 输出自适应确定，
因此标准自适应组合给出

```text
(epsilon_N + epsilon_phase1 + epsilon2, delta2)-DP.
```

实验 CSV/JSON 记录各方 noisy N、其均值、逐键或测量组命中上界、`B`、RDP 阶数、
`epsilon2`、总 epsilon、
预算余量以及每个 MAP-M 测量组的实际预算。每个方案单独审计；同时发布 MAP-S、MAP-L、
MAP-M 和 FM 输出必须再组合这些机制。

当前实现中的固定哈希 seed 和实验真值只用于可复现评价；真值不会作为协议输出，
第一轮和 FM 已改用公开完整值域，报告模式也不依赖本地支持度。由于模拟进程中的
seed 对服务器可见，`formal_end_to_end_dp` 仍为 `false`。RDP 通过状态只表示：若
部署替换为服务器未知的共享 keyed PRF 与新鲜私有随机性，该方案的理论 tuple 证书
落在声明预算内。
