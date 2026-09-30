# PrivFim 正向 Alpha 直接交集 MAP

> 原工程理论笔记。当前工程入口和预算约定见 [METHOD.md](METHOD.md)。涉及秘密共享随机性的
> 隐私结论依赖文中前提，不表示公开 seed 的实验程序已实现生产秘密 PRF。

本文描述协议版本 `positive-alpha-map-v18-private-quantile-bin-map-m` 中 MAP-S、MAP-L 和 MAP-M 的正式估计路径。
三种方案只上传目标 item 或本地 itemset 的正向私有 Alpha，不上传同属性的其他值，
也不通过补集并集恢复交集。FM-Full 和 FM-Other 的分桶并补是独立基线。

## 1. 问题与结论

设对齐用户全集为 `Omega`，`|Omega|=N`。候选项集被拆成 `s` 个本地块

\[
T=\bigcap_{i=1}^{s}A_i,\qquad I=|T|.
\]

客户端只上传各 `A_i` 的正向 DPFM Alpha。MAP-L 上传候选集合诱导出的全部唯一
本地投影；MAP-M 才在公开的单项/投影混合池中按第一轮 DP 猜测频数选最多 `k`
个键。三种方案均不读取本地未私有化支持度。服务端执行三步：

1. 从每个完整 Alpha 向量估计边缘基数 `n_i=|A_i|`；
2. 用集合论上下界限制 `I` 的搜索区间；
3. 用多个正向私有 Alpha 的联合分布对 `I` 做 uniform-prior MAP。

对于 1-项候选只有一个本地块，服务端直接使用该私有 Alpha 的边缘基数 MAP；
对于 2--4 项候选才进入 Fréchet 边界和联合似然路径。二者在最终 Top-k 中统一排序。

两个块时，给定真实边缘基数的联合分布是精确的。代码使用私有 Alpha 的边缘 MAP
作为 `n_i` 的 plug-in 值。三个及以上块时，仅知道各边缘集合并不足以唯一确定联合
分布；当前实现使用公共交集核近似，并明确保留这一理论边界。

## 2. FM Alpha 与 DPFM

每个 Sketch 坐标使用共享的用户 ID 哈希秩

\[
G(i)\sim\operatorname{Geom}\left(\frac{\gamma}{1+\gamma}\right),
\quad
F(a)=\Pr[G(i)\le a]=1-(1+\gamma)^{-a}.
\]

集合 `A` 的干净 Alpha 为

\[
M(A)=\max_{i\in A}G(i).
\]

若 `|A|=n`，则

\[
\Pr[M(A)\le a]=F(a)^n.
\]

一个报告键得到预算 `(epsilon_k, delta_k)` 后，DPFM 计算幻影元素数 `nu_k` 和
强制下限 `tau_k`。代码中的私有 Alpha 是

\[
Y_A=\max\{M(A),M(P_A),\tau_A\},\qquad |P_A|=\nu_A.
\]

因此单坐标 PMF 为

\[
p_n(a)=
\begin{cases}
0,&a<\tau_A,\\
F(\tau_A)^{n+\nu_A},&a=\tau_A,\\
F(a)^{n+\nu_A}-F(a-1)^{n+\nu_A},&a>\tau_A.
\end{cases}
\]

服务端把一个报告的全部 `m` 个坐标联合起来，搜索 `n in [0,N]`，得到边缘基数
估计 `n_hat`。这里不是把某一个 Alpha 位置直接换算成基数。

## 3. 为什么不能直接使用逐坐标最小值

交集满足 `A intersect B` 是 `A` 和 `B` 的子集，所以干净 Alpha 具有单调性

\[
M(A\cap B)\le\min\{M(A),M(B)\}.
\]

但通常不等号成立。`M(A)` 的最大秩可能来自 `A\B`，`M(B)` 的最大秩可能来自
`B\A`；二者的较小值仍未必来自交集。因此

\[
\min\{M(A),M(B)\}\ne M(A\cap B)
\]

一般成立。私有 Alpha 还包含彼此独立的幻影元素和截断，下述错误会更明显：

- 不能把逐坐标 `min(Y_A,Y_B)` 当成交集的 DPFM Alpha；
- 不能把这个最小值代入单集合 PMF `F(t)^I-F(t-1)^I`；
- `alpha_min` 只能提供随机的单调上界信号，不能单独识别交集频数。

本实现保留 Section 5.4 的“集合边界 + MAP”思路，但用正向 Alpha 的联合 PMF
替代上述单集合似然。

## 4. 集合论搜索边界

给定边缘集合大小 `n_1,...,n_s`，Frechet 界为

\[
L=\max\left(0,\sum_{i=1}^{s}n_i-(s-1)N\right),
\qquad
U=\min_i n_i.
\]

证明如下：交集是每个集合的子集，所以 `I<=U`；各集合补集的并集大小最多为
各补集大小之和，由 `I=N-|union A_i^c|` 得到下界。

代码用私有 Alpha 得到的 `n_hat_i` 代入该式，并取整数闭区间
`[ceil(L), floor(U)]`。这比把逐坐标最小 Alpha 当成一个交集 Sketch 更合理，
但它仍是 plug-in 边界，不是已经证明覆盖率的置信区间。论文若要声明置信覆盖率，
还需要为边缘 DPFM-MAP 推导并实现统一的置信界。

## 5. 两个正向 Alpha 的精确联合分布

令 `|A|=n_A`、`|B|=n_B`、`|A intersect B|=I`。把真实用户分成三个互斥区：

\[
A\setminus B,\quad B\setminus A,\quad A\cap B,
\]

大小分别为 `n_A-I`、`n_B-I` 和 `I`。给定观测上限 `a,b`，若
`a>=tau_A` 且 `b>=tau_B`，私有 Alpha 的联合 CDF 为

\[
\begin{aligned}
C_I(a,b)
&=\Pr(Y_A\le a,Y_B\le b\mid I)\\
&=F(a)^{n_A-I+\nu_A}
  F(b)^{n_B-I+\nu_B}
  F(\min(a,b))^I.
\end{aligned}
\]

若任一上限低于对应强制下限，CDF 为 0。这个式子显式区分了：

- 只属于 `A` 的真实用户；
- 只属于 `B` 的真实用户；
- 同时属于二者、在两个报告中使用相同公共秩的用户；
- 每个报告独立加入的幻影元素。

离散联合 PMF 由二维有限差分得到：

\[
p_I(a,b)=C_I(a,b)-C_I(a-1,b)-C_I(a,b-1)+C_I(a-1,b-1).
\]

`m` 个独立坐标的 log-likelihood 为

\[
\ell(I)=\sum_{r=1}^{m}\log p_I(y_{A,r},y_{B,r}).
\]

在 `[L,U]` 上使用均匀先验时，MAP 与该网格上的最大似然相同：

\[
\widehat I=\arg\max_{I\in[L,U]}\ell(I).
\]

实现先在同一候选网格的 64 个等距点上定位似然峰，再把峰值左右相邻的两个
粗网格区间按原始 `map_step`/`max_map_points` 分辨率完整求值。该自适应搜索不
改变候选网格、似然或先验；数值验证同时保留小规模全网格对照，防止峰值定位
优化悄然改变估计结果。

这就是 MAP-S/MAP-L/MAP-M 在两个本地块情况下的直接交集估计。它没有构造补集，也没有
执行 `N - union_count`。

## 6. 多块公共交集核近似

对 `s>=3`，只知道 `n_i` 和最终公共交集 `I`，不能确定所有二阶、三阶部分交叠。
例如三个集合的联合 Alpha 分布还依赖 `|A intersect B|`、`|A intersect C|` 和
`|B intersect C|`。因此仅由当前报告对象不能写出只含参数 `I` 的精确似然。

当前实现采用公共交集核：假设块之间共享的真实用户全部属于最终公共交集，其余
部分在各块间不再重叠。此时联合 CDF 为

\[
C_I(\mathbf a)=
\left[\prod_{i=1}^{s}F(a_i)^{n_i-I+\nu_i}\right]
F(\min_i a_i)^I.
\]

联合 PMF 由 `2^s` 个 CDF 角点作有限差分。该模型在 `s=2` 时退化为上一节的精确
公式；在 `s>=3` 时是明确的 plug-in 近似，不应写成无条件精确定理。

MAP-L/MAP-M 会先在同一客户端构造本地交集块，通常能减少服务端块数。因此它们
除了提供更直接的本地相关结构，也可能让更多候选落入精确两块情形。MAP-L 完整
保留所有候选投影但可能增加报告键；MAP-M 限制键数却可能令候选不可估计。

## 7. MAP-S、MAP-L 与 MAP-M 的预算

设客户端 `j` 的第二轮总预算为 `(epsilon_2j, delta_2j)`，固定报告键数为 `K_j`：

\[
\varepsilon_{j,k}=\frac{\varepsilon_{2j}}{K_j},
\qquad
\delta_{j,k}=\frac{\delta_{2j}}{K_j}.
\]

MAP-S 只包含目标单项键。MAP-L 的键集合是 `S` 中每个候选在客户端 `j` 上的
非空投影去重结果，不设上限；它不会无条件附带投影的组成单项。MAP-M 将 MAP-S
单项和 MAP-L 投影合并后按公开猜测频率最多选择 `k` 个键，因此
`K_j^M<=k`。女性、其他年龄或任意同属性非目标值不会获得 MAP 预算。

MAP-L/MAP-M 的权衡是：

- 候选使用的本地块更少，且同方交叠在客户端已精确形成；
- MAP-L 完整覆盖 `S`，但键数增加会摊薄单键预算；
- MAP-M 优先保留更高公开猜测频率的键，未选键不上传，对应候选只有在其本地块
  仍可由已上传单项重建时才估计。

因此不能仅由集合论证明 MAP-L 必然优于 MAP-S。

## 8. FM-Full 与 FM-Other

FM 基线不使用上述直接交集似然。对属性 `a` 的值域桶 `H_{a,v}`，完整值域构成
互斥完备划分：

\[
\Omega\setminus H_{a,v}
=\bigcup_{w\in D_a,w\ne v}H_{a,w}.
\]

FM-Full 上传每个相关属性的全部正向值域桶 Alpha。服务端合并所有非目标桶，估计
补集并集后返回 `N-union_count`。每个桶的 Alpha 在上传前已经独立完成 DPFM
私有化，每个桶都占一个预算键。

FM-Other 定义

\[
H_{a,OTHER}=\bigcup_{w\in D_a\setminus V_a(S)}H_{a,w}.
\]

服务端对目标 `v` 合并其他已点名值桶和 `OTHER` 桶，仍精确得到目标值补集。
它减少的是未被 `S` 点名的值域桶，而不是把 MAP 改成并补。

## 9. 隐私与实现前提

每个 Alpha 向量使用 DPFM 的单键预算，并在客户端内部加入幻影元素和强制下限。
若多个报告键复用同一用户哈希秩，隐私对象是完整联合输出，不能先假设各键独立。
本版本用第 10 节的重叠 RDP 会计审计全部客户端的第二轮 tuple，再与第一轮自适应
组合。服务端的边缘 MAP、联合 MAP、排序和 Top-k 都是后处理。

严格部署还要求：

1. 第二轮键只由第一轮 DP 输出 `S` 和公开元数据确定，零支持键也要报告；
2. FM-Full 使用预先公开的完整值域，不能从私有数据临时发现桶；
3. 各方对相同行使用一致的安全 keyed PRF 秩，服务端不能掌握秘密哈希密钥；
4. 幻影采样和 Laplace 噪声使用客户端新鲜秘密随机性；
5. 若同时发布多个实验方案，必须组合计算其隐私损失。

当前 Python 原型用固定种子追求可复现，不应直接宣称为部署级信息论 DP 系统。

## 10. VertiMRF 的借鉴与 MAP 的联合隐私

VertiMRF 是当前最接近本项目隐私问题的纵向 FM/DPFM 工作之一。它采用各方共享、
但服务器未知的秘密哈希密钥，并把同一哈希下产生的一组 Sketch 作为一个联合输出
进行隐私分析，而不是把每个 Sketch 当成相互独立的机制。其证明使用条件概率分解
和 RDP，再转换为近似 DP；跨方边缘估计阶段使用 FM Sketch 的并补/容斥，估计本身
不再额外消耗隐私预算，因为它是已发布 Sketch 的后处理。

这给本项目三个直接约束：

1. **哈希密钥必须是秘密的。** 客户端可以协商相同的 keyed PRF/哈希秩，但服务端
   不能知道密钥，也不能从实验固定 seed 推断真实秩。固定 seed 只适用于可复现实验。
2. **隐私对象是完整 Alpha tuple。** 对一个客户端，应将其第二轮全部报告键、全部
   Sketch 坐标和同一轮的共享哈希看作一个机制 `M_j`，先证明 `M_j(D_j)` 对相邻本地
   数据集满足 DP/RDP，再进行阶段和客户端组合。
3. **MAP 的后处理不产生新隐私损失。** 边缘基数 MAP、Frechet 边界、联合似然、排序
   和 Top-k 都只读取已私有化 Alpha；但第二轮的键集合必须只由第一轮 DP 输出 `S`
   和公开元数据决定，不能由本地未加噪支持度决定。

VertiMRF 的桶定理不能原样替代 MAP 的定理。其典型结构是每条记录在一个属性下只
命中一个值桶，而本项目可能同时发布 `{男}`、`{15岁}` 和 `{男,15岁}`。本节把其
条件链/RDP 证明中的命中桶数推广为报告键命中数，并显式审计完整 tuple。

### MAP 所需的联合 DP 定义

对客户端 `j`，令第二轮固定报告键为 `R_j`，一个用户记录 `x` 命中的键集合为

```text
A_j(x) = {r in R_j : x satisfies r}
```

并令 `q_j = max_x |A_j(x)|`。完整机制应写成

```text
M_j(D_j) = {Alpha(r, shared_hash, fresh_randomness) : r in R_j}
```

`q_j` 描述重叠发布结构，而不是通信键总数本身。MAP-M 使用测量组时，下面的 `q_j`
表示可同时命中的组数。设客户端 `j` 有 `d_j` 个与 `S` 相关的属性，则：

| 方案 | 一条记录可能命中的第二轮键 |
|---|---|
| FM-Full | 每个相关属性恰好一个完整值域桶，故 `q_j=d_j` |
| FM-Other | 每个相关属性恰好一个目标值或 OTHER 桶，故 `q_j=d_j` |
| MAP-S | 每个相关属性至多一个被 `S` 点名的单项，故 `q_j<=d_j` |
| MAP-L | 所有被该记录满足的候选本地投影键，最坏 `q_j<=|R_j^L|` |
| MAP-M / MAP-M-Bin | Top-k 键按属性投影分组；同组值桶互斥，最坏命中测量组数 `q_j<=G_j<=k` |

若 MAP-L 允许的最大本地投影阶数为 `h`，一个更结构化但仍保守的界是
`q_j <= sum_{l=1}^{h} C(d_j,l)`；实际 `R_j` 只包含 `S` 诱导出的投影，通常远小于
这个组合数。该量应由公开的 `S` 和属性归属计算，而不能扫描私有行后再决定报告键。

### 联合 RDP 定理

采用 add/remove 用户级相邻关系：`D' = D union {x}`，且该对齐用户的纵向属性可
同时出现在所有客户端。条件于任意固定的第一轮 DP transcript，第二轮报告键集合
`R_j` 是公开且固定的。

对客户端 `j` 的键 `r`，令 DPFM 的单哈希坐标纯 DP 参数为

```text
eta_jr = epsilon_jr / (4 * sqrt(m * log(1 / delta_jr))).
```

对 MAP-S、MAP-L 与 FM，报告原语是键，公开加权命中界定义为

```text
B = max_x sum_j sum_{r in A_j(x)} eta_jr^2.
```

MAP-M 的报告原语改为属性投影组 `g`：同组中的键至少有一个属性取值冲突，单个用户
最多命中一个桶，因此每个组可并行复用 `epsilon_2j/G_j` 与 `delta_2j/G_j`。代码对
MAP-S/L/FM 使用 `B <= sum_j q_j max_{r in R_j}(eta_jr^2)`，对 MAP-M 使用
`B <= sum_j G_j eta_jg^2`。对任意 `lambda >= 2`，全部客户端、一个哈希坐标的
联合输出满足

```text
D_lambda(M_2^(1)(D) || M_2^(1)(D')) <= 2 * lambda * B.
```

`m` 个独立 PRF 域分离坐标按 RDP 组合，得到

```text
D_lambda(M_2(D) || M_2(D')) <= 2 * m * lambda * B.       (1)
```

证明要点如下。

1. 对固定坐标，把受新增用户影响的报告原语按任意顺序排列。未命中的原语分布完全相同，
   隐私损失为 0。
2. 对命中原语作与 VertiMRF 附录相同的条件链分解。MAP-M 同一测量组内先由互斥值桶
   并行组合为一个原语；若当前 Alpha 没有被此前共享秩
   确定，则使用 DPFM phantom/truncation 标量机制的纯 DP 界；若已被同一共享秩
   确定，相邻数据下的对应条件分布相同，该项隐私损失为 0。
3. 对非零条件项应用纯 DP 到 RDP 的界，每个命中原语贡献至多
   `2 lambda eta^2`。求和得到单坐标的 `2 lambda B`。
4. 不同坐标使用独立、域分离的 PRF 输入和独立幻影随机性，RDP 可加，因此得到式
   (1)。共享哈希秩只在同一坐标内保留交集所需的联合模式，不破坏跨坐标组合。

RDP 到近似 DP 的标准转换给出

```text
epsilon_2(lambda)
  = 2 * m * lambda * B + log(1 / delta_2) / (lambda - 1).  (2)
```

代码取 `lambda>=2` 上的闭式最优点（若低于 2 则取 2），得到
`epsilon_2^cert`。若 `epsilon_2^cert <= epsilon_2^target`，则该方案的完整第二轮
tuple 在上述部署前提下满足 `(epsilon_2^target, delta_2)`-DP。注意，这不是把
`q_j` 乘到单键 approximate-DP epsilon 上；`q_j` 或 `G_j` 进入的是单坐标 RDP 的
平方加权命中项。

### 两轮组合

设总预算为 `epsilon`，并以 `rho_N` 预留 noisy N 的发布，则
`epsilon_N = rho_N * epsilon`，`epsilon_1 = (1-rho_N) * epsilon * phase1_ratio`，
`epsilon_2^target = (1-rho_N) * epsilon * (1-phase1_ratio)`。每个客户端的 noisy N
报告使用 `epsilon_N/P`；这里的 `epsilon_1` 和 `epsilon_2^target` 分别是跨客户端
组合后的阶段预算。

第一轮对公开完整值域的每属性直方图使用 Laplace 机制。每个客户端先按属性数拆分，
再跨客户端组合，得到纯 `epsilon_1`-DP。第二轮键集合只依赖第一轮 DP 输出和公开
元数据；由于式 (1) 对每个固定 transcript 都成立，自适应组合给出

```text
M(D) = (M_N(D), M_1(D; M_N(D)), M_2(D; M_N(D), M_1(D)))
is (epsilon_N + epsilon_1 + epsilon_2^cert, delta_2)-DP.   (3)
```

这里 `M_N` 是各客户端独立 Laplace 加噪行数的完整 tuple；每方使用
`epsilon_N/P`，跨方顺序组合为 `epsilon_N`，服务端均值是后处理。实验逐方案检查
`epsilon_N + epsilon_1 + epsilon_2^cert <= epsilon_total`。多个方案在同一数据
上同时发布时，仍须对这些方案再次组合，不能共用一次证书。

式 (1)--(3) 使用 add/remove 相邻关系。若论文改用 replace-one 相邻关系，变化前后
记录命中键的并集最多含 `2q_j` 个键，会计器必须把公开命中上界相应加倍后再审计。

## 11. 能证明与不能证明

当前实现可以严格说明：

- 单报告私有 Alpha 的边缘 PMF；
- 两个正向报告在给定真实边缘基数时的联合 CDF/PMF；
- Frechet 交集上下界；
- 两块情形下联合似然直接针对交集，而不是并补；
- FM-Full/FM-Other 的分桶并补集合恒等式；
- 在服务器未知共享 PRF、新鲜随机性和 add/remove 相邻关系下，完整第二轮重叠
  Alpha tuple 的联合 RDP 界及两轮组合预算证书。

当前实现不能宣称：

- 逐坐标最小 Alpha 就是交集 Alpha；
- plug-in 边缘基数等于真实边缘基数；
- 三块以上公共交集核是精确联合分布；
- 当前公开固定 seed 的模拟运行本身已经达到部署级端到端 DP；
- 同时发布多个实验方案仍只消耗单个方案的预算；
- MAP-L 或 MAP-M 在所有数据集和预算下都优于 MAP-S 或 FM；
- 固定总 epsilon 时增大 `m` 必然提高精度。

## 12. 代码对应

| 理论对象 | 实现 |
|---|---|
| 正向 MAP 报告 | `VerticalClient.round2_reports` |
| MAP-S/MAP-L/MAP-M 本地块 | `PrivFimServer._blocks_for_candidate` |
| 单集合边缘 MAP | `private_cardinality_map_estimate` |
| Frechet 边界 | `intersection_bounds` |
| 联合 CDF/有限差分 PMF | `_joint_log_cdf` / `_joint_log_probability` |
| 直接交集 MAP | `map_intersection_estimate` |
| FM 分桶报告 | `_fm_report_keys` / `_fm_membership` |
| FM 分桶并补 | `fm_category_intersection_estimate` |
| MAP-M 测量组与组命中上界 | `measurement_groups` / `public_measurement_group_overlap_bound` |
| 其他模式的公开报告键命中上界 | `public_overlap_bound` |
| 重叠 tuple RDP 会计 | `overlap_rdp_account` |
| MAP-M-Bin 私有等质量分箱 | `private_quantile_bins` / `binned_itemset` |
| MAP-M-Bin 分箱计划下行通信 | `binning_message_bytes` |

## 13. 参考文献

1. Adam Smith, Shuang Song, Abhradeep Guha Thakurta. *The
   Flajolet-Martin Sketch Itself Preserves Differential Privacy: Private Counting
   with Minimal Space*. NeurIPS 2020.
2. Zitao Li, Tianhao Wang, Ninghui Li. *Differentially Private Vertical Federated
   Clustering*. PVLDB 16(6), 2023.
3. Zitao Li et al. *VertiMRF: Differentially Private Vertical Federated Data
   Synthesis*. arXiv:2406.19008. 共享哈希 tuple 的联合隐私分析和 FM/CarEst 后处理。
4. `oldPrivFim.pdf`, Section 5.4. 本实现采用其“集合边界 + MAP”框架，但修正了
   将逐坐标最小 Alpha 直接视为交集 FM Alpha 的分布问题。
