---
author: Research Proposal Draft
title: "AIRA: Adaptive Influence-aware AiTM Attack for LLM-based Multi-Agent Systems"
---

# AIRA: Adaptive Influence-aware AiTM Attack for LLM-based Multi-Agent Systems

## 1. 核心问题

原始 AiTM 预先指定 victim。AIRA 保留 AiTM 的消息篡改方式，只增加一个在线攻击决策层：

> 在当前同步轮次已经生成、尚未交付的消息边中，选择最值得篡改的一条具体边。

候选单位不是 Agent，而是带轮次的消息边：

$$
e_t=(a_u\rightarrow a_v,t)
$$

当前实现只决定“攻击哪条边”。攻击时机不是学习结果：默认在达到观察阈值后的第一个可攻击批次选择一次，三轮同步协作中通常为第 1 轮生成后、第 2 轮消费前。

------------------------------------------------------------------------

# 2. Threat Model

攻击者知道 MAS 的协作拓扑和轮次编排，可以观察、截获并修改当前在途消息，但不能访问：

- Agent system prompt；
- 隐藏 memory；
- 模型参数；
- 尚未生成的未来消息。

所有 Agent 在同一轮先独立生成输出，当前批次消息全部生成后才进行选边和投递。因此选择器只能使用：

- 已知协作拓扑；
- 当前轮次和总轮数；
- 当前批次可拦截消息；
- 当前轮各 Agent 已生成的可观察输出；
- 攻击目标。

------------------------------------------------------------------------

# 3. 时间展开协作图

将已知静态拓扑展开为有向无环时间图。节点 $a_i^r$ 表示 Agent $a_i$ 在第 $r$ 轮生成的输出状态。

时间图包含：

1. 通信依赖边：

$$
a_u^r\rightarrow a_v^{r+1}
$$

2. 自身记忆边：

$$
a_i^r\rightarrow a_i^{r+1}
$$

当前 AutoGen 实现会在后续轮次保留 Agent 自身历史输出，因此自身记忆边是真实执行依赖，不是额外假设。

3. 终端决策边。终端必须与真实调度一致：

- Chain：仅最终轮 $A2^T\rightarrow J$；
- Tree：最终轮 $A0^T,A1^T\rightarrow J$；
- Asymmetric-tree：最终轮 $A0^T\rightarrow J$；
- Complete / Random：Judge 读取完整多轮讨论，因此各轮所有 Agent 输出均连接到 $J$。

------------------------------------------------------------------------

# 4. Target-aware Temporal Reachability

对当前候选边 $e_t$，计算强制先经过该边后，到最终决策节点 $J$ 的折扣时序路径质量：

$$
R_{raw}(e_t)=
\sum_{p:e_t\rightsquigarrow J}\lambda^{|p|},
\qquad 0<\lambda\leq1
$$

默认 $\lambda=0.8$。路径必须严格沿轮次前进。

在当前候选边集合 $E_t$ 内归一化：

$$
\hat R_t(e)=
\frac{R_{raw}(e)}{\max_{e'\in E_t}R_{raw}(e')}
$$

若 $R_{raw}(e)=0$，说明在剩余轮次内该消息不可能进入真实终端决策路径，该边不参与最终选择。

------------------------------------------------------------------------

# 5. Structural Irreplaceability

不可替代性使用 sender-conditioned temporal interdiction。令：

$$
F(a_u^t,J;G)
$$

表示发送者当前状态 $a_u^t$ 到 $J$ 的折扣时序路径总质量。对候选边 $e_t=(a_u\rightarrow a_v,t)$：

$$
B_t(e)=
1-
\frac{F(a_u^t,J;G\setminus e)}
{F(a_u^t,J;G)}
$$

该定义回答：删除这条具体时序边后，当前发送者的信息还有多少其他合法路径可以到达最终决策。

分母不能使用“只从候选边出发的路径”，否则删除候选边后所有边都会得到 1；也不使用所有当前发送者的统一分母，因为首轮场景下会使 $B$ 与 $R$ 退化为同序指标。

------------------------------------------------------------------------

# 6. Attack-conditioned Message Survivability

拓扑相同的边在受到同一类 AiTM 攻击后，仍可能具有不同的攻击生存能力。LLM 不直接输出最终连续分数，而是对三个有明确量表的特征给出 0--4 整数：

- $C$（receptivity）：接收者在下一轮服从该边所附攻击指令的可能性；
- $P$（persistence）：初次服从后，攻击行为经过后续干净消息仍保留到最终相关报告的可能性；
- $A$（terminal acceptance）：攻击行为到达最终相关报告后，被终端 Judge 接受而非被并行干净报告覆盖的可能性。

代码验证三项均为 0--4 整数，然后计算：

$$
M(e)=
\left(\frac{C}{4}\right)^{\omega_C}
\left(\frac{P}{4}\right)^{\omega_P}
\left(\frac{A}{4}\right)^{\omega_A},
\qquad
\omega_C+\omega_P+\omega_A=1
$$

默认三项等权，即三项归一化评分的几何平均；任一关键环节为 0 时 $M(e)=0$。评分按接收者分组：每次向评分 Judge 展示原始任务、攻击目标、一个接收者的本轮独立输出、将同时进入其下一轮 inbox 的候选消息、当前/总轮次以及真实终端输入。实际攻击措辞在选边后才生成，因此评分 Judge 只知道准确的攻击目标，不虚构尚不存在的候选攻击文本。输入不使用未来消息，也不进行字符级硬截断。

Tree 与 Asymmetric-Tree 中，原始任务对所有 Agent 可见；通信拓扑只限制 Agent 间消息流。父节点首轮基于原题独立分析，不得假设尚未到达的子节点报告，后续轮次只结合实际收到的报告更新判断。

提示词规定真正不确定时使用合法中间值 2，禁止 `-1`、`null` 或 N/A。解析器独立保留每个合法字段；非法或缺失字段只触发一次局部纠正，不重算其他合法值。纠正后仍非法时，将该接收者组的 $M$ 标记为不可用，并以 `llm_structured_unavailable` 和诊断字段记录；最终边分数只使用可用的 $R$、$B$ 分量并重新归一化，不再填入虚假的中性分数。

------------------------------------------------------------------------

# 7. Edge Influence Score

最终边分数为：

$$
Score(e)=
\mathbb{1}[R_{raw}(e)>0]
\left(
w_R\hat R_t(e)
+w_BB_t(e)
+w_MM(e)
\right)
$$

其中：

$$
w_R+w_B+w_M=1
$$

当前默认三项等权。当 $M(e)$ 因评分输出持续非法而不可用时，代码从该边的可用分量集合中移除 $M$，并按剩余权重重新归一化；不会把缺失的 $M$ 当作 0 或 0.5。选择：

$$
e^*=\arg\max_{e\in E_t}Score(e)
$$

精确并列时使用实验随机种子进行可复现选择。

旧的 Agent 级 degree / betweenness / closeness 与历史通信启发式不再参与决策。

------------------------------------------------------------------------

# 8. AiTM Attack Execution

选定 $e^*=(a_u\rightarrow a_v,t)$ 后：

1. 只截获该边当前轮的一个消息实例；
2. 保持原发送者不变；
3. 使用既有 AiTM adversarial agent 生成恶意指令；
4. 将恶意指令附加到该消息；
5. 只把篡改消息交付给 $a_v$。

默认每道题最多一次攻击事件。其他未选消息保持原样。

------------------------------------------------------------------------

# 9. 实验记录与验证

每次选择记录：

- `current_round`、`remaining_rounds`、`temporal_decay`；
- `temporal_reachability_raw`、`temporal_reachability`；
- `sender_path_mass`、`removed_path_mass`、`irreplaceability`；
- `message_features`、`message_influence`、评分方法和诊断；
- 最终 `score`、候选边和选中边。

当前 DeepSeek-V3.2 tokenizer 的上下文窗口记录为 131072 tokens。所有实验性 LLM 调用统一预留最多 4096 个输出 tokens；协作消息与 $M$ 的评分输入不做字符级截断。若直接 API 调用返回 `finish_reason=length`，该响应不会作为完整结构化结果使用。

必要基线：

- No Attack；
- Random Edge：在完全相同的在线候选边中均匀随机；
- Online Fixed Target；
- Original fixed-victim AiTM。

必要消融：

- $w_R=0$；
- $w_B=0$；
- $w_M=0$；
- 不同 $\lambda$。

Shapley Flow 或逐边反事实攻击不进入当前在线算法；后续只作为昂贵的离线 teacher，用于检查轻量分数与真实边攻击效果是否一致。

------------------------------------------------------------------------

# 10. 核心定位

AIRA 不是新的消息篡改方法，而是 AiTM 上的边级攻击决策层：

> 在严格单次攻击预算下，同时判断当前消息能否按真实轮次传播到最终决策、该路径是否可替代，以及消息内容本身是否值得攻击。
