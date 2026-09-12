---
author: Research Proposal Draft
title: "AIRA: Adaptive Influence-aware AiTM Attack for LLM-based
  Multi-Agent Systems"
---

# AIRA: Adaptive Influence-aware AiTM Attack for LLM-based Multi-Agent Systems

## 1. 核心思想

现有 AiTM 攻击主要关注 Agent
间通信消息的拦截与篡改，但默认攻击者已经知道目标 victim agent。

本文提出 **AIRA（Adaptive Influence-aware AiTM Attack）**：

> 攻击者通过监听 Agent 间通信行为，逐步恢复 MAS 拓扑结构，推断 Agent
> 角色，并结合通信影响力评估每个 Agent
> 的攻击价值，从而自适应选择最具影响力的 victim agent，最后执行 AiTM
> 攻击。

核心流程：

$$
\text{Communication Observation}
\rightarrow
\text{Topology \& Role Inference}
\rightarrow
\text{Agent Influence Estimation}
\rightarrow
\text{Adaptive Victim Selection}
\rightarrow
\text{AiTM Attack}
$$

------------------------------------------------------------------------

# 2. Threat Model（攻击模型）

考虑一个 LLM-MAS：

$$
G=(A,E)
$$

其中：

-   $A=\{a_1,a_2,\dots,a_n\}$：Agent 集合；
-   $E$：Agent 之间的通信边。

每个 Agent 表示为：

$$
a_i=(LLM_i,Role_i,Memory_i,Tool_i)
$$

## 攻击者能力

攻击者：

-   可以监听 Agent 间通信消息：

$$
m_{ij}^{t}
$$

-   可以截获并修改通信内容：

$$
m_{ij}^{t}\rightarrow m_{ij}^{t'}
$$

-   可以将修改后的消息重新发送给目标 Agent。

攻击者不能：

-   修改模型参数；
-   访问 Agent 内部 memory；
-   修改 system prompt；
-   直接控制 Agent。

因此，攻击者属于通信层观察者和操纵者。

------------------------------------------------------------------------

# 3. Communication Observation Module

攻击者首先在 MAS 运行过程中通过有限观察窗口收集通信轨迹：

$$
H_T=\{m_{ij}^{1},m_{ij}^{2},...,m_{ij}^{T}\}
$$

每条消息包含：

$$
m_{ij}^{t}=(sender,receiver,content,time)
$$

通信历史用于后续：

1.  拓扑结构恢复；
2.  Agent 角色推断；
3.  影响力计算。

观察窗口在 MAS 仍有后续通信时结束。达到预设的信息充分条件（如观察到固定数量的通信边或消息）后，攻击者基于当前历史进行一次 victim 选择；选择后的攻击在该 Agent 下一次接收通信时执行。

------------------------------------------------------------------------

# 4. Topology-aware Agent Analysis

## 4.1 Communication Graph Reconstruction

根据通信行为恢复 MAS 图：

$$
\hat{G}=(A,\hat{E})
$$

如果观察到：

$$
a_i\rightarrow a_j
$$

则建立通信边：

$$
e_{ij}=1
$$

------------------------------------------------------------------------

## 4.2 Topology Importance

对于每个 Agent，计算其结构影响：

### Degree Centrality

表示通信活跃程度：

$$
C_d(a_i)
$$

### Betweenness Centrality

表示信息传播中转能力：

$$
C_b(a_i)
$$

### Closeness Centrality

表示与其他 Agent 的距离：

$$
C_c(a_i)
$$

综合得到：

$$
S_{topo}(a_i)
$$

具体计算时，将不同拓扑指标进行归一化后加权：

$$
S_{topo}(a_i)
=
w_1 C_d(a_i)
+
w_2 C_b(a_i)
+
w_3 C_c(a_i)
$$

其中：

$$
w_1+w_2+w_3=1
$$

表示不同拓扑特征的重要程度。

表示 Agent 在通信拓扑中的重要程度。

------------------------------------------------------------------------

# 5. Role-aware Agent Inference

攻击者无法直接访问 Agent 的 system prompt，因此根据通信行为推断角色。

## 5.1 Role Representation

对于 Agent $a_i$：

$$
X_i=[messages_i,communication_i]
$$

其中：

-   $messages_i$：历史消息；
-   $communication_i$：交互模式。

## 5.2 Role Classification

利用 LLM-based role classifier：

输入：

-   Agent 历史消息；
-   通信行为。

输出：

$$
P(Role_k|X_i)
$$

例如：

     Role      Probability
  ---------- -------------
   Planner            0.85
   Executor           0.10
   Verifier           0.05

得到：

$$
S_{role}(a_i)
$$

根据角色推断概率以及不同角色对任务流程的影响程度计算：

$$
S_{role}(a_i)
=
\sum_k P(Role_k|X_i)\cdot R_k
$$

其中：

- $P(Role_k|X_i)$ 表示 Agent 属于角色 $Role_k$ 的推断概率；
- $R_k$ 表示该角色对最终任务结果的重要程度。

表示角色重要性。

------------------------------------------------------------------------

# 6. Communication Influence Analysis

仅考虑拓扑和角色仍然不足，因此进一步分析 Agent 对最终任务的实际影响。

## 6.1 Message Propagation Influence

统计：

-   消息被引用次数；
-   被传播范围；
-   后续 Agent 使用情况。

得到：

$$
I_{prop}(a_i)
$$

## 6.2 Final Decision Influence

分析 Agent 输出是否进入最终决策：

$$
I_{final}(a_i)
$$

综合得到：

$$
S_{comm}(a_i)
$$

具体计算时：

$$
S_{comm}(a_i)
=
\lambda_1 I_{prop}(a_i)
+
\lambda_2 I_{final}(a_i)
$$

其中：

$$
\lambda_1+\lambda_2=1
$$

表示消息传播影响和最终决策影响的权重。

表示通信影响力。

------------------------------------------------------------------------

# 7. Agent Influence Score（核心模块）

融合三个因素：

$$
Score(a_i)=
\alpha S_{topo}(a_i)
+
\beta S_{role}(a_i)
+
\gamma S_{comm}(a_i)
$$

其中：

$$
\alpha+\beta+\gamma=1
$$

最终得到 Agent 攻击价值排名。

------------------------------------------------------------------------

# 8. Adaptive Victim Selection

攻击者只在仍有后续入站通信、且其输出仍可传播至最终决策的候选 Agent 集合 (V_T) 中进行选择。选择最高影响力 Agent：

$$
a_v=\arg\max_{a_i\in V_T} Score(a_i)
$$

选定 victim 后，在当前任务中保持该目标不变；多轮任务中的重新选择作为动态切换扩展处理。

区别：

## 原始 AiTM

$$
\text{Fixed Victim}
\rightarrow
\text{Attack}
$$

## AIRA

$$
\text{Observation}
\rightarrow
\text{Influence Ranking}
\rightarrow
\text{Victim Selection}
\rightarrow
\text{AiTM}
$$

------------------------------------------------------------------------

# 9. AiTM Attack Execution

选择 victim 后，保持 AiTM 原始攻击机制。

步骤：

## Step 1: Message Interception

截获：

$$
m_{sv}
$$

## Step 2: Adversarial Reasoning

攻击 Agent 根据：

-   原始消息；
-   攻击目标；
-   当前上下文；

生成攻击策略。

## Step 3: Message Tampering

生成：

$$
m'_{sv}
$$

## Step 4: Injection

发送修改后的消息：

$$
m'_{sv}\rightarrow a_v
$$

诱导：

-   错误推理；
-   错误决策；
-   任务失败。

------------------------------------------------------------------------

# 10. Dynamic Victim Switching（扩展）

多轮 MAS 中：

每轮重新计算：

$$
Score_t(a_i)
$$

动态调整攻击目标。

------------------------------------------------------------------------

# 11. 实验设计

## Baselines

-   No Attack
-   Random Victim AiTM
-   Original AiTM
-   MAST

## Ablation

### w/o Topology

去除拓扑信息。

### w/o Role

去除角色信息。

### w/o Communication Influence

去除通信影响信息。

------------------------------------------------------------------------

# 12. 论文贡献总结

## Contribution 1

发现现有 AiTM 存在 victim agent 预定义限制。

## Contribution 2

提出通信观察驱动的 Agent influence modeling：

结合：

-   topology information；
-   role information；
-   communication influence。

## Contribution 3

提出 Adaptive Victim Selection，将智能目标选择能力引入
AiTM，提高攻击效率和泛化能力。

------------------------------------------------------------------------

# 核心定位

本文不是提出新的 AiTM 攻击，而是在 AiTM 基础上增加：

**Attack Decision Intelligence Layer**

解决：

> 攻击者如何自动发现最值得攻击的 Agent。
