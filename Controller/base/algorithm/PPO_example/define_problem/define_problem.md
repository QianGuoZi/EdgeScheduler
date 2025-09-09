# 任务调度与资源分配优化模型

本文档旨在对一个联合调度问题进行数学建模，其核心目标是在满足任务带宽需求的前提下，尽可能地使物理资源（CPU、内存、带宽）的负载更加均衡。

#### **1. 问题建模与拓扑**

##### **1.1 物理拓扑 (Physical Topology)**

*   **物理节点 (P)**: 系统的基础计算单元集合。
    *   每个节点 `p ∈ P` 拥有固定的 CPU 资源 `CPU_P(p)` 和内存资源 `RAM_P(p)`。
*   **物理链路 (R)**: 连接物理节点的网络路径集合。
    *   每条链路 `r ∈ R` 拥有固定的带宽资源 `BW_R(r)`。

##### **1.2 任务拓扑 (Task Topology)**

*   **任务节点 (N)**: 作业的基本计算任务集合。
    *   每个任务 `n ∈ N` 需要 `CPU_N(n)` 的 CPU 资源和 `RAM_N(n)` 的内存资源。
*   **任务链路 (V)**: 任务节点之间的数据传输需求集合，构成一个 P2P 网络。
    *   每条任务链路 `v ∈ V` 有一个最小带宽需求 `BW_min(v)` 和一个最大带宽需求 `BW_max(v)`。
    *   实际分配给链路 `v` 的带宽 `B(v)` 是一个决策变量，必须满足 `BW_min(v) ≤ B(v) ≤ BW_max(v)`。
    *   **注意**: 若一条任务链路所连接的两个任务节点被映射到同一个物理节点上，则该链路不消耗物理带宽资源。

#### **2. 决策变量与约束条件**

##### **2.1 决策变量**

1.  **任务-节点映射 (`X_pn`)**: 一个二进制变量。
    *   `X_pn = 1` 表示任务节点 `n` 被部署在物理节点 `p` 上。
    *   `X_pn = 0` 表示其他情况。

2.  **任务-链路映射 (`Y_vr`)**: 一个二进制变量。
    *   `Y_vr = 1` 表示任务链路 `v` 的数据流经过了物理链路 `r`。
    *   `Y_vr = 0` 表示其他情况。

##### **2.2 约束条件**

1.  **唯一性约束**: 每个任务节点必须且只能部署在一个物理节点上。
    \[ \forall n \in N, \sum_{p \in P} X_{pn} = 1 \]

2.  **CPU 容量约束**: 每个物理节点承载的所有任务的 CPU 需求总和不能超过该节点的 CPU 上限。
    \[ \forall p \in P, \sum_{n \in N} (X_{pn} \times \text{CPU}^N(n)) \le \text{CPU}^P(p) \]

3.  **内存容量约束**: 每个物理节点承载的所有任务的内存需求总和不能超过该节点的内存上限。
    \[ \forall p \in P, \sum_{n \in N} (X_{pn} \times \text{RAM}^N(n)) \le \text{RAM}^P(p) \]

4.  **带宽容量约束**: 每条物理链路上承载的所有任务链路的带宽需求总和不能超过该链路的带宽上限。
    \[ \forall r \in R, \sum_{v \in V} (Y_{vr} \cdot B(v)) \le \text{BW}^R(r) \]

#### **3. 优化目标函数**

优化目标是最小化一个由**负载均衡度 (L)** 和**带宽满足度 (D_BW)** 构成的加权函数。

##### **3.1 资源利用率**

*   **CPU 利用率**:
    \[ U^{\text{CPU}}(p) = \frac{\sum_{n \in N} (X_{pn} \cdot \text{CPU}^N(n))}{\text{CPU}^P(p)} \]
*   **内存利用率**:
    \[ U^{\text{RAM}}(p) = \frac{\sum_{n \in N} (X_{pn} \cdot \text{RAM}^N(n))}{\text{RAM}^P(p)} \]
*   **带宽利用率**:
    \[ U^{\text{BW}}(r) = \frac{\sum_{v \in V} (Y_{vr} \cdot B(v))}{\text{BW}^R(r)} \]

##### **3.2 负载均衡度 (L)**

负载均衡度通过计算各类资源利用率的**标准差**来衡量，值越小表示资源分配越均匀。

*   **平均利用率**:
    \[ \overline{U^{\text{CPU}}} = \frac{1}{|P|} \sum_{p \in P} U^{\text{CPU}}(p) \]
    \[ \overline{U^{\text{RAM}}} = \frac{1}{|P|} \sum_{p \in P} U^{\text{RAM}}(p) \]
    \[ \overline{U^{\text{BW}}} = \frac{1}{|R|} \sum_{r \in R} U^{\text{BW}}(r) \]
*   **各项负载**:
    \[ L^{\text{CPU}} = \sqrt{\frac{1}{|P|} \sum_{p \in P} (U^{\text{CPU}}(p) - \overline{U^{\text{CPU}}})^2} \]
    \[ L^{\text{RAM}} = \sqrt{\frac{1}{|P|} \sum_{p \in P} (U^{\text{RAM}}(p) - \overline{U^{\text{RAM}}})^2} \]
    \[ L^{\text{BW}} = \sqrt{\frac{1}{|R|} \sum_{r \in R} (U^{\text{BW}}(r) - \overline{U^{\text{BW}}})^2} \]
*   **综合负载均衡度**:
    \[ L = w_1 \times L^{\text{CPU}} + w_2 \times L^{\text{RAM}} + w_3 \times L^{\text{BW}} \]
    其中 `w1, w2, w3` 为权重系数，且 `w1 + w2 + w3 = 1`。

##### **3.3 带宽需求满足度 (D_BW)**

该指标衡量分配给任务链路的带宽满足其需求的程度，值越大表示满足得越好。

\[ D_{\text{BW}} = \frac{1}{|V|} \sum_{v \in V} \delta(v) \]
其中，
\[ \delta(v) = \begin{cases} 1 & \text{if } \text{BW}_{\text{max}}^v = \text{BW}_{\text{min}}^v \\ \frac{B(v) - \text{BW}_{\text{min}}^v}{\text{BW}_{\text{max}}^v - \text{BW}_{\text{min}}^v} & \text{otherwise} \end{cases} \]

##### **3.4 全局优化目标**

最终的优化目标是最小化以下函数：

\[ \text{Minimize:} \quad \gamma_1 L - \gamma_2 D_{\text{BW}} \]

其中 `γ1, γ2` 为权重系数，且 `γ1 + γ2 = 1`。这个目标函数旨在通过调整权重，在**负载均衡**和**带宽满足度**之间找到最佳平衡点。

---