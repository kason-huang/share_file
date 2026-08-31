# Calibra 设计哲学:按"指标设计逻辑 × 指标用途"重新梳理

本文档不以"4 阶段"或"19 analyzer"的结构切分 Calibra,而是按两个正交轴重新组织:**指标的设计逻辑(它在量什么、为什么这么设计)** × **指标的用途(它的输出喂给哪一层决策)**。和 `CALIBRA_ANALYSIS.md` 的"阶段视角 / 维度视角"形成第三层"指标语义视角"的递进。

---

## 核心设计逻辑:一套指标,服务三个决策层

Calibra 的所有指标都落在一个二维矩阵里:

- **设计逻辑轴**(行):指标在回答"数据的哪个方面有问题/什么性质"——从"记录本身可信吗"逐层上升到"动力学可学吗"。
- **用途轴**(列):指标的输出**喂给谁**——Calibra Score(打分)、integrity CI(信任门禁)、prune(精选)、review(复核排序)、policy 兼容、research。

关键洞察:**同一个指标可同时服务多个列**(如 LDLJ 既进 Score、又挂 integrity、又当 prune Stage1 阈值、又和 world_model surprise 交叉)。而 **Score 只挑了其中"轻量+可比+性质一致"的一小撮**,其余各有归宿。

---

## 指标按设计逻辑分组(7 组,从底层到高层)

### 第 1 组 · 采集层信任("记录本身可信吗")
**设计逻辑**:在谈运动好坏前,先确认信号没坏——时间戳一致吗、相机真在更新吗、帧没冻/没糊、关节标定没漂。这些是"信号源可信度",适合**二态门禁(block/inspect)**而非连续扣分。
- 时间戳/同步:timestamp jitter CV、dropout 率、action-obs 错位、camera-physics drift、camera lag
- 相机流:duplicate frame rate、camera freeze events、blurry fraction
- 标定:joint_offset_max_abs(calibration drift)
- 结构:action dropout、short episode fraction

### 第 2 组 · 运动学质量("动作平滑吗")
**设计逻辑**: jerk/加速度的不连续是控制缺陷的物理指纹。精妙处在**区分"真坏 vs 脚本化"**——高 spike + 低 vel_disc = 运动规划器采的(非缺陷);scripted-motion signature 检测器专门识别这类。
- LDLJ、jerk spike rate、velocity discontinuity rate、action-state divergence、motion collection signature

### 第 3 组 · 分布覆盖("分布广不广/有没有混策略")
**设计逻辑**: 用熵量宽度、用 PCA 量低秩、用 2-means 量策略多模态。覆盖≠多样性(熵量宽度=多样,PCA量空间填充=覆盖),所以分开测。
- 覆盖广度:action entropy、state entropy、PCA top-2 方差、episode 长度 CV/双峰
- 策略结构:trajectory diversity(2-means)、contact density、grasp events、phase balance(approach/contact/retract 占比)

### 第 4 组 · 动力学可学性("world model 能预测吗")— 三级递进
**设计逻辑**: 从线性 → 隐空间线性 → 训练神经网络,保真度递增、成本递增。这是 Calibra 的研究内核(验证手工指标=可学性代理)。
- **Level 1 线性**:transition_dynamics(拟合 `S_{t+1}=S_t+W·[S_t,A_t]+b`,报预测误差+转移熵)
- **Level 2 隐空间数值**:latent_dynamics(状态/转移体素冗余、Ridge R² 可预测性、dHSIC action-effect MI、可控性、独占新颖性)
- **Level 3 学习式**:world_model(训 RobotJEPA,per-episode surprise,learnability)

### 第 5 组 · 轨迹嵌入与影响力("哪条 episode 最有价值/最离群")
**设计逻辑**: 把变长轨迹压成定长向量,在嵌入空间里找离群和稀疏;influence 把 novelty+熵+接触相位合成"信息量"。
- ssl_embed(随机投影+时序聚合,余弦离群,全局稀疏度)
- influence(per-episode novelty kNN + 熵 + contact 相位)

### 第 6 组 · 接触/力觉("接触事件对不对")
**设计逻辑**: 力觉是接触类任务的直接信号,需要 force capability。force spike 用 MAD 稳健估,contact dropout 抓"该接触没接触"。
- force_torque(force spike MAD、contact dropout、per-episode force spike/contact density)

### 第 7 组 · policy 兼容("能不能直接微调某个 VLA")
**设计逻辑**: 不同 foundation model 有硬结构门槛(相机数、语言覆盖、chunk、频率、action dim),门控触发、不匹配零开销。
- gr00t / pi0 / openvla / octo 结构检查 + coverage/task_structure 的软 caveat

---

## 指标 × 用途 矩阵(谁喂谁)

| 指标组 | Calibra Score | integrity CI | prune Stage1 | prune Stage2 策略 | review 排序 | policy 兼容 | research |
|---|---|---|---|---|---|---|---|
| 1 采集信任 | 部分(jitter/dropout/action_dropout/short_ep 进分) | ✅ block/inspect | dropout/length 阈值 | — | anomaly | — | — |
| 2 运动质量 | ✅ smoothness 35 | motion-review(默认不挂) | ✅ jerk/vel_disc/LDLJ | — | quality risk | — | — |
| 3 分布覆盖 | ✅ coverage 25 + structure 15 | — | — | diversity(entropy) | coverage value | 软 caveat | — |
| 4 动力学可学 | ❌ 不进分 | — | — | ✅ energy/novelty/world-model | — | — | ✅ 核心 |
| 5 嵌入/影响 | ❌ 不进分 | — | — | ✅ novelty/influence | ✅ anomaly | — | — |
| 6 接触/力觉 | ❌ 不进分 | — | — | energy(部分) | — | — | — |
| 7 policy 兼容 | ❌ 不进分 | — | — | — | — | ✅ CompatibilityHint | — |

读法:**Score 只用第 1(部分)+ 2 + 3 组**;第 4–7 组全不进分,各有专门归宿(prune 策略 / review / policy / research)。

---

## Score 为何只挑一小撮(设计逻辑的延续)

把用途矩阵和"设计逻辑"合起来看,Score 的取舍就不是随意的:

- **第 4–6 组不进分**,因为它们**依赖可选能力**(force/torch)或**性质是定性**(contact density 是 INFO)或**成本高**(训 JEPA)——塞进统一分会破坏可比性。
- **第 1 组的相机类不进分**,因为它们是 block/inspect 二态,不适合连续扣分。
- **integrity gate** 把第 1+2 组(信任+运动)作为第 3 组(覆盖)拿满分的前提——即"低层不可信时,高层覆盖分自动失效",这是设计逻辑轴的**因果传递**在用途轴上的体现。

---

## 全项目的因果链(指标逻辑视角)

```
信号可信吗(组1) ──不可信就 block──► CI 挂
      │ 可信才往下
      ▼
动作平滑吗(组2) ──不平就扣 smoothness──► Score 降 + Stage1 剔除
      │
      ▼  integrity gate = (组1+组2 健康度) 打折组3
分布广/纯吗(组3) ──窄/混策略就扣 coverage+structure──► Score 降
      │
      ▼
动力学可学吗(组4) ──surprise×jerk 拆"损坏 vs 稀有"──► Stage2 策略选 coreset
      │
      ▼
哪条最有价值(组5) / 接触对不对(组6) ──► review 排序 + Stage2 多策略
      │
      ▼
能微调哪个 VLA(组7) ──► CompatibilityHint
```

---

## 一句话总结

**Calibra 不是"19 个 analyzer 的大杂烩",而是一条"从信号可信 → 运动平滑 → 分布覆盖 → 动力学可学 → 价值排序 → policy 兼容"的指标因果链**:低层指标是高层指标成立的前提(经 integrity gate 传递),同一指标可同时服务打分/CI/精选/复核/兼容多个用途,而 0–100 Score 只摘了链条最前段(组1部分+组2+组3)那一小撮"轻量可比"的指标做对外展示,其余指标在各自专门的决策层里发挥作用。

---

## 附:与 `CALIBRA_ANALYSIS.md` 三层视角的关系

| 视角 | 文档 | 切分轴 | 回答 |
|---|---|---|---|
| 阶段视角 | `CALIBRA_ANALYSIS.md` 第一节 | 顺序决策点 | "在哪一步淘汰谁" |
| 维度视角 | `CALIBRA_ANALYSIS.md` 第五节 | 评分维度 vs 流水线阶段 | "多样性是维度还是阶段" |
| 指标语义视角 | 本文档 | 设计逻辑 × 用途 | "指标量什么、喂给谁" |

三层递进:结构(怎么做)→ 概念边界(怎么分)→ 语义内核(为什么)。
