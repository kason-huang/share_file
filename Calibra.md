# Calibra 项目分析:逻辑、价值与 Analyzer 实现

本文档是对 Calibra 项目的完整分析,涵盖**逻辑架构、核心价值、已实现的 Analyzer** 三个维度。

---

## 一、项目逻辑:一条从"诊断"到"精选训练集"的流水线

Calibra 是一个面向**机器人模仿学习(imitation learning)数据集**的质量诊断与数据精选工具。它的核心论点是:机器人数据集里有两类浪费 GPU 的问题——**坏 episode(抖动、掉帧、同步错)** 和 **冗余 episode(同一行为的近似重复演示)**——而 Calibra 同时把两者解决。

整个逻辑链由 4 个阶段组成(`README` 的 pipeline 表),在 `calibra analyze` 这一条命令里被串成一份报告:

| 阶段 | 回答的问题 | 落地命令 |
|---|---|---|
| 1. Integrity 完整性 | 这个数据集能信吗? | `calibra integrity` |
| 2. Quality 质量 | 哪些 episode 是干净的? | `calibra audit` |
| 3. Coverage 覆盖 | 哪些 episode 是独特的? | `calibra review` |
| 4. Select 精选 | 只留真正重要的 | `calibra prune` |

**关键技术逻辑链**(读 `analyze.py`/`pipeline.py`/`score.py`/`strategy.py`/`pruning.py` 后梳理):

1. **Ingestion 层**:多格式适配器(hdf5 / lerobot / rlds / isaac_lab / mcap)把异构数据集归一化为一个 `EpisodeBatch`,并暴露 `capabilities`(能力标签,表明该数据集有没有视觉/力矩/本体感觉等)。

2. **Pipeline 编排**(`pipeline.py`):持有一个 `Analyzer` 列表,逐个跑,每个 analyzer 若 `requires` 的能力集不在 `batch.capabilities` 里就**跳过**而非报错;结果汇总成一份 `DiagnosticReport`。报告里还带 `config_hash`——基于 Calibra 版本 + policy + analyzer 版本集合算出的指纹,使两份报告"可比较"前能先确认是同一套分析逻辑。

3. **Calibra Score**(`score.py`):把诊断报告压成一个 0–100 分。四个维度带权重:**时间稳定性 25 / 控制平滑度 35 / 覆盖多样性 25 / 任务结构 15**。一个关键设计是 **integrity gate**:覆盖与结构分数要乘以一个门控因子(时间+平滑度的均值),防止"高熵噪声"在烂数据集上骗到高分。

4. **Regime 诊断**(`strategy.py`):从报告里抽出 spike/dropout/disc 等噪声标量,把数据集判为 **LOW / MODERATE / HIGH NOISE** 三种 regime。不同 regime 给出不同的 `CoresetSelector` 推荐配置——比如高噪声下质量阈值要收紧、低噪声下要放鬆以免误删稀有行为。阈值是**在 ALOHA/DROID-100/PushT 三个数据集上消融标定**的,代码里明确标注"随更多数据集积累待修订"。

5. **CoresetSelector**(`pruning.py`):两阶段精选——
   - **Stage 1 质量过滤**:按 jerk spike / 速度不连续 / 掉帧 / LDLJ / 长度阈值剔除坏 episode。其中有个很巧妙的 **contact-aware** 机制:`vel_disc` 可能来自控制噪声,也可能来自接触事件(抓取时的方向突变,但无明显 jerk);用 `mean_disc/mean_spike` 的比值区分二者并**动态放宽** vel_disc 阈值(PushT real 比值 24→3× 放宽,ALOHA 比值 1.9→不动)。
   - **Stage 2 多样性选择**:greedy k-center(farthest-point sampling)最大化最小两两行为距离,可选 GPU(PyTorch CUDA/MPS)加速,大数据集走 `ApproximateCoresetSelector` 的 MiniBatch 锦标赛(O(N×B))。除默认 diversity 策略外还支持 novelty / influence / energy / world-model 四种 Stage 2 策略。

整套逻辑形成一个闭环:**诊断 → 打分 → 判 regime → 按 regime 配参精选 → 输出可直接训练的 coreset 索引**。

---

## 二、项目价值

1. **降本:最多减少 75% 训练数据而不掉性能**。README 实测:PushT 保留 25% 即达全量数据 99.5% 性能;DROID-100 保留 75% 反超全量 +3%;三个数据集 × 三种 policy 族(BC-MLP/ACT/Diffusion)在 30% 保留率下平均比随机选 **+24.5%**。直接省 GPU 与采数成本。

2. **在训练前就拦住坏数据**,而非训完才发现。`calibra integrity` 一条命令给出 trust 报告(camera freeze、blurry、timestamp jitter/dropout、calibration drift 等),把"这批数据能不能信"前置。

3. **不引入新指标,而是把已有 analyzer + Score + regime 自适应 coreset 选择器组合成一屏叙事**(`analyze.py` docstring 原话)。对设计合作伙伴(design partner)友好:一条命令即得 trust + quality + coverage + 推荐训练集。

4. **科学严谨、可复现**:阈值标定基于消融、report 带 `config_hash` 可比对、`integrity gate` 防刷分、明确标注"启发式起点而非验证过的 retention curve"——诚实声明局限。

5. **开放生态**:多格式适配、policy-conditional 的 VLA 兼容性检查(GR00T/π0/OpenVLA/Octo)、HuggingFace Space 在线 demo、可嵌入 dataset card 的 badge。

---

## 三、已实现的 Analyzer(共 19 个)

所有 analyzer 都继承 `calibra/analyzers/base.py` 的 `Analyzer` 抽象类,实现 `analyze(batch, policy_family) -> AnalyzerResult`(含 `RiskFlag`、可选 `CompatibilityHint`、`raw_metrics`),声明 `name`、`requires`(能力集)、`version`。

### A. 默认始终运行的诊断 analyzer(`pipeline._default_analyzers`,10 个)

| # | 文件 | 类名 | 作用 |
|---|---|---|---|
| 1 | `temporal.py` | `TemporalAnalyzer` | 时间稳定性:时间戳抖动 CV、掉帧、相机-主时钟 lag、action-obs 错位、相机-物理 drift、**action dropout**(action 近零但 state 仍在动)。含 `calibrate_drift_thresholds()` 经验调参助手 |
| 2 | `smoothness.py` | `ControlSmoothnessAnalyzer` | 控制平滑度:LDLJ、jerk spike rate、速度不连续率、action-state 散度、**脚本化运动签名检测器**(高 spike+低 vel_disc = 运动规划器采的,非缺陷) |
| 3 | `coverage.py` | `CoverageEntropyAnalyzer` | 覆盖:action/state 边际 Shannon 熵、PCA 方差集中度、episode 长度分布与双峰提示。Diffusion/ACT 兼容 hint |
| 4 | `task_structure.py` | `TaskStructureAnalyzer` | 任务性质(非质量):接触密度、抓取事件、轨迹多模态(2-means)、短 episode IQR 离群;自动探测夹爪维度。Diffusion/ACT/Transformer hint |
| 5 | `phase_balance.py` | `PhaseBalanceAnalyzer` | approach/contact/retract 相位占比;contact 相 <10% 是 BC loss 修不了的结构缺陷 |
| 6 | `influence.py` | `InfluenceAnalyzer` | 每 episode 影响力分数(novelty kNN + 熵 + contact 相位),识别最有信息量的演示 |
| 7 | `transition_dynamics.py` | `TransitionDynamicsAnalyzer` | 拟合线性前向动力学 `S_{t+1}=S_t+W·[S_t,A_t]+b`,报每 episode 预测误差与转移熵 |
| 8 | `latent_dynamics.py` | `LatentDynamicsAnalyzer` | World-Model Observability Phase 1:状态/转移空间拓扑覆盖与冗余、Ridge 可预测性 R²、因果 action-effect MI(dHSIC)、action 可控性、每 episode 独占新颖性 |
| 9 | `ssl_embed.py` | `SSLTrajectoryEmbedderAnalyzer` | 随机投影+时序聚合把变长轨迹嵌入定长向量,余弦距离检测行为离群与全局稀疏度 |
| 10 | `force_torque.py` | `ForceTorqueContactAnalyzer` | 力/力矩与接触传感器:力冲击 spike(MAD)、接触 dropout、每 episode 力 spike/接触密度 |

### B. Integrity 专用 analyzer(由 `calibra analyze` 的 `_combined_analyzers` 额外并入,4 个)

| # | 文件 | 类名 | 作用 |
|---|---|---|---|
| 11 | `duplicate_frame.py` | `DuplicateFrameAnalyzer` | 单帧级近重复帧(像素活跃度低于阈值,掉帧/传感器停滞),带 bootstrap CI |
| 12 | `camera_freeze.py` | `CameraFreezeAnalyzer` | 连续近同帧的**持续 run**(相机/驱动卡死,与单帧信号区分) |
| 13 | `blur.py` | `BlurAnalyzer` | Laplacian 方差做依赖无关的锐度代理,IQR 离群检测比绝对阈值更鲁棒 |
| 14 | `calibration_drift.py` | `CalibrationDriftAnalyzer` | 静止(hold)帧里 command 与观测关节态的逐电机系统偏移——LeRobot #3758 那类 stale leader/follower 标定漂移 |

### C. Policy-conditional VLA 兼容性 analyzer(仅当 `policy_family` 匹配时启用,4 个)

| # | 文件 | 类名 | 目标模型与检查项 |
|---|---|---|---|
| 15 | `gr00t.py` | `GR00TCompatibilityAnalyzer` | NVIDIA GR00T N1:视觉/语言标注/chunk-length/控制频率 15–120Hz/action dim 7/8/14/16/Isaac Sim 相机-物理 drift |
| 16 | `pi0.py` | `Pi0CompatibilityAnalyzer` | Physical Intelligence π0(flow-matching):chunk 50、10–100Hz、7/14 维、轨迹平滑 LDLJ>−15(flow 对 jerk 敏感) |
| 17 | `openvla.py` | `OpenVLACompatibilityAnalyzer` | Stanford OpenVLA:单相机、100% 语言覆盖、≥10 步、3–30Hz、7 维、256-bin 离散裕度(熵≥2 bits/dim) |
| 18 | `octo.py` | `OctoCompatibilityAnalyzer` | Berkeley Octo:1–2 相机、语言/goal、window=4、5–100Hz、7/14 维、≥50 episodes |

### D. 可选高阶 analyzer(需显式开启,1 个)

| # | 文件 | 类名 | 作用 |
|---|---|---|---|
| 19 | `world_model.py` | `WorldModelConsistencyAnalyzer` | 训练轻量 RobotJEPA(PyTorch),报每 episode "surprise"——world-model 可学性的离线代理;用 surprise×jerk 交叉区分**损坏 episode**(高 surprise+高 jerk) vs **新颖 episode**(高 surprise+低 jerk)。`Pipeline(world_model=True)` 启用,无 torch 时优雅跳过 |

### 跨 analyzer 的复用关系

- `_bootstrap_ci` 定义在 `temporal.py`,被 duplicate_frame / smoothness / task_structure / phase_balance / coverage 复用;
- `compute_visual_activity`(在 `calibra.temporal.drift`)被 duplicate_frame / camera_freeze / gr00t / temporal 共用;
- **组合调用**:`pi0` 调 `ControlSmoothnessAnalyzer` 取 LDLJ;`influence` 调 phase_balance / task_structure 取 contact 比例与夹爪维度;`pruning.py` 的 energy / world-model / novelty 策略直接消费 transition_dynamics / world_model / latent_dynamics 的 `raw_metrics`。

---

## 四、数据流总览

```
异构数据集 (hdf5/lerobot/rlds/isaac_lab/mcap)
        │  ingestion 层归一化
        ▼
   EpisodeBatch (+ capabilities 能力标签)
        │
        ▼
   Pipeline (按 requires 跳过缺失能力的 analyzer,逐个跑)
        │  ┌─────────────────────────────────────────┐
        │  │ 19 个 Analyzer (无状态, batch in → AnalyzerResult out)│
        │  └─────────────────────────────────────────┘
        ▼
   DiagnosticReport (RiskFlags + CompatibilityHints + raw_metrics + config_hash)
        │
        ├──────────────────┬──────────────────┬──────────────────┐
        ▼                  ▼                  ▼                  ▼
   Calibra Score      Regime 诊断        integrity 评分      Policy 兼容性
   (0-100, 4维度)    (LOW/MOD/HIGH)     (trust 报告)       (GR00T/π0/...)
        │                  │
        └──────┬───────────┘
               ▼
        CoresetSelector (Stage1 质量过滤 + Stage2 greedy 最大覆盖)
        │   · contact-aware 阈值动态放宽
        │   · diversity/novelty/influence/energy/world-model 多策略
        ▼
   PruningResult → coreset 索引 (可直接喂训练)
```

---

**一句话总结**:Calibra 的价值是"在训练前用一套 19 个无状态诊断 analyzer 把数据集打分、判噪声 regime、再按 regime 自适应地用质量过滤 + greedy 最大覆盖精选出 coreset",实现在 4 个公开数据集上最多省 75% 数据而不掉性能。Analyzer 体系是它的地基,Score/Regime/CoresetSelector 是建在其上的决策层。
