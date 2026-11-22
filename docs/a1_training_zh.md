# A1 采样与更新说明

当前最小链路是：同一物理状态下双方选择动作，原生 MuJoCo 执行，统一命中／犯规结算，记录 learner 的真实选择，再做显式 PPO 更新并保存历史对手。`A1` 分支增加有界循环与一键离线动作库重建，仍是开发训练实现，不是已经训练成功的击剑策略。当前结果和剩余风险见 [进度说明](a1_readiness_zh.md)。

## 模块职责

| 模块 | 当前职责 |
| --- | --- |
| `sim/a1_env.py` | 双物理槽同步提交，原生积分，逐物理步结算；错误保留为无效局 |
| `sim/a1_policy.py` | 30 维观测编码、五技能掩码概率与价值网络 |
| `sim/a1_rollout.py` | 只记录真实选择，按实际时长累计回报，完整回合提交或整局放弃 |
| `sim/a1_selfplay.py` | 独立冻结快照、环境兼容性检查、单局采样、零更新命令行 |
| `sim/a1_ppo.py` | 校验完整 on-policy 数据后，显式执行一次全批 Adam 更新 |
| `sim/a1_train.py` | 有预算的完整回合自博弈循环、持久 Adam、逐批历史对手和 checkpoint |
| `sim/a1_prepare.py` | 从五份合法本地源数据重建并逐项匹配已验证 v5 数值指纹 |

`a1`、`a2` 是物理槽位；当前两边都运行 A1 家族策略，不是最终 A1 与连续关节 A2 的交叉对打。A2 尚未接入，不能把当前五技能接口当作 A2 的动作空间。

## 观测与动作

每侧观测包含自身 7 个关节角、7 个关节速度，对手剑尖及握点相对自身靶区的位置、对手剑尖速度，以及自身的 busy、当前技能和执行进度。空间量转换到观察者本地坐标。只使用现在的状态，不暴露对手技能、阶段、策略快照 ID 或未来轨迹。当前由仿真真值提供观测；实机感知和噪声尚未验证。

五个动作是 Guard、Stab、SlashLeft、ParryLeft、ParryRight。Guard 在就绪时表示等待一个 20 ms 决策周期；执行非零技能后，控制器保持该技能直至完成并回到合法入口。期间输出的强制 0 不属于新的 actor 决策。有限技能执行期间无法任意中断，这个限制必须在最终与 A2 的比较中说明。

网络为两层 64 单元 Tanh，actor 输出五个 logits，critic 输出标量。速度除以 `4 rad/s` 只是数值缩放，不是电机速度限制；当前仍使用同一原尺寸模型、候选力矩上限和代理碰撞几何。

## 变时长回报和探索

每个 transition 从自身实际选择开始，到再次允许自身选择或回合结束为止。一次等待通常为 0.02 s，一次技能可能持续约 25–30 s；不能给它们使用相同的“每决策步折扣”。当前 `gamma(dt)=exp(-dt/120 s)`，终点奖励也按其实际到达时间折扣。完整回合使用 Monte Carlo 回报（按秒的 lambda 为 1），不对正在执行的技能任意截断采样。

80 s 是本游戏的回合上限，超时结算为零收益平局，终局 bootstrap 为 0。数值、证据和控制故障不是超时或平局：整局暂存数据作废，记录错误，停止当前批次。双犯规双方 −1，因此不能用一侧奖励取反生成另一侧奖励。

如果就绪时均匀随机选五招，几乎总会立刻进入长技能，难以探索“先等对手，再启动格挡”。初始化因此用可学习偏置给非零技能总计约 `0.0049875` 的每 20 ms 启动概率，平均等待约 4 s，四个非零技能等分；所有技能仍可选。后续由网络学习是否等待、选什么和何时开始。默认随机采样；argmax 会让初始网络一直 Guard，不适合作为此初始化的评估模式。

## 快照与数据一致性

`FrozenSnapshotPool` 保存最多四个独立、无梯度的 CPU 快照。每局先选择对手并保持不变；learner 轮换物理槽位。初始池只有一个未训练快照，不能把它称为成熟的历史对手池。有界循环每次完成更新后保存新版本；这建立历史快照，不承诺短训后已经出现多样或高水平的对手。

快照身份绑定权重、编码器、技能库、数值配置、实际物理参数、场景配置和环境实现。不同模型／规则的数据不得混训。batch identity 在采样前固定；旧 log probability 和 value 在更新前用未变权重及原 action mask 重算，防止把旧策略的数据伪装成当前 on-policy 样本。对手数据不混入 learner 的批次。

## 零更新复现

在仓库根目录使用已安装 MuJoCo、NumPy、Torch 的 Python 环境。此次本地核查使用 Python 3.13.11、MuJoCo 3.4.0、Torch 2.7.0，CPU 单线程；没有新增依赖。第三方 ASE 源文件及派生库须按 [动作执行说明](a1_execution_zh.md) 在本地准备，不能假定普通 git clone 包含这些二进制。

新环境安装使用 `requirements-a1.txt`；CPU 安装和全部五份源文件链接见根目录 [README](../README.md)。已验证重建使用 NumPy 2.5.4。`sim.a1_prepare` 只有完成四阶段几何检查并与 v5 参考数组逐项匹配后才报告 `complete`；失败输出保留供排查，不应继续作为已验证动作库使用。

```bash
python -m sim.a1_selfplay \
  --bank outputs/a1_dual_parry_hold8_dt1ms_candidate_v5/library \
  --runtime-profile config/a1_runtime_rk4_candidate.json \
  --output outputs/a1_zero_update_new \
  --episodes 2 --seed 20261122
```

命令只采样，不更新权重；输出目录必须不存在。最多接受八个完整回合，每局上限 80 s，遇错立即保存并停止，不自动补局。此公开入口已通过接口测试，但尚未另行在原机械臂上重跑；本轮两局原生证据来自早期冻结诊断驱动。两者随机数流组织不同，因此同一 seed 不承诺重现早期两局逐动作相同的序列；早期精确复现应使用其冻结驱动。

每局保存观测及下一观测、真实动作与 mask、旧概率／价值、开始和结束时刻、回报及完整原生轨迹；另保存权重身份和环境快照。`sim.a1_ppo` 没有自动训练入口：由调用方把已提交的完整回合交给 `ppo_update_once`，明确传入采样前身份和兼容性元数据。

## 本轮一次更新的证据

`outputs/a1_zero_update_native_round1/` 保存两局原始采样；`outputs/a1_real_batch_optimizer_smoke_round1/` 保存转换来源、一次更新报告和 `one_update_candidate.pt`。该 checkpoint 明确标为技术测试，不是可用于最终展示的已训练策略。更新报告的 `checkpoint_written=false` 指更新模块本身不写文件；外层测试驱动另行保存了这个候选 checkpoint。

两局共 510 条 learner 决策；其中仅四次非零技能选择，主要样本是等待。一局平局、一局 learner 失分，没有 learner 获胜样本。一次更新通过只能说明损失、梯度、权重和快照隔离链路可用；是否学会观察对手、正确格挡和有效进攻，需要后续有界短训及独立冻结评估。

持续训练时应传入同一个显式 Adam 保存优化器状态，每批先采完当前策略的数据再更新；不要反复调用内部新建 Adam 来冒充连续训练，也不要把同一旧批次反复重贴新身份。当前一次全批更新是最小技术检查，并非已经调好的完整 PPO 训练配置。

本轮完整回归为 396 项测试全部通过，另通过 Python 编译和 `git diff --check`。复验命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -q
```

单元／集成测试通过不等于击剑能力评估；真实回合和一次更新的证据仍以上述独立输出为准。

## A1 分支的有界短训入口

从新准备的库开始，每次指定一个未存在的目录：

```bash
python -m sim.a1_prepare --source-dir outputs/ase_sources --output outputs/a1_prepared
python -m sim.a1_train --bank outputs/a1_prepared/library \
  --runtime-profile config/a1_runtime_rk4_candidate.json \
  --output outputs/a1_train_small --updates 2 --episodes-per-update 2 --seed 20261122
```

默认四局、两次更新；最多 8 次更新，每批 2／4／6／8 局，总数不能超过 32，每局上限 80 s。每批从当前冻结身份采完全部有效数据，再调用一次 PPO 更新；同一个 Adam 状态跨批保存。每局对手在开局前从历史快照池选择，局内不变，learner 的物理槽位轮换。未实现 checkpoint 自动恢复，不会无人值守扩大预算。

输出包含 `initial.pt`、`final_candidate.pt`、`snapshots/`、`update_01/` 等批次目录、源文件哈希和 `summary.json`。每局保存完整原生轨迹与决策样本，每批保存 `optimizer_report.json` 和新的候选 checkpoint。中途失败输出 `failure.json`，保留此前已完成更新及无效局，当前未完成批次不进入学习。

这次发布没有把 ASE 原数据、派生数组或旧 checkpoint 放进 Git；提交的是重建代码、配方、数值指纹、训练／测试代码和精简验证记录。既有两局及一次更新的详细本地证据不冒充新入口的验证，新增入口的实际运行结果以 [精简验证记录](a1_verification.json) 为准。

发布前已实际运行公开入口的一批两局和一次更新，得到 453 条真实决策，包含靶区失分和双方 −1 的双犯规，更新及快照保存完成。另从原始五源文件完整重建出与 v5 一致的数值曲线。本地完整测试为 411 项通过；无数据缓存的 Git 副本为 407 项运行、两个外部数据夹具跳过、无失败。尚未对更新后策略开展独立能力评估。
