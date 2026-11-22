# A1 技能执行与物理验收

最新进展见 [A1 动作验证和训练环境进度](a1_readiness_zh.md)：显式 RK4／1 ms 和 Right25／Left30 已通过指定连续攻防及反击检查，开始新环境调试。下文保留早期候选的原始执行结果，不能混用不同版本证据。

A1 负责学习选择技能和启动时机，A2 负责学习连续关节指令。这里实现的是 A1 的有限技能执行层和训练前验收，不是已经训练好的策略。场景槽位 `a1`、`a2` 都可以运行独立的 A1 执行器；这不改变最终 A2 的连续动作空间。

## 本轮物理结果

保持原模型尺寸、1.080 m shinai、1.75 m 底座间距、靶区、碰撞掩码和临时力矩上限不变，使用 2 ms 物理步、10 ms 底层控制和 20 ms 决策周期。

后续 26 秒人工竖剑拨挡已通过当前剑尖运动 1 mm 后触发的连续两次 Stab 防守，含镜像、延迟开打与 1 ms 复验；不能把旧的提前启动正例冒充看到攻击后的反应。100 秒防守／反击／再防守链也合法完成，但反击首击落后 0.3 秒。横斩竖直 Left 在显式 1 ms／0.5 ms 配置中已有完整防守正例，但原 2 ms 仍失败，实际轨迹仍有数值敏感性；全局默认不变，完整 A1 未开放训练。最新定义、节拍版本和完整证据见 [A1 防守动作构建与验证](a1_defence_zh.md)；下表保留原 ASE 派生候选的执行与失败记录。

| 测试 | 结果 | 能支持的结论 |
| --- | --- | --- |
| 原五技能闭环执行 | 11 案全部完成：Guard、两侧各四个单技能、两侧各一条五动作衔接链；实现 hash 在运行期间不变 | 执行和实际 Guard 退出条件可用，不等于攻防有效 |
| Stab 对 Guard | 两侧均约 12.6767 s 合法首击，并完整返回 | 已有一条有效刺击正例 |
| 原 SlashLeft 对 Guard | 两侧均空挥 | 不能作为有效攻击交给 RL |
| 局部增强 SlashLeft 对 Guard | 最终 C2 四倍时长版本两侧均约 13.9112 s 合法首击、完整返回；其他四技能逐 bit 不变 | 增强横斩已有单技能攻击正例，尚未验收增强后的混合攻防链 |
| 两个原 ASE 派生拨挡对 Stab | 11 个方向／启动组合，0 个合法全窗成功；7 个合法失败、4 个物理无效 | 原防守候选不通过，不开放训练 |

右拨挡在 `1.0 s` 启动时，`10.650 s` 已真实碰剑，但仍于 `12.6767 s` 入靶，是“发生接触仍漏防”的合法反例。另四个提前拦截案例将攻击者 J2 推到假定限位，依赖额外限位约束，均判无效。精确复现显示该刺击在当前 J2 临时 15 N·m 上限下的静态重力余量不足约 1 N·m；碰剑后可能进入重力需求超过上限的姿态。不能增加上限或关闭限位检查来补办成功。

这些是固定 Guard／固定刺击的开发对照，不是自主反应、多方向覆盖或 self-play 成果。增强横斩实际峰速约 `3.891 rad/s`；目前遵循位姿优先阶段约定，仅记录速度，不宣称满足设计目标 `2 rad/s` 或实机约束。后续已按此诊断改变拨挡方向、接触持续时间和退出路径，未提高力矩上限；新动作仍须单独验收，不能继承原曲线的衔接结果。

本地完整证据：

- `outputs/a1_native_closed_clock4_round2/summary.json`：闭环、双侧单技能和衔接链。
- `outputs/a1_adapted_slash_c2_native_round1/summary.json`：最终增强横斩双侧原生复验、数值导出／重载和未改技能核对。
- `outputs/a1_closed_opportunities_round2/summary.json`、`outputs/a1_closed_opportunities_round3/summary.json`：全部 11 个拨挡组合及配对结果。
- `outputs/a1_right_parry_limit_audit_v1/summary.json`：J2 约束、重力需求、实际与参考状态及精确末步复现。

较早的闭环 round1 因运行中代码校验增强而保留为非稳定开发记录，不用于上述通过结论。所有详细输出只在本地，不是 GitHub 上的公共实验包。

## 动作接口

| ID | 候选技能 | 定义 |
| --- | --- | --- |
| 0 | Guard | 空闲时保持共同警戒；忙碌时继续当前动作，不取消或重启 |
| 1 | Stab | ASE 派生刺击及完整恢复 |
| 2 | SlashLeft | ASE 派生横斩及完整恢复；名称不保证攻击有效 |
| 3 | ParryLeft | 原库为 ASE 派生拨挡；另行替换时须标明人工配方及验证范围 |
| 4 | ParryRight | 原库为 ASE 派生候选；新防守版本使用明确的人工竖剑拨挡配方 |

每个非零技能是一段连续关节轨迹，不是一个静态 pose，也不是整场对打脚本。未来策略在允许决策时选择技能；执行器只接收该选择、自身时钟和自身编码器状态，不读取对手技能、未来脚本或比赛结果。

`sim/a1_skills.py` 保留先前几何候选的全部关节样本、时钟和来源段，以共享结点速度／加速度的五次 Hermite 曲线连接样本，并追加显式的五次关节返回段。曲线在指令空间从同一个七关节 Guard 出发并精确返回。新的插值、关键帧选取、空间适配及重定时都是派生加工，不能称为原始 ASE 专家动作或模仿学习成果。

`source_time_scale=4` 明确将原候选的源段时长放大四倍，独立返回段则按自身参考速度／加速度参数定时。它是当前物理可行性试验的参数，不是测得的电机能力。原时钟、解析参考峰速和加速度都保留在导出文件中；机器人和 shinai 尺寸没有改变。

## 进入和退出条件

`sim/a1_control.py` 不写 `qpos`，只产生底层位置控制目标。忙碌时仅 `0` 是合法的继续指令；其他选择不会改变正在执行的动作或时钟。

指令到达 Guard 不等于实际机械臂已经返回。执行器继续保持 Guard，直到实际最大关节误差不超过 `0.05 rad`、最大关节速度不超过 `0.10 rad/s`，且连续满足 `0.10 s`，才开放下一次技能选择。反馈断档超过 `0.021 s` 会重新累计稳定时长；名义动作结束后 `3 s` 仍未满足条件则试验无效，不能用晚到的反馈补办完成。这些是技能衔接的入口／出口条件，不是实机安全认证或全程速度限制。

正式比赛和技能验收都使用相同的 v2 全剑身入靶规则，没有根据技能 ID 或阶段给予得分、免伤或格挡成功。验收在首击后继续观察，覆盖显式关节返回、实际稳定等待和后续动作；计分不会因为进入 settling 阶段而关闭。

## 本地复现

先按 `config/ase_pilot_v2.json` 中固定的来源及 hash 准备五个合法使用的本地缓存。生成器不自动下载，也不加载不受信任的 pickle。旧的 `plan1_skills fetch` 只获取历史两个源文件，不能代替这五个缓存。

在仓库根目录运行，并为每次试验选择新的输出目录：

```bash
python -m sim.ase_pilot --config config/ase_pilot_v2_calibrated_candidate.json --output outputs/a1_pilot_new --ik
python -m sim.a1_skills --pilot-dir outputs/a1_pilot_new --source-time-scale 4 --output outputs/a1_bank_new
python -m sim.a1_validation --pilot outputs/a1_pilot_new --source-time-scale 4 --output outputs/a1_validation_new
```

最后一条运行 Guard 对照、两侧各四个单技能，以及两侧各一条完整衔接测试链。该测试链用于检验衔接，不是 A1 的固定比赛策略。`--no-transitions` 可以只运行单技能测试。以下另加成对拨挡开发采样：

```bash
python -m sim.a1_validation --pilot outputs/a1_pilot_new --source-time-scale 4 --output outputs/a1_opportunities_new --no-transitions --opportunities --parry-starts 0 1.5 3
```

原横斩空挥的修复是独立、可选的 `sim/a1_adaptation.py`：以原候选时钟 `3.45 s` 为中心、半宽 `1.35 s` 的局部平滑握点偏移，标称峰值为 `[+0.160, +0.0296, -0.0848] m`，保持剑轴和共同 Guard 不变。逐原始样本的 IK 分支进行幅度 continuation 后，重新检查完整相邻路径及实际 C2 曲线。原 q、时钟和任务目标不覆盖，适配数组和完整参数另存；它是人工开发适配的 ASE-derived 动作，不是原始专家横斩。

```bash
python -m sim.a1_adaptation --pilot-dir outputs/a1_pilot_new --source-time-scale 4 --output outputs/a1_bank_adapted_new
python -m sim.a1_validation --bank outputs/a1_bank_adapted_new --output outputs/a1_adapted_validation_new
```

`--bank` 按导出时的完整曲线执行，不能同时覆盖时钟或混用 `--pilot`。上面的增强库完整验证命令可检查新的混合衔接；本轮已完成的是增强横斩自身的双侧完整执行，不能继承原横斩的混合链结果。

库导出包含 `bank.json` 和 numeric NPZ，可通过 `load_a1_library()` 重新核验模型 hash、曲线、共同 Guard 和采样代理几何。试验保存场景、参考库、实际关节／控制／力矩轨迹、启动及完成事件、全部入靶记录和实现 hash。`shared_paths.npz` 的稠密样本不能代替完整 C2 曲线定义；物理回放应使用记录的实际状态。

派生数据与实验输出留在本地 `outputs/`，不自动公开第三方动作数据，也不自动提交或 push。

## 训练前的通过条件

单技能合法执行只证明动作能完成。进攻还必须在 Guard 配对对照中合法命中；防守必须在相同攻击命令、初始状态、物理设置和完整窗口下，先于基线命中发生真实拦截，并在整个窗口阻止对方有效剑身入靶。双方预期动作也必须实际被接受并完成。

还需检验邻近启动时刻、过早／过晚与错误方向反例、小扰动、连续攻防衔接，以及始终等待／始终攻击／始终防守等退化策略。有限技能库不能防住任意 A2 连续轨迹，必须记录盲区。通过这些前提后才构建可训练环境、共同观测和奖励，接入 A1 选择策略与 self-play 对手池。

所有当前导出保持 `admitted_to_training=false`、`may_start_RL=false`。详见[整体攻防验收要求](skill_validation_zh.md)。
