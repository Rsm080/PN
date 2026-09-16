# 受限空间机器人铣削势能网络（PN）复现

这是对论文《受限空间机器人铣削三维姿态规划：一种势能网络架构》中**可由论文公式独立复现的算法部分**的 Python 实现。

实现范围：

- 式（5）—（7）：平底刀、圆角刀、球头刀三维姿态流形；
- 式（11）、（14）—（21）：邻近度、硬/软势函数和四层势能网络；
- 式（22）—（30）：中心差分势场力、虚拟质量—阻尼动力学和 RK4 积分；
- 附录 A 的接口化约束：关节/驱动范围、碰撞间距、Jacobian 条件数和加工变形；
- 一个无需机器人专有模型即可运行的三维动态受限空间基准。

## 与原论文实验的边界

论文未随文给出 ABB IRB-6660/KUKA KR500 的完整运动学标定、V-HACD 碰撞体、刚度辨识矩阵、切削力原始数据和刀路文件。因此本工程能复现**算法结构与数值流程**，不能声称逐点复现论文实验曲线。`robot_constraints.py` 提供回调接口，拿到这些数据后可以直接接入。

## 运行

在本目录中执行：

```powershell
python run_demo.py
python -m unittest discover -s tests -v
```

如果系统的 `python` 命令不可用，可改用 Codex 工作区自带的 Python 可执行文件。

脚本按论文消融设置运行 E1（\(\mu_P=0\)）与 E2（\(\mu_P=5\)），且两者均使用 \(M=10,C=5\)。结果写入 `results/E1_mu0/` 和 `results/E2_mu5/`：

- `trajectory.csv`：路径位置、三个姿态参数、速度、势能和最小邻近度；
- `trajectory.npz`：便于后续 NumPy 分析的完整数组；
- `summary.txt`：可行性和数值摘要；
- `overview.png`：轨迹、势能和最小邻近度图。

根结果目录还包含 `comparison.csv` 和 `comparison.txt`，汇总两组的速度、加速度、jerk 的平均 range/SD。因为这里使用的是公开公式构造的替代基准而非论文未公开的原始机器人数据，这些数值用于验证流程，不应与论文表格逐项对齐。

## 代码结构

```text
pn_milling/
  geometry.py           # SE(3) 算子与姿态流形（式 5—7、附录 B）
  potentials.py         # 邻近度与硬/软势函数（式 11、15、17）
  network.py            # 四层势能网络（式 18—21）
  planner.py            # 势场动力学、中心差分、RK4（式 22—30）
  constraints.py        # 通用约束构造器
  robot_constraints.py  # 真实机器人模型的回调适配器
  plotting.py           # 无 Matplotlib 的结果绘图
```

## 最小用法

```python
import numpy as np
from pn_milling.constraints import box_group
from pn_milling.network import PotentialNetwork
from pn_milling.planner import PlannerConfig, plan_path

network = PotentialNetwork(
    [box_group("姿态范围", [-1, -1, -1], [1, 1, 1], 0.2)],
    performance_weight=0.0,
)

path = np.column_stack([np.linspace(0, 1, 101), np.zeros((101, 2))])
result = plan_path(
    path,
    feed_rate=0.1,
    q0=np.zeros(3),
    network=network,
    config=PlannerConfig(mass=10.0, damping=5.0),
)
```

## 数学与数值说明

1. 论文式（16）中二阶导数的符号有误。本实现按式（15）直接求导，正确形式为

   \[
   \sigma_h''(\mathcal H)=\frac{2\lambda(1-\mathcal H^3)}{\mathcal H^3}.
   \]

2. 论文硬约束在 \(\mathcal H\le 0\) 时严格返回 `inf`。有限差分若采样到 `inf` 会无法直接相减，因此规划器只在求梯度时把 `inf` 替换为一个很大的有限屏障值；势能评价本身仍保持论文定义。

3. `PotentialNetwork` 对同质元素先取均值，再在硬约束组和软约束组内做加权平均，最后用 \(\mu_P\) 融合，逐层对应式（18）—（21）。
