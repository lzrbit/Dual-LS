import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


baseline_max = np.array([33448, 66256,  3400, 15400,  7904,  9136, 81640, 35512])
proposed_max_gss = np.array([1000, 983, 379, 743, 592, 599, 930, 790])
proposed_max_reservoir = np.array([1000, 666, 34, 121, 60, 69, 386, 141])
proposed_max = 100*(proposed_max_gss+proposed_max_reservoir)/baseline_max



# max memory gss
# Task: 1  Max number is  1000
# Task: 2  Max number is  983
# Task: 3  Max number is  379
# Task: 4  Max number is  743
# Task: 5  Max number is  592
# Task: 6  Max number is  599
# Task: 7  Max number is  930
# Task: 8  Max number is  790


# max memory reservoir
# Task: 1  Max number is  1000
# Task: 2  Max number is  666
# Task: 3  Max number is  34
# Task: 4  Max number is  121
# Task: 5  Max number is  60
# Task: 6  Max number is  69
# Task: 7  Max number is  386
# Task: 8  Max number is  141


# -------------------------------------------------------------
# 0.  参数
# -------------------------------------------------------------
task_id  = 2                       # 选取任务
data_dir = Path("./processed_data")
colors   = {"proposed": "#00008B", "JT": "#8B0000"}

# -------------------------------------------------------------
# 1.  Nature‑style 设置（缩窄宽度）
# -------------------------------------------------------------
mpl.rcParams.update({
    "font.family"      : "sans-serif",
    "font.sans-serif"  : ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset" : "dejavusans",
    "figure.dpi"       : 300,
    "figure.figsize"   : (1.2, 2.0),      # ← 更窄
    "axes.linewidth"   : 0.6,
    "axes.spines.top"  : True,
    "axes.spines.right": True,
    "legend.frameon"   : False,
    "xtick.direction"  : "in",
    "ytick.direction"  : "in",
    "xtick.major.size": 1,   # ← 主刻度线长度（默认 3.5–4.0）
    "ytick.major.size": 1,
    "xtick.major.width": 0.3,  # ← 主刻度线粗细
    "ytick.major.width": 0.3,
})



# -------------------------------------------------------------
# ❶ Dual LS vs JT
# -------------------------------------------------------------
means   = [100, proposed_max[task_id]]        # 先 JT，再 Dual LS
# ---------------------------------------------
# ❶ 重新定义 x 位置和柱宽
# ---------------------------------------------
x_pos   = [0, 0.45]   # 从 1 → 0.45，中心更近
bar_w   = 0.25        # 略加宽

fig, ax = plt.subplots()
bars = ax.bar(
    x_pos, means,
    capsize=3,
    width=bar_w,
    color=[colors["JT"], colors["proposed"]],
    edgecolor="black", linewidth=0.3, alpha=0.75,
    error_kw={"elinewidth": 0.5, "capthick": 0.5}
)

# ---------------------------------------------
# 顶端数值（不变）
# ---------------------------------------------
margin = 0.01
for bar in bars:
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        h + margin,
        f"{h:.3g}%",
        ha="center", va="bottom", fontsize=6
    )

ax.set_xticks(x_pos)
ax.set_xticklabels(["JT", "Dual LS"], fontsize=6)
ax.set_ylabel("Relative memory", fontsize=6)
ax.tick_params(axis="both", labelsize=6)
ax.set_ylim(0, 110)

fig.tight_layout()

fig.savefig(f"./plot_figures/memory_Task_{task_id+1}.pdf", dpi=300, bbox_inches="tight", pad_inches=0)

plt.show()
