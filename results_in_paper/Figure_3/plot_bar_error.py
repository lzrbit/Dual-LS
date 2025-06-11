import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

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
# 2.  读取数据（n_tasks × 3）
# -------------------------------------------------------------
proposed = np.load(data_dir / "mr_clser_combined.npy")[task_id]
JT       = np.load(data_dir / "mr_joint_combined.npy")[task_id]
vanilla  = np.load(data_dir / "mr_vanilla_combined.npy")[task_id]

prop_mean, prop_up, prop_low = proposed
jt_mean,   jt_up,   jt_low  = JT
vanilla_mean,   vanilla_up,   vanilla_low  = vanilla

# -------------------------------------------------------------
# ❶ Dual LS vs JT
# -------------------------------------------------------------
means   = [jt_mean, prop_mean]          # 先 JT，再 Dual LS
yerr_up = [jt_up,   prop_up]
yerr_dn = [jt_low,  prop_low]


x_pos   = [0, 0.45]   # 从 1 → 0.45，中心更近
bar_w   = 0.23        # 略加宽

fig, ax = plt.subplots()
ax.bar(
    x_pos, means,
    yerr=[yerr_dn, yerr_up],
    capsize=3,
    width=bar_w,
    color=[colors["JT"], colors["proposed"]],  # 先红，后蓝
    edgecolor="black", linewidth=0.3, alpha=0.75,
    error_kw={"elinewidth": 0.5, "capthick": 0.5}
)

ax.set_xticks(x_pos)
ax.set_xticklabels(["JT", "Dual LS"], fontsize=6)  # 标签顺序对应
ax.set_ylabel("Missing rate error/%", fontsize=6)
ax.tick_params(axis="both", labelsize=6)

# set the y axis limit
ax.set_ylim(0, 11)  # 0.5% 误差
fig.tight_layout()
fig.savefig(f"./plot_figures/error_A_Task_{task_id+1}.pdf",
            dpi=300, bbox_inches="tight", pad_inches=0)


# -------------------------------------------------------------
# ❷ Dual LS vs Vanilla
# -------------------------------------------------------------
means   = [vanilla_mean, prop_mean]     # 先 Vanilla，再 Dual LS
yerr_up = [vanilla_up,   prop_up]
yerr_dn = [vanilla_low,  prop_low]

fig, ax = plt.subplots()
ax.bar(
    x_pos, means,
    yerr=[yerr_dn, yerr_up],
    capsize=3,
    width=bar_w,
    color=[colors["JT"], colors["proposed"]],  # 仍用同一调色板：红+蓝
    edgecolor="black", linewidth=0.3, alpha=0.75,
    error_kw={"elinewidth": 0.5, "capthick": 0.5}
)

ax.set_xticks(x_pos)
ax.set_xticklabels(["Vanilla", "Dual LS"], fontsize=6)
ax.set_ylabel("Missing rate error/%", fontsize=6)
ax.tick_params(axis="both", labelsize=6)


fig.tight_layout()
fig.savefig(f"./plot_figures/error_B_Task_{task_id+1}.pdf",dpi=300, bbox_inches='tight',pad_inches=0 )



plt.show()
