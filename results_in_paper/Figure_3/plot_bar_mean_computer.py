import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


baseline_mean = np.array([33448, 66256,  3400, 15400,  7904,  9136, 81640, 35512])/2


proposed_mean_gss = np.array([178, 316, 23, 122, 61, 65, 656, 627])
proposed_mean_reservoir = np.array([400, 422, 16, 75, 38, 51, 258, 76])
proposed_mean = 100*(proposed_mean_gss+proposed_mean_reservoir)/baseline_mean


# mean memory gss
# Task: 1  is  178.97631674766887
# Task: 2  is  316.8983762879529
# Task: 3  is  23.27248002719701
# Task: 4  is  122.56029241604409
# Task: 5  is  61.84859732283934
# Task: 6  is  65.4039343690828
# Task: 7  is  656.5300450804276
# Task: 8  is  627.6833375573766


# mean memory reservoir
# Task: 1  Mean number is  400.00272294077604
# Task: 2  Mean number is  422.3039306927983
# Task: 3  Mean number is  16.917628205128207
# Task: 4  Mean number is  75.37846767074183
# Task: 5  Mean number is  38.71588687411387
# Task: 6  Mean number is  51.86035090498633
# Task: 7  Mean number is  258.4904391660971
# Task: 8  Mean number is  76.34890992296978



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
means   = [100, proposed_mean[task_id]]        # 先 JT，再 Dual LS
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
ax.set_xticklabels(["Vanilla", "Dual LS"], fontsize=6)
ax.set_ylabel("Relative computational resource", fontsize=6)
ax.tick_params(axis="both", labelsize=6)
ax.set_ylim(0, 110)

fig.tight_layout()

fig.savefig(f"./plot_figures/compute_Task_{task_id+1}.pdf", dpi=300, bbox_inches="tight", pad_inches=0)

plt.show()
