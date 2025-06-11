
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ------------------------------------------------------------------
# 1.  Figure‑wide styling  (Nature: Helvetica, thin axes, in‑ticks)
# ------------------------------------------------------------------
mpl.rcParams.update({
    # fonts
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    # figure / axes
    "figure.dpi": 300,
    "figure.figsize": (3.54, 2.0),    # 90 mm × 51 mm  (single‑column)
    # Uncomment the next line for double‑column width:
    # "figure.figsize": (7.48, 3.0),  # 190 mm × 76 mm
    "axes.linewidth": 0.6,
    "legend.frameon": False,
    "axes.spines.top": True,
    "axes.spines.right": True,

})

# ------------------------------------------------------------------
# 2.  Data loading
# ------------------------------------------------------------------
buffer_name = "gss"          # "reservoir", "gss", ...
task_id = 3
file_path = pathlib.Path(f"./extracted_csv/{buffer_name}_Task_{task_id}.csv")
df = pd.read_csv(file_path)
print("Loaded file:", file_path.as_posix())
buffer_nb_gss = df.to_numpy()                # shape (N, 1) expected


buffer_name = "reservoir"          # "reservoir", "gss", ...
file_path = pathlib.Path(f"./extracted_csv/{buffer_name}_Task_{task_id}.csv")
df = pd.read_csv(file_path)
print("Loaded file:", file_path.as_posix())
buffer_nb_reservoir = df.to_numpy()                # shape (N, 1) expected



if buffer_nb_gss.shape[0] != buffer_nb_reservoir.shape[0]:
    print("Error: The two buffers have different lengths.")
    # fill 0s for the shorter one
    if buffer_nb_gss.shape[0] < buffer_nb_reservoir.shape[0]:
        diff = buffer_nb_reservoir.shape[0] - buffer_nb_gss.shape[0]
        buffer_nb_gss = np.concatenate((np.zeros((diff, buffer_nb_gss.shape[1])), buffer_nb_gss), axis=0)
    else:
        diff = buffer_nb_gss.shape[0] - buffer_nb_reservoir.shape[0]
        buffer_nb_reservoir = np.concatenate((np.zeros((diff, buffer_nb_reservoir.shape[1])), buffer_nb_reservoir), axis=0)


buffer_nb = buffer_nb_gss + buffer_nb_reservoir
all_num = buffer_nb.shape[0]


# Cumulative training‑sample targets for each task
training_samples = np.array([0, 33448, 66256,  3400, 15400,  7904,  9136, 81640, 35512])


# ------------------------------------------------------------------
# 3.  Plot
# ------------------------------------------------------------------
fig, ax1 = plt.subplots()

# — Primary axis: cumulative training samples (dashed line)
ax1.plot([252668-all_num, 252668],
         [0, training_samples[task_id]],
         linestyle="--",
         linewidth=1,
         color="#00008B",        # colour‑blind‑safe red‑orange
         label="JT")
ax1.set_xlabel("Training steps", fontsize=6)
ax1.set_ylabel("Number of used samples", fontsize=6, color="#00008B")
ax1.tick_params(axis="y", labelcolor="#00008B")

# — Secondary axis: buffer size curve
ax2 = ax1.twinx()
ax2.plot(np.arange(252668-all_num, 252668),
            buffer_nb,
         linewidth=0.8,
         color="#8B0000",        # colour‑blind‑safe blue
         label="Dual-LS")
ax2.set_ylabel("Number of used samples", fontsize=6, color="#8B0000")
ax2.tick_params(axis="y", labelcolor="#8B0000")

# — Harmonise grids (light, dotted)
ax1.grid(axis="x", color="0.9", linewidth=0.4, linestyle=":")
ax1.grid(axis="y", color="0.9", linewidth=0.4, linestyle=":")

# — Combine legends from both axes
lines = ax1.get_lines() + ax2.get_lines()
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper center", fontsize=6)

ax1.tick_params(axis='both', labelsize=6)
ax2.tick_params(axis='y', labelsize=6)
ax1.set_xlim(252668-all_num, 252668)
# ax1.set_ylim(0, training_samples[task_id]+1000)
# ax2.set_ylim(0, 2000+20)

ax1.set_ylim(0, training_samples[task_id]+300)
ax2.set_ylim(0, 500+20)

fig.tight_layout()

fig.savefig(f"./plot_figures/number_compare_Task_{task_id}.pdf",dpi=300, bbox_inches='tight',pad_inches=0 )
plt.show()
