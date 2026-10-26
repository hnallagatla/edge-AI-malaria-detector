import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "grid.alpha": 0.4,
    "figure.dpi": 300
})


models = [
    {"name": "Small FP32 640px", "energy": 0.52, "map": 0.975, "color": "#d62728", "marker": "X"},
    {"name": "Small FP16 640px", "energy": 0.286, "map": 0.980, "color": "#1f77b4", "marker": "o"},
    {"name": "Small INT8 640px", "energy": 0.256, "map": 0.968, "color": "#2ca02c", "marker": "s"},
   
  
    {"name": "Small FP32 960px", "energy": 0.682, "map": 0.965, "color": "#1f77b4", "marker": "o"},
    {"name": "Small FP16 960px", "energy": 0.464, "map": 0.955, "color": "#2ca02c", "marker": "s"},
    {"name": "Small INT8 960px", "energy": 0.343, "map": 0.940, "color": "#1f77b4", "marker": "o"},   
]


sorted_models = sorted(models, key=lambda x: (x["energy"], -x["map"]))

frontier = []
max_map = -1.0 


for m in sorted_models:
    if m["map"] > max_map:
        frontier.append(m)
        max_map = m["map"]

frontier_energy = [m["energy"] for m in frontier]
frontier_map = [m["map"] for m in frontier]

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(frontier_energy, frontier_map, linestyle='--', color='gray', linewidth=1.5, zorder=1, label="Pareto Frontier")

for m in models:
    ax.scatter(
        m["energy"],
        m["map"],
        s=120,
        c=m["color"],
        marker=m["marker"],
        zorder=2,
        edgecolors='black',
        alpha=0.8
    )
   
    
    y_offset = -0.002 if "INT8" in m["name"] else 0.002
    ax.text(
        m["energy"],
        m["map"] + y_offset,
        m["name"].replace("Small ", ""), 
        ha='center',
        va='bottom' if "INT8" not in m["name"] else 'top',
        fontsize=9
    )

ax.set_title("Energy vs. Accuracy Trade-offs in Edge Malaria Detection", pad=15)
ax.set_xlabel("Energy per Inference (Joules) $\\rightarrow$ Lower is Better")
ax.set_ylabel("mAP50-95 (Accuracy) $\\rightarrow$ Higher is Better")

ax.set_xlim(min([m["energy"] for m in models]) * 0.85, max([m["energy"] for m in models]) * 1.05)
ax.set_ylim(min([m["map"] for m in models]) * 0.98, max([m["map"] for m in models]) * 1.01)

ax.grid(True, linestyle=':')
ax.legend(loc="lower right", framealpha=1.0)

yyplt.tight_layout()
plt.savefig("pareto_frontier_complex.png", bbox_inches='tight')
print("Saved IEEE-formatted plot to 'pareto_frontier_complex.png'")