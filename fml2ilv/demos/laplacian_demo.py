"""
Laplacian Eigenvalue Demo — Erdős–Rényi Graphs
FML2ILV / CS-E4740  |  Alex Jung
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ── parameters ────────────────────────────────────────────────────────────────
N   = 80          # number of nodes (same for all three graphs)
SEED = 42

configs = [
    dict(p=0.04, label="Sparse",  color="#4A90D9"),   # below connectivity threshold
    dict(p=0.10, label="Medium",  color="#E8A838"),   # near threshold ln(N)/N ≈ 0.056
    dict(p=0.35, label="Dense",   color="#5CB85C"),   # well-connected
]

# ── figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10), facecolor="#0f1117")
fig.suptitle(
    "Laplacian Spectrum of Erdős–Rényi Graphs  $G(n,p)$,  $n={}$".format(N),
    fontsize=14, color="white", fontweight="bold", y=0.97
)

# 3 rows × 2 cols  (graph left | eigenvalue histogram right)
outer = gridspec.GridSpec(3, 2, figure=fig,
                          left=0.04, right=0.97,
                          top=0.93, bottom=0.05,
                          hspace=0.45, wspace=0.30)

for row, cfg in enumerate(configs):
    p, label, color = cfg["p"], cfg["label"], cfg["color"]

    # ── build graph & Laplacian ───────────────────────────────────────────────
    G = nx.erdos_renyi_graph(N, p, seed=SEED)
    L = nx.laplacian_matrix(G).toarray().astype(float)
    eigenvalues = np.linalg.eigvalsh(L)       # sorted ascending, λ₀ ≈ 0

    n_components = sum(1 for v in eigenvalues if v < 1e-8)
    lambda_2     = eigenvalues[1]             # algebraic connectivity (Fiedler)
    lambda_max   = eigenvalues[-1]

    # ── left: graph drawing ───────────────────────────────────────────────────
    ax_g = fig.add_subplot(outer[row, 0])
    ax_g.set_facecolor("#0f1117")

    pos = nx.spring_layout(G, seed=SEED, k=1.2/np.sqrt(N))

    # draw edges first (thin, semi-transparent)
    nx.draw_networkx_edges(G, pos, ax=ax_g,
                           edge_color=color, alpha=0.25, width=0.6)
    # draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax_g,
                           node_color=color, node_size=30, alpha=0.90)

    ax_g.set_title(
        f"{label}  ($p={p}$)   "
        f"edges: {G.number_of_edges()}   "
        f"components: {n_components}",
        color="white", fontsize=9, pad=4
    )
    ax_g.axis("off")

    # ── right: eigenvalue histogram ───────────────────────────────────────────
    ax_e = fig.add_subplot(outer[row, 1])
    ax_e.set_facecolor("#181c25")
    for spine in ax_e.spines.values():
        spine.set_edgecolor("#333")

    ax_e.hist(eigenvalues, bins=30, color=color, alpha=0.80, edgecolor="none")

    # mark Fiedler value & λ_max
    ax_e.axvline(lambda_2,   color="white",   lw=1.2, ls="--",
                 label=f"$\\lambda_2={lambda_2:.2f}$  (Fiedler)")
    ax_e.axvline(lambda_max, color="#ff6b6b", lw=1.2, ls=":",
                 label=f"$\\lambda_{{max}}={lambda_max:.1f}$")

    ax_e.set_xlabel("Eigenvalue", color="#aaa", fontsize=8)
    ax_e.set_ylabel("Count",      color="#aaa", fontsize=8)
    ax_e.tick_params(colors="#888", labelsize=7)
    ax_e.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    ax_e.legend(fontsize=7, framealpha=0.2,
                labelcolor="white", facecolor="#0f1117")

    # connectivity threshold annotation on bottom row only
    if row == 0:
        thresh = np.log(N) / N
        ax_e.set_title(
            f"Connectivity threshold  $p^* = \\ln n / n \\approx {thresh:.3f}$",
            color="#aaa", fontsize=7.5, pad=3
        )

plt.savefig("/mnt/user-data/outputs/laplacian_demo.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("saved.")
