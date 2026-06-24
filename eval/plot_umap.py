import os
import numpy as np
import matplotlib.pyplot as plt
import umap
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))

embeddings = np.load(os.path.join(_HERE, "cohort_b_embeddings.npy"))
labels     = np.load(os.path.join(_HERE, "cohort_b_labels.npy"))

mask = labels >= 0
X, y = embeddings[mask], labels[mask]

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

print(f"Running UMAP on {X.shape[0]} samples ...")
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
X_umap = reducer.fit_transform(X)

fig, ax = plt.subplots(figsize=(7, 6))
colors = ["#2166ac" if lb == 0 else "#d6604d" for lb in y]
ax.scatter(X_umap[:, 0], X_umap[:, 1], c=colors, s=20, alpha=0.8, edgecolors="none")
ax.legend(handles=[
    Patch(color="#2166ac", label="Non-smoker"),
    Patch(color="#d6604d", label="Smoker"),
])
ax.set_title("MethCLR Embeddings — Cohort B (GSE224807)\nAUC=0.7764 | TOP_PROBES=20,000", fontsize=11)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
plt.tight_layout()

out = os.path.join(_HERE, "umap_cohort_b.png")
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.show()
