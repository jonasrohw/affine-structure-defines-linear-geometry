"""Generate all figures for the paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# --- Figure 1: Side-by-side dimension sweeps ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Salzburg
dims_s = [8, 16, 32, 64]
affine_acc = [73.0, 97.0, 100.0, 27.5]
ax1.bar(range(len(dims_s)), affine_acc, color=['#d62728', '#2ca02c', '#2ca02c', '#d62728'],
        edgecolor='black', linewidth=0.5, width=0.6)
ax1.set_xticks(range(len(dims_s)))
ax1.set_xticklabels([str(d) for d in dims_s])
ax1.set_xlabel('Hidden dimension')
ax1.set_ylabel('Affine accuracy (%)')
ax1.set_title('(a) Salzburg: Affine map accuracy')
ax1.set_ylim(0, 110)
ax1.axhline(y=100, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
for i, v in enumerate(affine_acc):
    ax1.text(i, v + 2.5, f'{v:.0f}%', ha='center', fontsize=9)

# Graz
dims_g = [8, 16, 32, 64]
heldout_acc = [42.5, 90.0, 85.0, 67.5]
ax2.bar(range(len(dims_g)), heldout_acc, color=['#d62728', '#2ca02c', '#1f77b4', '#d62728'],
        edgecolor='black', linewidth=0.5, width=0.6)
ax2.set_xticks(range(len(dims_g)))
ax2.set_xticklabels([str(d) for d in dims_g])
ax2.set_xlabel('Hidden dimension')
ax2.set_ylabel('Held-out accuracy (%)')
ax2.set_title('(b) Graz: Relation generalization')
ax2.set_ylim(0, 110)
ax2.axhline(y=10, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Chance (10%)')
for i, v in enumerate(heldout_acc):
    ax2.text(i, v + 2.5, f'{v:.0f}%', ha='center', fontsize=9)
ax2.legend(loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.savefig('figures/fig1_dimension_sweeps.pdf')
plt.savefig('figures/fig1_dimension_sweeps.png')
plt.close()
print("Figure 1 saved.")


# --- Figure 2: Salzburg three regimes (line plot with annotations) ---
fig, ax = plt.subplots(figsize=(6, 4))

dims = [8, 16, 32, 64]
affine_transformer = [73.0, 97.0, 100.0, 27.5]
affine_existence = [70.0, 99.0, 100.0, None]  # no existence test at dim=64
obj_acc = [99.5, 100.0, 100.0, 100.0]

ax.plot(dims, affine_transformer, 'o-', color='#1f77b4', linewidth=2, markersize=8,
        label='Transformer affine acc', zorder=3)
ax.plot([8, 16, 32], [70.0, 99.0, 100.0], 's--', color='#ff7f0e', linewidth=1.5,
        markersize=7, label='Existence test (ceiling)', zorder=3)
ax.plot(dims, obj_acc, '^:', color='#2ca02c', linewidth=1.5, markersize=7,
        label='Object prediction acc', zorder=3)

ax.annotate('Bottleneck:\nno solution\nexists', xy=(8, 73), xytext=(12, 55),
            fontsize=8, ha='center', arrowprops=dict(arrowstyle='->', color='gray'),
            color='#d62728')
ax.annotate('Sweet spot:\nstructure emerges', xy=(24, 98.5), xytext=(24, 80),
            fontsize=8, ha='center', color='#2ca02c')
ax.annotate('Memorization:\nno pressure\nfor structure', xy=(64, 27.5), xytext=(52, 50),
            fontsize=8, ha='center', arrowprops=dict(arrowstyle='->', color='gray'),
            color='#d62728')

ax.set_xlabel('Hidden dimension')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Salzburg: Three regimes of affine structure')
ax.set_xticks(dims)
ax.set_ylim(0, 110)
ax.legend(loc='center left', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig2_salzburg_regimes.pdf')
plt.savefig('figures/fig2_salzburg_regimes.png')
plt.close()
print("Figure 2 saved.")


# --- Figure 3: Graz dimension sweep detail ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

dims = [8, 16, 32, 64]
heldout = [42.5, 90.0, 85.0, 67.5]
contrastive_pos = [85.2, 88.8, 88.8, 90.0]
contrastive_neg = [94.0, 97.8, 99.2, 97.6]
partner_r0 = [0.217, 0.430, 0.180, 0.098]
partner_r1 = [0.195, 0.422, 0.179, 0.130]
affine_test = [26.0, 50.0, 55.0, 63.0]

ax1.plot(dims, heldout, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='Held-out acc')
ax1.plot(dims, affine_test, 's--', color='#ff7f0e', linewidth=1.5, markersize=7, label='Affine probe test')
ax1.axhline(y=10, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax1.set_xlabel('Hidden dimension')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('(a) Generalization vs capacity')
ax1.set_xticks(dims)
ax1.set_ylim(0, 100)
ax1.legend(loc='lower right', framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.text(10, 13, 'chance', fontsize=8, color='gray')

ax2.plot(dims, partner_r0, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='R0 partner sim')
ax2.plot(dims, partner_r1, 's-', color='#ff7f0e', linewidth=2, markersize=7, label='R1 partner sim')
ax2.set_xlabel('Hidden dimension')
ax2.set_ylabel('Cosine similarity')
ax2.set_title('(b) Embedding partner similarity')
ax2.set_xticks(dims)
ax2.set_ylim(0, 0.5)
ax2.legend(loc='upper right', framealpha=0.9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig3_graz_dim_sweep.pdf')
plt.savefig('figures/fig3_graz_dim_sweep.png')
plt.close()
print("Figure 3 saved.")


# --- Figure 4: Graz phase progression (for appendix) ---
fig, ax = plt.subplots(figsize=(8, 4.5))

phases = ['Phase 2\n(10 feat,\ntrain/test\nsubjects)', 'Phase 3\n(10 feat,\nheld-out\ntriples)',
          'Phase 3j\n(7 feat,\nno contrastive)', 'Phase 3l\n(7 feat,\n+contrastive\ndim=64)',
          'Phase 3n\n(6 feat,\n+contrastive\ndim=64)', 'Phase 3o3\n(5 feat,\n+contrastive\ndim=16)']
held_out = [0.0, 35.0, 50.6, 69.3, 87.8, 90.0]
colors = ['#d62728', '#d62728', '#ff7f0e', '#2ca02c', '#2ca02c', '#2ca02c']

bars = ax.bar(range(len(phases)), held_out, color=colors, edgecolor='black', linewidth=0.5, width=0.7)
ax.set_xticks(range(len(phases)))
ax.set_xticklabels(phases, fontsize=8)
ax.set_ylabel('Held-out accuracy (%)')
ax.set_title('Graz: Progression from memorization to generalization')
ax.set_ylim(0, 105)

for i, v in enumerate(held_out):
    ax.text(i, v + 2, f'{v:.0f}%' if v > 0 else '0%', ha='center', fontsize=9, fontweight='bold')

ax.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('figures/fig4_graz_progression.pdf')
plt.savefig('figures/fig4_graz_progression.png')
plt.close()
print("Figure 4 saved.")


# --- Figure 5: Contrastive data scaling (for appendix) ---
fig, ax = plt.subplots(figsize=(6, 4))

labels = ['No contrastive\n(phase3j)', 'Contrastive\n6.25K seq', 'Contrastive\n62.5K seq']
heldout_vals = [50.6, 30.7, 69.3]
colors = ['#1f77b4', '#d62728', '#2ca02c']

bars = ax.bar(range(len(labels)), heldout_vals, color=colors, edgecolor='black',
              linewidth=0.5, width=0.5)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Held-out accuracy (%)')
ax.set_title('Contrastive learning: data quantity matters\n(7 features, 441 entities)')
ax.set_ylim(0, 85)
ax.axhline(y=4.76, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax.text(2.4, 6, 'chance (4.8%)', fontsize=8, color='gray')

for i, v in enumerate(heldout_vals):
    ax.text(i, v + 1.5, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig5_contrastive_scaling.pdf')
plt.savefig('figures/fig5_contrastive_scaling.png')
plt.close()
print("Figure 5 saved.")

print("\nAll figures generated.")
