#!/usr/bin/env python3
"""Create three vector diagram figures for Venice workshop."""

newfontsize = 18

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def create_left_panel():
    """Three parallel equal-length vectors: verb → gerund."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Vector parameters
    start_x = [0.5, 1.5, 2.5]
    start_y = [0.5, 0.5, 0.5]
    dx, dy = 1.5, 2.0  # Same displacement for all (parallel, equal length)

    labels_start = ['TAKE', 'LIE', 'DRINK']
    labels_end = ['TAKING', 'LYING', 'DRINKING']

    for i in range(3):
        ax.annotate('', xy=(start_x[i] + dx, start_y[i] + dy),
                    xytext=(start_x[i], start_y[i]),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        ax.text(start_x[i] - 0.1, start_y[i] - 0.2, labels_start[i],
                fontsize=newfontsize, ha='center', va='top')
        ax.text(start_x[i] + dx + 0.1, start_y[i] + dy + 0.2, labels_end[i],
                fontsize=newfontsize, ha='center', va='bottom')

    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.tight_layout()
    fig.savefig('venice/panel_left.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Created panel_left.png")

def create_middle_panel():
    """Three vectors of different lengths, randomly distributed, different directions."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Randomly distributed vectors, different directions and lengths
    # Format: (name, date, start_x, start_y, end_x, end_y, name_ha, name_va, name_xoff, name_yoff, date_ha, date_va, date_xoff, date_yoff)
    data = [
        ('MADONNA', '1958-08-16', 0.5, 2.5, 2.8, 3.8, 'center', 'top', -0.4, -0.15, 'left', 'bottom', 0.1, 0.1),
        ('ELVIS', '1935-01-08', 3.5, 1.0, 2.6, 2.5, 'center', 'top', 0, -0.15, 'right', 'bottom', 0.5, 0),  # arrow points to "01"
#        ('ELVIS', '1935-01-08', 3.5, 1.0, 2.6, 2.5, 'center', 'top', 0, -0.15, 'right', 'bottom', -0.3, 0.1),  # arrow points to "01"
        ('STING', '1951-10-02', 1.5, 0.5, 2.0, 2.1, 'center', 'top', -0.1, -0.15, 'left', 'bottom', -0.95, 0),  # arrow points to "10"
    ]

    for name, date, sx, sy, ex, ey, n_ha, n_va, n_xo, n_yo, d_ha, d_va, d_xo, d_yo in data:
        ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        ax.text(sx + n_xo, sy + n_yo, name, fontsize=newfontsize, ha=n_ha, va=n_va)
        ax.text(ex + d_xo, ey + d_yo, date, fontsize=newfontsize, ha=d_ha, va=d_va)

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.tight_layout()
    fig.savefig('venice/panel_middle.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Created panel_middle.png")

def create_right_panel():
    """Three parallel equal-length vectors with clouds and x marks."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Vector parameters (parallel, equal length)
    start_x = [0.5, 1.8, 3.1]
    start_y = [0.5, 0.5, 0.5]
    dx, dy = 1.5, 2.0

    labels_start = ['GOULD', 'BACH', 'JOHANN\nGEORG II']
    labels_end = ['PIANO', 'ORGAN', 'TRUMPET']

    labels_start = ['GOULD', 'BACH', 'JOHANN\nGEORG II']
    labels_end = ['PIANO', 'ORGAN', 'TRUMPET']

    # Cloud offsets (where x appears relative to arrow tip) and label positions
    # Format: (x_offset, y_offset, label_ha, label_va, label_x_off, label_y_off)
    x_positions = [
        (0.0, -0.35, 'center', 'top', 0, -0.25),      # PIANO: bottom middle, label below outside
        (-0.2, 0.35, 'left', 'center', 0.15, 0),      # ORGAN: keep as before
        (0.25, -0.25, 'left', 'center', 0.15, 0),     # TRUMPET: keep as before
    ]

    for i in range(3):
        if i!=2: continue
        end_x = start_x[i] + dx
        end_y = start_y[i] + dy

        # Draw arrow
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x[i], start_y[i]),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

        # Draw cloud around arrow tip
        cloud = mpatches.FancyBboxPatch(
            (end_x - 0.5, end_y - 0.4), 1.0, 0.9,
            boxstyle="round,pad=0.1,rounding_size=0.3",
            facecolor='white', edgecolor='gray', linestyle='--', linewidth=1.2,
            zorder=0
        )
        ax.add_patch(cloud)

        # Draw x mark for true target
        x_off, y_off, ha, va, lx_off, ly_off = x_positions[i]
        x_pos = end_x + x_off
        y_pos = end_y + y_off
        ax.plot(x_pos, y_pos, 'x', markersize=10, color='black', mew=2)
        ax.text(x_pos + lx_off, y_pos + ly_off, labels_end[i], fontsize=newfontsize, ha=ha, va=va)

        # Start label
        ax.text(start_x[i] - 0.1, start_y[i] - 0.2, labels_start[i],
                fontsize=newfontsize, ha='center', va='top')

    ax.set_xlim(-0.5, 6)
    ax.set_ylim(-0.5, 3.8)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.tight_layout()
    fig.savefig('venice/johanngeorgii.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Created panel_right.png")

if __name__ == '__main__':
    #create_left_panel()
    #create_middle_panel()
    create_right_panel()
    print("Done!")
