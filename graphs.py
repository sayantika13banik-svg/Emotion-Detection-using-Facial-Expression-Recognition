import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
counts   = [3995, 436, 4097, 7215, 4965, 4830, 3171]
colors   = ['#E74C3C', '#8E44AD', '#3498DB', '#F1C40F', '#2ECC71', '#1ABC9C', '#E67E22']
fig, ax = plt.subplots(figsize=(14, 4))
# figsize=(14,4) — wide and short, suits a horizontal flow diagram
ax.set_xlim(0, 14)  # x axis goes from 0 to 14
ax.set_ylim(0, 4)   # y axis goes from 0 to 4
ax.axis('off')      # hide axis lines and ticks — cleaner look

# each step in the pipeline — (x position, label, color)
steps = [
    (1,    'Input\nImage',       "#0700C7"),
    (3,    'Grayscale\n48×48',   "#0700C7"),
    (5,    'Normalize\nPixels',  '#0700C7'),
    (7,    'Conv\nLayers',       '#0700C7'),
    (9,    'Pooling\nLayers',    '#0700C7'),
    (11,   'Fully\nConnected',   "#0700C7"),
    (13,   'Softmax\nOutput',    '#0700C7'),
]

for x, label, color in steps:
    # draw a colored rectangle for each step
    ax.add_patch(plt.Rectangle(
        (x - 0.7, 1.2),  # (x, y) of bottom-left corner
        1.4,              # width of rectangle
        1.6,              # height of rectangle
        color=color,
        alpha=0.85,       # slight transparency
        zorder=2          # draw on top of arrows
    ))
    # add label text inside the rectangle
    ax.text(
        x, 2.0,           # center of rectangle
        label,
        ha='center',      # center horizontally
        va='center',      # center vertically
        fontsize=9,
        fontweight='bold',
        color='white',    # white text on colored background
        zorder=3          # draw on top of rectangle
    )

# draw arrows between each step
for i in range(len(steps) - 1):
    x_start = steps[i][0] + 0.7      # right edge of current box
    x_end   = steps[i+1][0] - 0.7    # left edge of next box
    ax.annotate(
        '',                           # no text on arrow
        xy=(x_end, 2.0),             # arrow tip (destination)
        xytext=(x_start, 2.0),       # arrow tail (source)
        arrowprops=dict(
            arrowstyle='->',          # standard arrow shape
            color='#555555',
            lw=1.5                    # line width
        ),
        zorder=1
    )

# add emotion label output below the last box
ax.text(
    13, 0.7,
    "['angry', 'disgust', 'fear',\n'happy', 'neutral', 'sad', 'surprise']",
    ha='center',
    fontsize=8,
    color='#555555'
)

plt.title('Model Pipeline — Flow Diagram', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('graph2_flow_diagram.png', dpi=150, bbox_inches='tight')
# bbox_inches='tight' — makes sure nothing is cut off at edges
plt.show()
print("Saved: graph2_flow_diagram.png")
