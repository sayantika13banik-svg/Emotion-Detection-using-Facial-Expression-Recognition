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
ax.set_xlim(0, 14)  
ax.set_ylim(0, 4)   
ax.axis('off')      

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

    ax.add_patch(plt.Rectangle(
        (x - 0.7, 1.2),  
        1.4,             
        1.6,             
        color=color,
        alpha=0.85,       
        zorder=2          
    ))

    ax.text(
        x, 2.0,           
        label,
        ha='center',      
        va='center',      
        fontsize=9,
        fontweight='bold',
        color='white',    
        zorder=3          
    )

for i in range(len(steps) - 1):
    x_start = steps[i][0] + 0.7      
    x_end   = steps[i+1][0] - 0.7    
    ax.annotate(
        '',                          
        xy=(x_end, 2.0),             
        xytext=(x_start, 2.0),       
        arrowprops=dict(
            arrowstyle='->',         
            color='#555555',
            lw=1.5                   
        ),
        zorder=1
    )

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
plt.show()
print("Saved: graph2_flow_diagram.png")
