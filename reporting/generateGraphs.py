from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sample data - replace with your actual rankings
# Lower number means more important (1 is most important)
rankings = np.array([
    [1, 2, 3],  # Algorithm 1
    [2, 1, 3],  # Algorithm 2
    [3, 2, 1]   # Algorithm 3
])

algorithms = ['Algorithm 1', 'Algorithm 2', 'Algorithm 3']
features = ['Feature A', 'Feature B', 'Feature C']

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(rankings, annot=True, cmap="YlOrRd_r", fmt="d",
            xticklabels=features, yticklabels=algorithms)

plt.title('Feature Importance Rankings Across Algorithms')
plt.xlabel('Features')
plt.ylabel('Algorithms')

# Save the figure
plt.tight_layout()
plt.savefig('feature_importance_ranking.png', dpi=300, bbox_inches='tight')
plt.show()