from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sample data - replace with your actual rankings
# Lower number means more important (1 is most important)
# dataset = "Austrailian_Credit"
dataset = "Heart"
# dataset = "Compas"
rankings = np.array([
    #Austrailian
    # [2.2,11.8,8.4,5.6,5.6,11.2,9.2,1,12.4,4.2,11.2,8.2,10.4,3.6],  # XG Boost
    # [1.8,13,12.2,4.8,4.8,12.8,9,1.4,8.6,5.6,10.6,8.2,9.4,2.8],  # Logistic Regression
    # [1.2,10.6,12.6,4.8,11,7.6,11.4,6.6,7.2,11.25,8.4,5,5.2,2.2],   # Multilayer Perceptron
    # [2.8,10.6,12.2,7.8,5.6,12,7.2,1,2.6,4.8,12.8,11.2,9.6,4.8],   # Random Forest

    #Heart
    [11.2,3.2,3,8.4,6.8,10.2,11.6,8,10.2,8.2,6.2,1.6,2.2],  # XG Boost
    [12.8,2,6.4,8.6,10.8,7.4,6.6,8,4.8,10,9.6,2.4,1.6],  # Logistic Regression
    [7.4,8.8,8.6,3.2,7,6,6.2,2.8,7.8,12.6,8.4,4.2,8],   # Multilayer Perceptron
    [11.2,3.4,2.8,11.6,9.6,12.4,9.8,7.4,6,7,6,2.2,1.4]   # Random Forest
    
    #Compas Data
    # [1.6,3.2,1.4,5.2,9,5.8,7.2,7.8,3.8],  # XG Boost
    # [1,3,2,5,7,6,8.4,8.6,4],  # Logistic Regression
    # [1.8,4,1.2,9,5,3,6.6,8,6.4],   # Multilayer Perceptron
    # [1.4,1.6,3,5,6.2,6.8,8.8,8.2,4]   # Random Forest
])

algorithms = ['XG', 'LR', 'MLP', 'RF']
# Austrailian features
# features = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14']

# Heart features
features = ['Age','Sex','Chest_Pain_Type','resting_blood_pressure','serum_cholestoral','fasting_blood_sugar','_resting_electrocardiographic_results','maximum_heart_rate_achieved','exercise_induced_angina','_oldpeak','_the_slope_of_the_peak_exercise_ST_segment','number_of_major_vessels_colored_by_flourosopy','thal']

# Compas features
# features = ['age', 'recidivated', 'number_of_prior_crimes', 'months_in_jail', 'felony', 'misdemeanor', 'woman', 'man', 'black']

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(rankings, annot=True, cmap="YlOrRd_r", fmt=".1f",
            xticklabels=features, yticklabels=algorithms)

plt.title(dataset + ' Data Feature Importance Rankings Across Algorithms(5 iterations)')
plt.xlabel('Features')
plt.ylabel('Algorithms')

# Save the figure
plt.tight_layout()
plt.savefig('feature_importance_ranking_'+dataset+'.png', dpi=300, bbox_inches='tight')
plt.show()