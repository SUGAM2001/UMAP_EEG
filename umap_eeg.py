import pandas as pd
import numpy as np
import umap.umap_ as umap  
import matplotlib.pyplot as plt
import seaborn as sns

# Load your datasets
class1 = pd.read_csv("D:\\feature\\Class2_0_all_fet.csv")
class2 = pd.read_csv("D:\\feature\\Class2_10_all_fet.csv")
class3 = pd.read_csv("D:\\feature\\Class2_80_all_fet.csv")

# Combine the datasets
result_combined = np.vstack((class1, class2, class3))

# Create labels for the classes
a = class1.shape[0]
b = class2.shape[0]
c = class3.shape[0]
labels = np.array([0] * a + [1] * b + [2] * c)

# Perform UMAP dimensionality reduction
reducer = umap.UMAP(n_neighbors=15, min_dist=0.2, n_components=2, random_state=42)
embedding = reducer.fit_transform(result_combined)

# Plot UMAP results
plt.figure(figsize=(10, 7))
sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette='Set1', s=60, alpha=0.7)
plt.title('UMAP Clustering of EEG Features', fontsize=16)
plt.xlabel('UMAP 1', fontsize=12)
plt.ylabel('UMAP 2', fontsize=12)
plt.legend(title='Classes')
plt.tight_layout()
plt.show()

