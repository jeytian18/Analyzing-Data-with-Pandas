# Load necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Data Exploration
print(df.head())
print(df.info())
print(df.describe())

# Data Visualization
plt.plot(df.index, df['sepal length (cm)'])
plt.title('Sepal Length Trend')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.show()
