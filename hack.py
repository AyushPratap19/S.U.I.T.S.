import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = '/Users/ayushpratapsingh/Desktop/akshay/The Lannister Data Chronicles_ Decrypting Diabetes.csv'
df = pd.read_csv(file_path)


print(df.head())


diabetes_dist = df['Diabetes_012'].value_counts().sort_index()


plt.figure(figsize=(10, 6))
sns.barplot(x=diabetes_dist.index, y=diabetes_dist.values, palette='viridis')
plt.xlabel('Diabetes Classification')
plt.ylabel('Number of Individuals')
plt.title('Distribution of Diabetes Classifications')
plt.xticks([0, 1, 2], ['No diabetes', 'Prediabetes', 'Onset of diabetes'])
plt.show()

relevant_features = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'PhysActivity']

fig, axes = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle('Health Behaviors and Outcomes by Diabetes Classification')

for i, feature in enumerate(relevant_features):
    sns.boxplot(data=df, x='Diabetes_012', y=feature, ax=axes[i//2, i%2], palette='viridis')
    axes[i//2, i%2].set_xlabel('Diabetes Classification')
    axes[i//2, i%2].set_ylabel(feature)
    axes[i//2, i%2].set_xticklabels(['No diabetes', 'Prediabetes', 'Onset of diabetes'])

fig.delaxes(axes[2, 1])

plt.tight_layout()
plt.show()

corr_matrix = df.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix of Health Features')
plt.show()