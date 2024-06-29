
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/Users/karanpandre/Documents/INTERNSHIP/Prodigy/TASK 4/twitter_training.csv')


print("First few rows of the dataset:")
print(df.head())
print()


print("Shape of the dataset:")
print(df.shape)
print()


print("Checking for missing values:")
print(df.isnull().sum())
print()


plt.figure(figsize=(8, 6))
sns.countplot(x=df.columns[2], data=df, palette='viridis')
plt.title('Sentiment Distribution in Social Media Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

