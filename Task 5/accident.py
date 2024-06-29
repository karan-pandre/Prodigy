import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

num_records = 10000


data = {
    'ID': range(1, num_records + 1),
    'Start_Time': [datetime(2023, random.randint(1, 12), random.randint(1, 28),
                            random.randint(0, 23), random.randint(0, 59)) for _ in range(num_records)],
    'End_Time': [datetime(2023, random.randint(1, 12), random.randint(1, 28),
                          random.randint(0, 23), random.randint(0, 59)) + timedelta(hours=random.randint(1, 6))
                 for _ in range(num_records)],
    'Severity': np.random.choice([1, 2, 3, 4], size=num_records, p=[0.1, 0.3, 0.4, 0.2]),
    'Weather_Condition': np.random.choice(['Clear', 'Rain', 'Snow', 'Fog', 'Hail'], size=num_records),
    'Road_Conditions': np.random.choice(['Dry', 'Wet', 'Snowy', 'Icy'], size=num_records),
    'Temperature(F)': np.random.normal(loc=60, scale=15, size=num_records),
    'Hour': [],
    'DayOfWeek': [],
    'Month': [],
    'City': ['City' + str(random.randint(1, 10)) for _ in range(num_records)],
    'State': ['State' + str(random.randint(1, 5)) for _ in range(num_records)],
    'Timezone': np.random.choice(['US/Pacific', 'US/Eastern', 'US/Central', 'US/Mountain'], size=num_records)
}

for time in data['Start_Time']:
    data['Hour'].append(time.hour)
    data['DayOfWeek'].append(time.weekday())
    data['Month'].append(time.month)


df = pd.DataFrame(data)


print("First few rows of the dataset:")
print(df.head())
print()


df.to_csv('synthetic_traffic_accidents.csv', index=False)


print("Shape of the dataset:")
print(df.shape)
print()


print("Checking for missing values:")
print(df.isnull().sum())
print()


df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['End_Time'] = pd.to_datetime(df['End_Time'])


df['Weather_Condition'] = df['Weather_Condition'].astype('category')
df['Road_Conditions'] = df['Road_Conditions'].astype('category')
df['Timezone'] = df['Timezone'].astype('category')


plt.figure(figsize=(10, 6))
sns.countplot(x='Road_Conditions', data=df, palette='viridis')
plt.title('Distribution of Accidents by Road Conditions')
plt.xlabel('Road Conditions')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(x='Weather_Condition', data=df, palette='viridis')
plt.title('Distribution of Accidents by Weather Conditions')
plt.xlabel('Weather Conditions')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Hour', bins=24, kde=True, color='skyblue', edgecolor='black')
plt.title('Accidents by Time of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.show()
