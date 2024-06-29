import pandas as pd
import matplotlib.pyplot as plt


try:
    
    df = pd.read_csv('/Users/karanpandre/Documents/INTERNSHIP/Prodigy/Task1/API_SP.POP.TOTL_DS2_en_csv_v2_23.csv', delimiter=',', on_bad_lines='skip')
except pd.errors.ParserError as e:
    print("ParserError: ", e)
    exit()


print(df.head())


recent_year = df.columns[-1]


population_data = df[['Country Name', recent_year]].dropna()


population_data[recent_year] = pd.to_numeric(population_data[recent_year], errors='coerce')

print(population_data.head())


plt.figure(figsize=(12, 6))


plt.hist(population_data[recent_year], bins=50, edgecolor='k', alpha=0.7)

plt.title('Population Distribution by Country')
plt.xlabel('Population')
plt.ylabel('Number of Countries')


plt.show()
