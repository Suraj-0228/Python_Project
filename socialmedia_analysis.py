import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. UPLOAD AND LOAD DATASET
df = pd.read_csv('socialmedia_data.csv')
print('Dataset loaded successfully.')

# 2. DATA CLEANING
print('\nMissing values:')
print(df.isnull().sum())
df = df.drop_duplicates(subset=['User_ID'])
print('\nDuplicates removed. Remaining rows:', len(df))
print('\nCleaning done. Final shape:', df.shape)

# 3. FEATURE ENGINEERING
# Categorize Stress Level (High Stress if >= 7)
df['High Stress'] = df['Stress_Level'].apply(lambda x: 'Yes' if x >= 7 else 'No')
# Total Weekly Screen Time Estimate
df['Estimated Weekly Screen Time'] = (df['Daily_Phone_Hours'] * 5) + (df['Weekend_Screen_Time_Hours'] * 2)
print('\nNew columns added:')
print(df[['High Stress', 'Estimated Weekly Screen Time']].head())

# 4. BASIC EDA - DATA SUMMARY
avg_phone = df['Daily_Phone_Hours'].mean()
avg_social = df['Social_Media_Hours'].mean()
avg_sleep = df['Sleep_Hours'].mean()
avg_productivity = df['Work_Productivity_Score'].mean()
print('\n--- SOCIAL MEDIA & SCREEN TIME SUMMARY ---')
print('Total Users Analyzed       :', len(df))
print('Average Daily Phone Hours  :', round(avg_phone, 2))
print('Average Social Media Hours :', round(avg_social, 2))
print('Average Sleep Hours        :', round(avg_sleep, 2))
print('Average Productivity Score :', round(avg_productivity, 2))

# 5. USAGE BY OCCUPATION
occ_usage = df.groupby('Occupation')['Daily_Phone_Hours'].mean().sort_values(ascending=False)
print('\n--- AVG DAILY PHONE HOURS BY OCCUPATION ---')
print(occ_usage)
print('\nHighest usage:', occ_usage.idxmax())

# 6. USAGE BY GENDER
gender_usage = df.groupby('Gender')['Daily_Phone_Hours'].mean()
print('\n--- AVG DAILY PHONE HOURS BY GENDER ---')
print(gender_usage)

# 7. SLEEP VS STRESS LEVEL
stress_sleep = df.groupby('High Stress')['Sleep_Hours'].mean()
print('\n--- AVG SLEEP HOURS BY STRESS LEVEL ---')
print(stress_sleep)

# 8. VISUALIZATIONS
# --- VIZ 1: Pie Chart - Average Phone Hours by Occupation ---
plt.figure(figsize=(8, 8))
plt.pie(occ_usage.values, labels=occ_usage.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Average Daily Phone Hours by Occupation')
plt.savefig('viz1_occupation_usage.png')
plt.show()

# --- VIZ 2: Bar Chart - Device Type Distribution ---
device_counts = df['Device_Type'].value_counts()
plt.figure(figsize=(10, 5))
plt.bar(device_counts.index, device_counts.values, color='steelblue')
plt.title('Device Type Distribution')
plt.xlabel('Device Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('viz2_device_distribution.png')
plt.show()

# --- VIZ 3: Line Chart - Age vs Average Phone Hours ---
age_usage = df.groupby('Age')['Daily_Phone_Hours'].mean().reset_index()
plt.figure(figsize=(10, 5))
plt.plot(age_usage['Age'], age_usage['Daily_Phone_Hours'], marker='o', color='crimson')
plt.title('Trend of Daily Phone Hours by Age')
plt.xlabel('Age')
plt.ylabel('Average Daily Phone Hours')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('viz3_age_trend.png')
plt.show()
print('VIZ 3 saved.')

# --- VIZ 4: Heatmap - Correlation Matrix ---
# Select only numerical columns for correlation
numeric_df = df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.savefig('images/viz4_correlation_heatmap.png')
plt.show()
print('VIZ 4 saved.')
