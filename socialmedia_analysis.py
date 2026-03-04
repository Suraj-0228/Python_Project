import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

def load_data():
    if os.path.exists('socialmedia_data.csv'):
        df = pd.read_csv('socialmedia_data.csv')
        print('Dataset loaded successfully.')
        return df
    else:
        print("Error: 'socialmedia_data.csv' not found.")
        return None

def clean_data(df):
    print('\n--- DATA CLEANING ---')
    print('Missing values:')
    print(df.isnull().sum())
    df = df.drop_duplicates(subset=['User_ID'])
    print('Duplicates removed. Remaining rows:', len(df))
    print('Cleaning done. Final shape:', df.shape)
    return df

def feature_engineering(df):
    print('\n--- FEATURE ENGINEERING ---')
    # Categorize Stress Level (High Stress if >= 7)
    df['High Stress'] = df['Stress_Level'].apply(lambda x: 'Yes' if x >= 7 else 'No')
    # Total Weekly Screen Time Estimate
    df['Estimated Weekly Screen Time'] = (df['Daily_Phone_Hours'] * 5) + (df['Weekend_Screen_Time_Hours'] * 2)
    print('New columns added: [High Stress, Estimated Weekly Screen Time]')
    print(df[['High Stress', 'Estimated Weekly Screen Time']].head())
    return df

def basic_eda(df):
    avg_phone = df['Daily_Phone_Hours'].mean()
    avg_social = df['Social_Media_Hours'].mean()
    avg_sleep = df['Sleep_Hours'].mean()
    avg_productivity = df['Work_Productivity_Score'].mean()
    print('\n--- SOCIAL MEDIA & SCREEN TIME SUMMARY ---')
    print(f'Total Users Analyzed       : {len(df)}')
    print(f'Average Daily Phone Hours  : {round(avg_phone, 2)}')
    print(f'Average Social Media Hours : {round(avg_social, 2)}')
    print(f'Average Sleep Hours        : {round(avg_sleep, 2)}')
    print(f'Average Productivity Score : {round(avg_productivity, 2)}')

def usage_by_occupation(df):
    occ_usage = df.groupby('Occupation')['Daily_Phone_Hours'].mean().sort_values(ascending=False)
    print('\n--- AVG DAILY PHONE HOURS BY OCCUPATION ---')
    print(occ_usage)
    print(f'\nHighest usage: {occ_usage.idxmax()}')
    return occ_usage

def usage_by_gender(df):
    gender_usage = df.groupby('Gender')['Daily_Phone_Hours'].mean()
    print('\n--- AVG DAILY PHONE HOURS BY GENDER ---')
    print(gender_usage)

def sleep_vs_stress(df):
    stress_sleep = df.groupby('High Stress')['Sleep_Hours'].mean()
    print('\n--- AVG SLEEP HOURS BY STRESS LEVEL ---')
    print(stress_sleep)

def plot_occupation_usage(occ_usage):
    plt.figure(figsize=(8, 8))
    plt.pie(occ_usage.values, labels=occ_usage.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    plt.title('Average Daily Phone Hours by Occupation')
    plt.savefig('viz1_occupation_usage.png')
    plt.show()

def plot_device_distribution(df):
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

def plot_age_trend(df):
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

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig('viz4_correlation_heatmap.png')
    plt.show()

def regression_analysis(df):
    print('\n--- REGRESSION ANALYSIS (Predicting Daily Phone Hours) ---')
    data = df.copy()
    
    # Preprocessing: Encode categorical features
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])
    data['Occupation'] = le.fit_transform(data['Occupation'])
    data['Device_Type'] = le.fit_transform(data['Device_Type'])
    
    # Define features (X) and target (y)
    features = ['Age', 'Gender', 'Occupation', 'Social_Media_Hours', 'Sleep_Hours', 'Stress_Level']
    X = data[features]
    y = data['Daily_Phone_Hours']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict and Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error : {round(mse, 4)}')
    print(f'R-squared Score   : {round(r2, 4)}')
    
    print('\nModel Coefficients:')
    for feature, coef in zip(features, model.coef_):
        print(f'{feature}: {round(coef, 4)}')

def classification_task(df):
    print('\n--- CLASSIFICATION TASK (Predicting High Stress) ---')
    if 'High Stress' not in df.columns:
        print("Error: Please run 'Feature Engineering' (Option 2) first.")
        return
        
    data = df.copy()
    
    # Preprocessing
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])
    data['Occupation'] = le.fit_transform(data['Occupation'])
    data['Device_Type'] = le.fit_transform(data['Device_Type'])
    data['High Stress Label'] = le.fit_transform(data['High Stress']) # No -> 0, Yes -> 1
    
    # Define features (X) and target (y)
    features = ['Age', 'Gender', 'Occupation', 'Daily_Phone_Hours', 'Social_Media_Hours', 'Sleep_Hours']
    X = data[features]
    y = data['High Stress Label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f'Model Accuracy: {round(acc * 100, 2)}%')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=['Low Stress (No)', 'High Stress (Yes)']))
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title('Confusion Matrix: High Stress Prediction')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('viz5_classification_confusion_matrix.png')
    plt.show()

def main():
    df = load_data()
    if df is None:
        return

    while True:
        print("\n" + "="*40)
        print("  SOCIAL MEDIA USAGE ANALYSIS MENU")
        print("="*40)
        print("1. Data Cleaning")
        print("2. Feature Engineering")
        print("3. Basic EDA Summary")
        print("4. Usage by Occupation")
        print("5. Usage by Gender")
        print("6. Sleep vs Stress Level")
        print("7. Plot: Occupation Usage (Pie Chart)")
        print("8. Plot: Device Distribution (Bar Chart)")
        print("9. Plot: Age Trend (Line Chart)")
        print("10. Plot: Correlation Heatmap")
        print("11. Regression Analysis (Predict Daily Phone Hours)")
        print("12. Classification Task (Predict High Stress)")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-12): ")

        match choice:
            case '1':
                df = clean_data(df)
            case '2':
                df = feature_engineering(df)
            case '3':
                basic_eda(df)
            case '4':
                usage_by_occupation(df)
            case '5':
                usage_by_gender(df)
            case '6':
                sleep_vs_stress(df)
            case '7':
                occ_usage = df.groupby('Occupation')['Daily_Phone_Hours'].mean().sort_values(ascending=False)
                plot_occupation_usage(occ_usage)
            case '8':
                plot_device_distribution(df)
            case '9':
                plot_age_trend(df)
            case '10':
                plot_correlation_heatmap(df)
            case '11':
                regression_analysis(df)
            case '12':
                classification_task(df)
            case '0':
                print("Exiting...")
                break
            case _:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

