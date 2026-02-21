# Exploratory Data Analysis (EDA) Report
**Project:** Social Media Usage and Screen Time Analysis

---

## 1. Executive Summary
This report summarizes the Exploratory Data Analysis (EDA) performed on the "Social Media Usage and Screen Time" dataset. The goal of this analysis was to uncover patterns, trends, and correlations between user demographics, screen time habits, and their corresponding effects on well-being parameters such as sleep and stress.

## 2. Dataset Overview
The analysis was conducted on a dataset containing **50,000 user records**. The dataset includes 13 distinct features encompassing demographic details (Age, Gender, Occupation) and behavioral metrics (Daily Phone Hours, Social Media Hours, Sleep Hours, Stress Level, etc.).

## 3. Data Cleaning & Preprocessing
To ensure data quality and integrity before analysis, the following steps were taken:
*   **Missing Values Check:** The dataset was scanned for null values. No missing values were detected across any of the columns.
*   **Duplicate Removal:** Duplicate entries based on the unique `User_ID` were identified and removed, ensuring each record represents a distinct user. The final dataset size remained at 50,000 rows, indicating no duplicates were present.

## 4. Feature Engineering
New metrics were derived to enable deeper analysis:
*   **High Stress Indicator:** Users with a `Stress_Level` of 7 or higher (on a scale of 10) were categorized as having "High Stress" (`Yes`/`No`).
*   **Estimated Weekly Screen Time:** A new column was calculated by combining weekday and weekend screen time habits `(Daily_Phone_Hours * 5) + (Weekend_Screen_Time_Hours * 2)`.

## 5. Key Insights & Findings

### 5.1 General Behavioral Averages
Across the entire user base, the following overall averages were observed:
*   **Average Daily Phone Hours:** ~ 6.51 hours
*   **Average Social Media Hours:** ~ 4.27 hours
*   **Average Sleep Hours:** ~ 6.50 hours
*   **Average Work Productivity Score:** ~ 5.50 / 10

### 5.2 Usage Trends by Occupation
Analyzing device usage across different professions revealed slight variations:
1.  **Business Owner:** 6.54 hours/day (Highest Usage)
2.  **Male / Female / Freelancer / Student / Professional:** All maintained a similar average, hovering closely around 6.48 to 6.52 hours per day.

### 5.3 Stress and Sleep Correlation
An aggregated look at sleep hours segmented by the newly created "High Stress" indicator showed very little variance in sleep duration based on stress levels:
*   Users **without** High Stress averaged **6.49 hours** of sleep.
*   Users **with** High Stress averaged **6.51 hours** of sleep.

## 6. Visualizations Summary
Four key visualizations were generated during the script execution to graphically represent the data:
1.  **Pie Chart (Average Phone Hours by Occupation):** Illustrates the proportional share of daily phone usage across different occupations.
2.  **Bar Chart (Device Type Distribution):** Displays the count of users categorized by their primary device type.
3.  **Line Chart (Age vs. Average Phone Hours):** Plots the trendline of daily phone usage across different age groups to spot generational differences.
4.  **Correlation Heatmap:** A matrix displaying the statistical correlation coefficients between all numerical features (e.g., how closely Social Media Hours correlate with Productivity or Stress).

## 7. Conclusion
The dataset indicates a heavily normalized distribution of phone usage across different demographics. Despite expectations, extreme stress levels do not show a drastic correlation with sleep deprivation within this specific dataset. The generated modular Python script (`socialmedia_analysis.py`) successfully automates the extraction of these statistical insights and visualizations.
