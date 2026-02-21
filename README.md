# 📱 Social Media Usage & Screen Time Analyzer

A Python-based data analysis project that explores the relationship between users' social media habits, screen time, and lifestyle factors.

---

## 📌 Project Overview
This project analyzes a dataset of 50,000 user records to identify patterns and correlations between age, occupation, daily phone hours, and other lifestyle variables. By using Python data analysis libraries and visualizations, it provides insights into how different demographics interact with their devices.

**Key Objectives:**
- Analyze how **Age** correlates with **Average Daily Phone Hours**.
- Understand the distribution of **Daily Phone Hours** across the user base.
- Visualize the breakdown of different **Occupations** among the users.

---

## ✨ Features
This project executes a comprehensive Exploratory Data Analysis (EDA) pipeline:

1.  **📊 Data Overview & Cleaning**: Automatically handles missing values, removes duplicates, and prints a statistical summary of the dataset.
2.  **⚙️ Feature Engineering**: Calculates "Estimated Weekly Screen Time" and flags users with "High Stress".
3.  **📈 Usage Summaries**: Groups and calculates average daily phone usage by Occupation, Device Type, and Gender.
4.  **📉 Visualizations**:
    *   **Pie Chart**: Average Daily Phone Hours by Occupation.
    *   **Bar Chart**: Distribution of Device Types.
    *   **Line Chart**: Trend of Daily Phone Hours across Age.
    *   **Heatmap**: Correlation Matrix of all numerical features.

---

## 🛠️ Technologies Used
*   **Python 3.x**: Core programming language.
*   **Pandas**: For efficient data loading, processing, grouping, and manipulation.
*   **Matplotlib**: For generating professional graphs and charts.
*   **CSV**: Lightweight data storage format.

---

## 📂 Project Structure
```text
Social Media Usage and Screen Time Analysis/
│
├── socialmedia_analysis.py # The main EDA Python script
├── socialmedia_data.csv    # The dataset file (Required)
└── README.md               # Project documentation (This file)
```

---

## 🚀 How to Install & Run

### Prerequisites
Make sure you have **Python** installed. You can check by running:
```bash
python --version
```

### 1. Install Required Libraries
Open your terminal or command prompt and run:
```bash
pip install pandas numpy matplotlib seaborn
```

### 2. Prepare the Data
Ensure the file `socialmedia_data.csv` is located in the same folder as `socialmedia_analysis.py`.

### 3. Run the Analysis
Execute the script using the following command:
```bash
python socialmedia_analysis.py
```

---

## 📝 Author
*   **Name**: [Your Name]
*   **Class**: XII
*   **Subject**: Computer Science
