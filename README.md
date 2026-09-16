# Tourism Experience Analytics
### Classification, Prediction, and Recommendation System

A data science project that analyzes tourism data to predict user experience ratings, classify user travel preferences, and recommend personalized destinations — deployed as an interactive Streamlit application.

---

## 📌 Problem Statement

Tourism agencies and travel platforms aim to enhance user experiences by leveraging data to provide personalized recommendations, predict user satisfaction, and classify potential user behavior. This project analyzes user preferences, travel patterns, and attraction features to achieve three primary objectives: **regression**, **classification**, and **recommendation**.

## 🎯 Objectives

| Task | Type | Goal |
|---|---|---|
| **1. Rating/Popularity Prediction** | Regression | Predict the rating/popularity score a user might associate with a destination |
| **2. Preference Classification** | Classification | Predict a user's likely travel preference category based on their profile |
| **3. Destination Recommendation** | Recommendation System | Suggest personalized destinations using content-based filtering |

## 🧰 Skills & Tools Used

- **Data Cleaning & Preprocessing** — handling missing values, standardizing categories, deduplication
- **Exploratory Data Analysis (EDA)** — trend analysis, correlation checks, visualizations
- **Feature Engineering** — one-hot encoding, aggregation, TF-IDF vectorization
- **Machine Learning** — Regression (Linear Regression, Random Forest, XGBoost), Classification, Recommendation Systems (content-based filtering with cosine similarity)
- **Data Visualization** — Matplotlib, Seaborn
- **Deployment** — Streamlit

## 📂 Dataset

The dataset consists of the following tables:

| File | Description |
|---|---|
| `Expanded_Destinations.csv` | Destination details — name, state, type, popularity, best time to visit |
| `Final_Updated_Expanded_Users.csv` | User demographics and travel preferences |
| `Final_Updated_Expanded_UserHistory.csv` | User visit history and experience ratings |
| `Final_Updated_Expanded_Reviews.csv` | User-written reviews and ratings per destination |

## 📁 Project Structure

```
tourism-experience-analytics/
├── dataset/                        # Raw and processed data
│   ├── Expanded_Destinations.csv
│   ├── Final_Updated_Expanded_Users.csv
│   ├── Final_Updated_Expanded_UserHistory.csv
│   ├── Final_Updated_Expanded_Reviews.csv
│   └── processed/
│       └── master.csv              # Cleaned, merged dataset
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_regression_model.ipynb
│   ├── 04_classification_model.ipynb
│   └── 05_recommendation_system.ipynb
├── src/
│   └── build_master_dataset.py     # Data merging & cleaning pipeline
├── models/
│   ├── model.pkl                   # Trained regression model
│   └── model_columns.pkl           # Feature columns for inference alignment
├── app.py                          # Streamlit application
├── requirements.txt
└── README.md
```

## ⚙️ Setup Instructions

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/tourism-experience-analytics.git
cd tourism-experience-analytics
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Build the master dataset**
```bash
python src/build_master_dataset.py
```

**5. Run the Streamlit app**
```bash
streamlit run app.py
```

## 🧪 Methodology

1. **Data Cleaning** — merged 4 raw tables into a single master dataset, handled missing values, standardized categorical text, validated rating ranges (1–5).
2. **EDA** — analyzed destination popularity, rating distributions, user demographics, and preference trends across states and destination types.
3. **Feature Engineering** — one-hot encoded categorical features (`State`, `Type`, `BestTimeToVisit`, `Preferences`, `Gender`); dropped identifier columns not useful for prediction.
4. **Model Training**
   - **Regression**: Compared Linear Regression, Random Forest, and XGBoost using MSE, RMSE, MAE, and R².
   - **Recommendation**: Built a content-based filtering system using TF-IDF vectorization on destination `Type` + `State`, ranked by cosine similarity.
5. **Evaluation** — selected the best-performing regression model based on R² and error metrics; evaluated recommendations qualitatively against user visit history.
6. **Deployment** — packaged the trained model and feature columns with `pickle`, served through a Streamlit interface for interactive predictions and recommendations.

## 📊 Results

| Model | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| Linear Regression | — | — | — | — |
| Random Forest | 0.3007 | 0.548 | — | — |
| XGBoost | — | — | — | — |

*(Fill in with your final comparison table once all models are evaluated.)*

## 🚀 Streamlit App Features

- **Rating/Popularity Prediction** — input user and destination details to get a predicted score
- **Personalized Recommendations** — get top-5 destination suggestions based on visit history and content similarity
- **Insights Dashboard** — visualize popular destinations, rating trends, and user demographics

## 📈 Business Use Cases

- **Personalized Recommendations** — suggest attractions based on user preferences and past behavior
- **Tourism Analytics** — insights into popular attractions and regions
- **Customer Segmentation** — classify users for targeted marketing
- **Customer Retention** — improve satisfaction through personalization

## 👤 Author

Prakhar Singh

## 📄 License

This project is for educational purposes as part of a data science capstone project.
