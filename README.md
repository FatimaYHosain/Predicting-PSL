# Predicting-PSL


# 🏏 Predicting PSL Match Winners Using Machine Learning

The **Pakistan Super League (PSL)** is a popular T20 cricket league that attracts top cricketing talent from around the world. Due to the numerous factors influencing a team’s success—such as player performance, match conditions, and team composition—predicting match outcomes can be complex. This project aims to build a machine learning model that predicts the winner of a PSL match based on historical performance data.

---

## 📊 Dataset Description

This project utilizes a **ball-by-ball PSL dataset** containing detailed match-level information from **2016 to 2020**.

### ✅ Features Include:
- `psl_year`
- `match_number`
- `team_1`, `team_2`
- `inning`, `over`, `ball`
- `runs`, `total_runs`
- `wickets`, `is_four`, `is_six`
- `is_wicket`, `wicket`, `wicket_text`
- `result`

This comprehensive dataset covers:
- Match statistics  
- Team compositions  
- Ball-level events  
- Match outcomes  

---

## 🧠 Model Implementation

We implemented and compared multiple machine learning algorithms to determine which performs best at predicting match outcomes.

### 🔍 Models Used

#### ✅ Random Forest Classifier
- Ensemble method using multiple decision trees.
- **Accuracy**: `80.03%`
- **Prediction Example**:
  - Outcome Probability: `[[0.9383, 0.0617]]`
  - **93.8%** chance of **Team 2 winning**
  - **6.1%** chance of **Team 1 winning**

#### ✅ Support Vector Machine (SVM)
- Supervised learning model for classification.
- **Accuracy**: `78.52%`
- **Prediction Example**:
  - Outcome Probability: `[[0.4641, 0.5359]]`
  - **53.6%** chance of **Team 1 winning**
  - **46.4%** chance of **Team 2 winning**

#### ✅ XGBoost Classifier
- Gradient boosting framework using decision trees.
- **Accuracy**: `81.13%`
- **Prediction Example**:
  - Outcome Probability: `[[0.7840, 0.2159]]`
  - **78.4%** chance of **Team 2 winning**
  - **21.6%** chance of **Team 1 winning**

#### ✅ Decision Tree Classifier
- Splits data based on feature values.
- **Accuracy**: `81.13%`
- **Prediction Example**:
  - Outcome Probability: `[[0.7143, 0.2857]]`
  - **71.4%** chance of **Team 2 winning**
  - **28.6%** chance of **Team 1 winning**

---

## 📈 Model Comparison and Insights

| Model               | Accuracy | Best Predicted Class |
|--------------------|----------|----------------------|
| XGBoost Classifier | 81.13%   | Team 2               |
| Random Forest      | 80.03%   | Team 2               |
| Decision Tree      | 81.13%   | Team 2               |
| SVM                | 78.52%   | Team 1               |

### 🏆 Final Conclusion:

- **XGBoost Classifier (XGBC)** is the best-performing model in terms of:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
- **Random Forest** performs well but slightly underperforms compared to XGBoost.
- **Decision Tree** shows signs of **overfitting** (better on training data than test).
- **SVM** has the lowest overall performance.

> ✅ Therefore, **XGBoost Classifier is recommended for deployment** due to its strong predictive power and balanced performance.

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn (for visualization)
- Jupyter Notebook / Google Colab

---

## 📁 Project Structure

```

.
├── data/                 # Raw and processed PSL datasets
├── notebooks/            # EDA and model training notebooks
├── models/               # Trained model files
├── src/                  # Scripts for preprocessing and training
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies

```

---

## 📌 Future Improvements

- Add real-time match prediction dashboard
- Incorporate more contextual features (pitch type, weather, toss result)
- Use deep learning models for comparison
- Automate match result fetching and updating

---

## 📬 Contact

For any queries or collaboration opportunities, feel free to reach out via GitHub or email.

---
```
