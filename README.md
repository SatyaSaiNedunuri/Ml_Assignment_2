# Job Posting Fraud Detection - Machine Learning Classification

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Models and Performance](#models-and-performance)
- [Author](#author)

---

## Problem Statement

Online job postings have become a primary channel for job seekers, but fraudulent job advertisements pose significant risks to individuals seeking employment. Fake job postings can lead to identity theft, financial loss, and personal data misuse. This project aims to develop a machine learning-based classification system to automatically identify fraudulent job postings, helping job seekers avoid potential scams.

The objective is to build and compare multiple classification algorithms that can accurately distinguish between legitimate and fraudulent job postings based on various features such as company information, job requirements, and posting characteristics. By implementing six different machine learning models, we can evaluate which approach provides the most reliable fraud detection capability.

---

## Dataset Description

### Source

The dataset used for this project is the "Real/Fake Job Posting Prediction" dataset from Kaggle, created by user shivamb. It contains employment-related information scraped from various job posting websites.

**Dataset Link:** [https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

### Dataset Characteristics

| Characteristic         | Details                                 |
| ---------------------- | --------------------------------------- |
| **Total Instances**    | 17,880 job postings                     |
| **Total Features**     | 18 features                             |
| **Features Used**      | 12 features (after selection)           |
| **Target Variable**    | fraudulent (binary: 0 = Real, 1 = Fake) |
| **Class Distribution** | Real: 95.16%, Fake: 4.84%               |
| **Data Type**          | Mixed (categorical, binary, text)       |
| **Missing Values**     | Present in multiple columns             |

### Features Used in the Model

The following 12 features were selected for model training:

**Binary Features:**

- `telecommuting` - Whether remote work is available (0/1)
- `has_company_logo` - Presence of company logo (0/1)
- `has_questions` - Screening questions included (0/1)

**Categorical Features:**

- `employment_type` - Type of employment (Full-time, Part-time, Contract, etc.)
- `required_experience` - Required experience level
- `required_education` - Required education qualification
- `industry` - Industry sector
- `function` - Job function or department

**Derived Features:**

- `has_company_profile` - Company profile provided (0/1)
- `has_description` - Job description present (0/1)
- `has_requirements` - Job requirements listed (0/1)
- `has_benefits` - Benefits information included (0/1)

### Key Dataset Insights

1. **Class Imbalance:** The dataset exhibits significant class imbalance with only 4.84% fraudulent postings, requiring careful handling through techniques like class weighting and stratified sampling.

2. **Missing Data:** Several text-based columns have substantial missing values, which were addressed by creating binary indicator features representing the presence or absence of information.

3. **Feature Selection Rationale:** Features were selected based on their availability, relevance to fraud detection, and minimal missing data. Text-heavy features were converted to binary indicators as fraudulent postings often lack detailed information.

---

## Models and Performance

### Model Comparison Table

| ML Model Name       | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression | 0.9573   | 0.8459 | 0.7358    | 0.2377 | 0.3586 | 0.3912 |
| Decision Tree       | 0.9532   | 0.7589 | 0.6190    | 0.2984 | 0.4021 | 0.3831 |
| K-Nearest Neighbors | 0.9593   | 0.8243 | 0.7895    | 0.1836 | 0.2973 | 0.3623 |
| Naive Bayes         | 0.8972   | 0.8584 | 0.3671    | 0.7582 | 0.4950 | 0.4403 |
| Random Forest       | 0.9687   | 0.8826 | 0.8438    | 0.3279 | 0.4720 | 0.5046 |
| XGBoost             | 0.9676   | 0.8893 | 0.8269    | 0.3525 | 0.4943 | 0.5101 |

### Performance Observations

| ML Model Name           | Observation about model performance                                                                                                                                                                                                                                                                                                                                          |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression** | Shows strong overall accuracy at 95.73% with decent AUC score. However, the low recall (23.77%) indicates it misses many fraudulent cases. The model is conservative in predicting fraud, resulting in high precision but limited fraud detection coverage. Suitable when false positives are costly.                                                                        |
| **Decision Tree**       | Demonstrates moderate performance with balanced precision-recall tradeoff. The slightly lower AUC (75.89%) suggests limited ability to rank predictions. The model tends to overfit on training data despite pruning parameters, leading to moderate generalization. Works well for interpretability but not optimal for this imbalanced dataset.                            |
| **K-Nearest Neighbors** | Achieves highest precision (78.95%) among all models, meaning predictions of fraud are highly reliable. However, very low recall (18.36%) indicates it only catches a small fraction of actual fraud cases. Distance-based classification struggles with the high-dimensional feature space and class imbalance.                                                             |
| **Naive Bayes**         | Exhibits the highest recall (75.82%), successfully identifying most fraudulent postings. However, lower precision (36.71%) results in many false alarms. The probabilistic approach handles class imbalance better than other models. Best choice when catching all fraud cases is priority, despite some legitimate postings being flagged.                                 |
| **Random Forest**       | Delivers excellent balanced performance with strong metrics across the board. AUC of 88.26% and MCC of 0.5046 indicate robust classification ability. The ensemble approach effectively handles feature interactions and reduces overfitting. Provides good balance between precision and recall, making it reliable for production use.                                     |
| **XGBoost**             | Achieves the best overall performance with highest MCC score (0.5101) and excellent AUC (88.93%). The gradient boosting approach with careful hyperparameter tuning handles class imbalance effectively. Shows best balance between all metrics, making it the recommended model for deployment. Slightly better recall than Random Forest while maintaining high precision. |

### Key Findings

1. **Best Overall Model:** XGBoost demonstrates superior performance with the highest MCC score (0.5101) and best balanced metrics, making it the recommended choice for production deployment.

2. **High Precision Models:** K-Nearest Neighbors and Random Forest achieve precision above 78%, making them suitable when minimizing false positives is critical.

3. **High Recall Model:** Naive Bayes achieves 75.82% recall, catching the most fraud cases at the cost of more false alarms. Useful for initial screening before manual review.

4. **Class Imbalance Impact:** All models struggle with the 95:5 class imbalance. Ensemble methods (Random Forest and XGBoost) handle this challenge most effectively through built-in mechanisms.

5. **Trade-offs:** There is a clear precision-recall tradeoff. Business requirements should determine whether to prioritize catching all fraud (high recall) or ensuring flagged cases are truly fraudulent (high precision).

---

## Author

NEDUNURI SATYA SAI PRABHAKARA KAMESWAR
Course: Machine Learning
Month: February 2026

---

**Last Updated:** February 16, 2026
