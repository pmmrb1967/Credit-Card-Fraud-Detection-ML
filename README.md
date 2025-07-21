# Credit Card Fraud Detection with Standard Python Libraries

This repository contains a step-by-step guide and implementation for building a credit card fraud detection system using fundamental Python data science libraries like Pandas, NumPy, Scikit-learn, and Matplotlib. It focuses on addressing the critical challenges inherent in such datasets, particularly the extreme class imbalance.

## Project Overview

Credit card fraud detection is a complex and vital task for financial institutions. The primary goal is to accurately identify fraudulent transactions to prevent financial losses and maintain customer trust. This project tackles the common real-world problem of imbalanced data, where fraudulent transactions are extremely rare compared to legitimate ones.

This notebook provides a comprehensive walkthrough, diving into:

* **Data Loading and Exploratory Data Analysis (EDA):** Understanding the dataset's characteristics, with a focus on class distribution and key features like 'Time' and 'Amount'.
* **Data Preprocessing:** Including feature scaling and preparing the data for model training.
* **Handling Imbalanced Data:** Employing techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the minority class in the training set.
* **Model Training and Comparison:** Training and evaluating multiple classical machine learning models (e.g., Logistic Regression, Decision Tree, Random Forest, Gradient Boosting), focusing on appropriate metrics for imbalanced datasets (Precision, Recall, F1-Score, ROC AUC).
* **Model Persistence:** Demonstrating how to save and load the best-performing model for future predictions.
* **Prediction Example:** Applying the trained model to make predictions on new, unseen data.

## Dataset

This project utilizes the **Kaggle Credit Card Fraud Detection dataset**.

**Key Dataset Information:**
* Contains transactions made by European cardholders in September 2013 over two days.
* Highly imbalanced: Only 492 frauds out of 284,807 transactions (approximately **0.172%** are fraudulent).
* Features `V1, V2, … V28` are the result of a PCA transformation due to confidentiality.
* `Time`: Seconds elapsed between each transaction and the first transaction in the dataset.
* `Amount`: Transaction amount.
* `Class`: The target variable, where `1` indicates fraud and `0` indicates legitimate.

**Important Note:** Given the severe class imbalance, evaluation metrics like **Area Under the Precision-Recall Curve (AUPRC)** and a comprehensive **Confusion Matrix** are more meaningful than simple accuracy.

**To obtain the dataset (`creditcard.csv`), please download it directly from:**
[Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Key Challenges Addressed

1.  **High Data Volume:** Demands fast and efficient models for near real-time response.
2.  **Imbalanced Data:** The minuscule proportion of fraudulent transactions makes detection difficult for conventional ML algorithms.
3.  **Data Availability/Privacy:** Anonymized features necessitate focusing on patterns within transformed data.
4.  **Misclassified Data:** Acknowledging potential inaccuracies in labeled data.
5.  **Adaptive Fraudster Techniques:** Requires dynamic and adaptable detection models.

## Strategies Implemented

This project explores and implements strategies to overcome the above challenges:

* **Simple and Fast Models:** Prioritizing the use of models that can quickly classify transactions.
* **Handling Imbalance:** Utilizing specialized methods like **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the training data.
* **Dimensionality Reduction:** Leveraging the already PCA-transformed `V-features` which inherently aids in privacy and computational efficiency.
* **Reliable Data Sources:** Using a well-known and authentic (though anonymized) dataset.
* **Interpretable and Adaptive Models:** Focusing on models that allow for better understanding and adaptation.

## How to Use This Project Locally

To run this project on your local machine, follow these steps:

1.  **Download the Project Files:**
    * If you've cloned the repository, navigate to the project directory:
        ```bash
        cd Credit-Card-Fraud-Detection-Python
        ```
    * If you downloaded a `.zip` file, extract its contents to a folder on your computer. Then, open your terminal or command prompt and navigate to that folder.

2.  **Download the Dataset:**
    * This notebook requires the `creditcard.csv` dataset. Download it directly from the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) page.
    * **Place the `creditcard.csv` file directly into the main project folder** (the same folder where `ML_example3_without.ipynb` is located).

3.  **Install Required Libraries:**
    * Open your terminal or command prompt.
    * Run the following command to install all necessary Python libraries:
        ```bash
        pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
        ```
    * *(Note: `imbalanced-learn` is specifically required for the SMOTE technique used to handle imbalanced data.)*

4.  **Launch Jupyter Notebook/Lab and Run the Project:**
    * In your terminal or command prompt, while still in the project directory, execute the following command:
        ```bash
        jupyter notebook
        # OR
        jupyter lab
        ```
    * This will open a new tab in your web browser, showing the Jupyter interface.
    * Click on `ML_example_credit_card_fraud_detection.ipynb` to open the notebook.
    * You can then run the cells sequentially to execute the fraud detection model.

---
