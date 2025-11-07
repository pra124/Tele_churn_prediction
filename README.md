# Tele_churn_prediction
Project Overview

The Telco Customer Churn Prediction System is designed to identify whether a customer is likely to discontinue services from a telecommunication company. The main goal of this project is to analyze customer data, detect patterns that lead to churn, and build a predictive model that can help businesses take proactive steps to retain their customers.

This system uses a machine learning pipeline that includes data cleaning, feature engineering, model training, evaluation, and deployment. The final model provides accurate churn predictions based on customer demographics, service usage, and contract details.
Techniques Used
1. Data Preprocessing

To ensure the dataset was clean and ready for model training, several preprocessing techniques were applied:

Handling Missing Values: Missing or blank values were filled using KNN Imputation, which estimates missing data based on the nearest neighbors with similar characteristics.

Feature Transformation: Quantile-based binning was applied on numeric columns like MonthlyCharges and TotalCharges to categorize customers into spending levels (low, medium, high) for better pattern recognition.

Encoding Categorical Variables: All categorical columns such as Gender, Partner, Dependents, Contract, and PaymentMethod were converted into numeric format using suitable encoding methods like Label Encoding and One-Hot Encoding.

Feature Scaling: The StandardScaler technique was used to standardize numerical columns so that all features are on the same scale. This prevents larger values from dominating smaller ones during training.

Dropping Irrelevant Columns: Unnecessary columns such as JoinYear were removed after creating backups, ensuring only relevant features were used for prediction.

2. Model Building

Different machine learning algorithms were implemented and compared to identify the most effective one for churn prediction:

Logistic Regression: Used as a baseline linear model to establish a reference point.

Decision Tree: Built to capture non-linear relationships but showed overfitting with high training accuracy and lower testing accuracy.

Random Forest: An ensemble of multiple decision trees that improved performance and reduced overfitting compared to a single tree.

Gradient Boosting: The final chosen model. It builds trees sequentially, where each tree corrects the errors of the previous ones. This method achieved the highest test accuracy (around 80.55%) and demonstrated excellent generalization ability.

3. Model Evaluation

Various evaluation metrics were used to measure and compare model performance:

Accuracy: To check the percentage of correct predictions.

Precision, Recall, and F1-Score: To measure the model’s effectiveness in identifying actual churners and minimizing false predictions.

Train-Test Split: The dataset was divided into training and testing sets to evaluate the model’s ability to generalize on unseen data.

4. Model Saving and Deployment

After identifying Gradient Boosting as the best-performing model:

The trained model and scaler were saved using serialization techniques to ensure they could be reused without retraining.

A Flask-based web application was developed to deploy the model, allowing users to input customer details and receive real-time churn predictions.

5. Final Selection Justification

Gradient Boosting was finalized as the best model because it effectively balanced accuracy and generalization. It handled both linear and non-linear relationships in the dataset, minimized overfitting through regularization, and achieved the highest test accuracy among all models tested.
