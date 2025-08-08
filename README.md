# Classification-with-Logistics-Regression-
This code builds a binary classifier using Logistic Regression on the Breast Cancer dataset. It splits the data into training and test sets, standardizes the features, and fits the model. Performance is evaluated using confusion matrix, precision, recall, F1-score, and ROC-AUC.
Procedure for Logistic Regression Classification
1. Import Libraries – Load necessary Python packages such as pandas, numpy, matplotlib, and scikit-learn modules.
2. Load Dataset – Use the load_breast_cancer() dataset from sklearn.datasets for binary classification.
3. Split Data – Divide the dataset into training and testing sets using train_test_split, keeping the class distribution with stratify.
4. Standardize Features – Apply StandardScaler to normalize features for better model performance.
5. Train Model – Fit a LogisticRegression model using the scaled training data.
6. Predict & Evaluate – Generate predictions, compute the confusion matrix, classification report, and ROC-AUC score.
7. Visualize ROC Curve – Plot the ROC curve to assess model discrimination capability.
8. Tune Threshold – Adjust the classification threshold and observe its effect on recall and precision.
9. Explain Sigmoid Function – Plot the sigmoid curve to show how logistic regression converts raw scores into probabilities.
