from data_processing import df_pro
import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score,f1_score,precision_score, roc_auc_score
import joblib
# Independent feature that have all columns except Attrition
X = df_pro.drop("Attrition", axis =1)   #(axis = 0 working with rows) and (axis = 1 working with columns)
#Dependent variable
Y = df_pro["Attrition"]
#spliting data 80% train and 20% test
#0.2 test size means 20% test data is split and random state control the randomness in dataset during spliting
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) 
#Feature Scaling
''' Feature scaling is the process of normalizing or standardizing the range of 
    independent variables (features) in a dataset. It ensures that no single feature 
    dominates the model simply because it has a larger magnitude.  '''
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#model training 
''' Logistic Regression learns by repeatedly adjusting weights to reduce error.
    Each adjustment = 1 iteration. It keeps going until either: weight stablized or limit hits.'''
#class_weight="balanced":(we use class_weight because our dataset is highly imbalanced)
#Class weight automatically assigns higher importance (weight) to the minority class and lower weight to the majority class.
model = LogisticRegression(class_weight="balanced",max_iter=1000) #create logistic regression classifier from sklearn
model.fit(X_train,Y_train)   # train the model.
"""predict_proba() predict the probability of each employee and return table having 2 columns. 
    First column return is probability of class 0 and Second column is probability of class 1 (e.g
    employee 1 --> class Yes probability = 0.8 and class No probability = 0.2 results 80% chance that
    employee will leave the company in future) """
#[:,1] tells, only pick the probability of class "Yes" means if probability is greater than threshold probability then empolyee will leave the company.
#Threshold Probability : It is cutoff value used to convert continuous probability into binary class.
Y_probability =model.predict_proba(X_test)[:,1]
"""Now, adjust threshold prabability from 0.5(default) to 0.44 results it return true(yes) and false(no)
    that's why astype(int) is used it convery true/false into 0/1."""
Y_prediction = (Y_probability>=0.44).astype(int)
"""Intercept : The intercept is the starting point of prediction when all feature values are zero."""
#sklearn store model intercept in array(intercept_), that's why we also give array index [0].
print("Intercept:", model.intercept_[0])
"""zip() function : zip() function is used to combine multiple lists(arrays), tuples, strings etc.
    It return a zip object containing paired elements."""
#X.columns are input columns
#model.coef_ is a list(array) that store coefficient at index [0].
#dict() or list() convert zip object into dictionary or list.
print("Coefficients:", list(zip(X.columns, model.coef_[0])))
#Evaluation Section
#1. Accuracy metrics: Show how many predictions are correct out of total.
#Accuracy take actual data and predicted output and check how many are correct
accuracy = accuracy_score(Y_test,Y_prediction)
print(f"Accuracy: {accuracy}")
#2. Confusion Matrix: simple table used to evaluate how well a classification.
"""Confusion matrix break the result into 4 special categories by comparing the actual 
    values(Truth) vs. the predicted values(guesses).
    1. True Negatives (TN): The model predicted 0, and the truth was 0. (Correct!)
    2. False Positives (FP): The model predicted 1, but the truth was 0. (Wrong - "False Alarm")
    3. False Negatives (FN): The model predicted 0, but the truth was 1. (Wrong - "Missed Case")
    4. True Positives (TP): The model predicted 1, and the truth was 1. (Correct!)"""
    #confusion matrix is look like
"""[True Negatives(TN)    False Positives(FP)]
   [False Negatives(TN)   True Positive(TP)]"""
matrix = confusion_matrix(Y_test,Y_prediction)
print(f"Confusion Matrix: {matrix}")
#3. Recall Metric: Recall check the completeness of predictions. Focus on Missing case.
"""Accuracy looks at the whole picture, Recall only focuses on the "positives." It is sometimes called Sensitivity.
    e.g. Imagine you are testing 100 people for a virus and 10 actual have virus, model predict 
    9 have virus, then recall is 90%."""
#formula for Recall = TP / (TP + FN)
recall = recall_score(Y_test,Y_prediction)
print(f"Recall score: {recall}")

#4. Precision score :  Check the correctness of predictions. 
"""The model predicts 10 people have the condition.In reality, only 8 of those people actually 
    have it.The other 2 were healthy people who got a False Alarm. """
#formula for precision = TP / (TP + FP) 
precision = precision_score(Y_test, Y_prediction)
print(f"Precision: {precision}")

#5. F1 score: Metric used in classification problems to balance precision and recall.
# formula for f1 score = 2 [(precision*recall) / (precision+recall)]
f1 = f1_score(Y_test, Y_prediction)
print(f"F1 score: {f1}")
"""ROC-AUC score measures how well a machine learning model can distinguish between two classes."""
auc = roc_auc_score(Y_test, Y_prediction)
print(f"ROC-AUC Score: {auc}")

joblib.dump(model, "employee_attrition_model.pkl")
joblib.dump(scaler, "scaler.pkl")