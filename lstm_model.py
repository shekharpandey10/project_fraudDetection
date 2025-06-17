# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score,f1_score, roc_auc_score


import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('./synthetic_transactions.csv')

data.shape

data.head(10)

data.tail(10)

data.dtypes

if data.isnull().values.any():
    print('Unfortunately, there are missing values in the dataset\n')
    data.dropna(inplace=True)
    print('Shape : ', data.shape)
else:
    print('Fortunately, there aren\'t missing values in the dataset.')

plt.figure(figsize=(10, 6))
sns.boxplot(data=data, orient='h') # horizontal
plt.title('Outliers')
plt.grid(axis='y')
plt.show()

data.columns

data['type'].unique()

data['type'].value_counts()

plt.figure(figsize=(7,3))
plt.title('type vs counts')
sns.countplot(data=data,x='type',palette='coolwarm')
plt.xlabel('Type')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=1)
plt.show()

data['type'].replace({'CASH_OUT':0, 'PAYMENT':1, 'CASH_IN':2, 'TRANSFER':3, 'DEBIT':4}, inplace=True)

data['type'].value_counts()

data.head()

# check labels in "isFraud" feature
data['isFraud'].unique()

data['isFraud'].value_counts()

Target_counts = data['isFraud'].value_counts()

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=Target_counts.index, y=Target_counts.values)
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Target Counts \n (isn\'t Fraud = 0 || is Fraud = 1)')
plt.xticks()
plt.grid(axis='y')
plt.show()

data.drop(['oldbalanceOrg', 'newbalanceOrig'], axis=1, inplace=True)

data.columns

data.shape # (rows, columns)

data.corr()

plt.figure(figsize=(8, 8))
sns.heatmap(data.corr(), annot=True, linewidths=0.9, fmt=".1f", cmap='Spectral')
plt.show()

""" ## Correlation between oldbalanceOrg and newbalanceOrig = (0.99)

"""

##plt.scatter(data['oldbalanceOrg'], data['newbalanceOrig'], label='Data')
##plt.xlabel('oldbalanceOrg')
##plt.ylabel('newbalanceOrig')
##plt.title('oldbalanceOrg vs. newbalanceOrig')
##plt.grid(True)
##plt.show()

plt.scatter(data['amount'], data['newbalanceDest'], label='Data') # choose appropriate column
plt.xlabel('amount') # edit with the actual name of column
plt.ylabel('newbalanceDest') # edit with the actual name of column
plt.title('amount vs. newbalanceDest') # change the plot title
plt.grid(True)
plt.show()

"""## Dropping unnecessary features based on correlation"""

##data.drop(['isFlaggedFraud', 'step'], axis=1, inplace=True)
if 'isFlaggedFraud' in data.columns and 'step' in data.columns:
    data.drop(['isFlaggedFraud', 'step'], axis=1, inplace=True)
else:
    print("Columns 'isFlaggedFraud' and/or 'step' not found in the DataFrame.")

data.columns

data.shape

data.info()

data.head() # Default : first 5 rows

data.tail() # Default : last 5 rows

data.describe().T

data.shape # (rows, columns)

"""## Splitting the data and target"""

# X Data
X = data.drop(['isFraud'], axis=1)
print('X shape is : ' , X.shape)
print()

# y Data
y = data['isFraud']
print('y shape is : ' , y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# Splitted Data
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)

"""## Data Scaling / Feature Scaling"""

# Standardization (Z-Score Normalization)
# StandardScaler for Data

scaler = StandardScaler()

# Fit the scaler on the training data
scaler.fit(X_train)

# Transform the training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""## Applying "Logistic Regression" Algorithm"""

Model_LR = LogisticRegression()
Model_LR.fit(X_train_scaled, y_train)
y_pred_LR = Model_LR.predict(X_test_scaled)

# Quick evaluation
Train_Accuracy = Model_LR.score(X_train_scaled, y_train)
Test_Accuracy = Model_LR.score(X_test_scaled, y_test)
print(f'Training accuracy: {Train_Accuracy*100:.2f} %')
print(f'Testing accuracy: {Test_Accuracy*100:.2f} %')

"""## The Evaluation"""

# Confusion Matrix

CM = confusion_matrix(y_true=y_test, y_pred=y_pred_LR)
ConfusionMatrixDisplay(CM, display_labels=data['isFraud'].unique()).plot()
plt.title('Confusion Matrix Without Normalization')
plt.show()

print(classification_report(y_test, y_pred_LR))

# Accuracy = (TP + TN) / (TP + TN + FP + FN)
Accuracy_LR = accuracy_score(y_test, y_pred_LR)
print(f'➤➤➤ Accuracy Score : {Accuracy_LR * 100 : .2f} %\n')


# Precision = TP / (TP + FP)
Precision_LR = precision_score(y_test, y_pred_LR)
print(f'➤➤➤ Precision Score : {Precision_LR * 100 : .2f} %\n')


# Recall = TP / (TP + FN)
Recall_LR = recall_score(y_test, y_pred_LR)
print(f'➤➤➤ Recall Score : {Recall_LR * 100 : .2f} %\n')

# F1 Score = 2 × ((Precision * Recall) / (Precision + Recall))
F1_Score_LR = f1_score(y_test, y_pred_LR)
print(f'➤➤➤ F1 Score : {F1_Score_LR * 100 : .2f} %\n')


ROC_AUC_LR = roc_auc_score(y_test, y_pred_LR)
print(f'➤➤➤ AUC_ROC : {ROC_AUC_LR * 100 : .2f} %\n')

Scores = [Accuracy_LR, Precision_LR, Recall_LR, F1_Score_LR, ROC_AUC_LR]
Score_Names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']

# Plot
plt.figure(figsize=(7, 5))
plt.pie(Scores, labels=Score_Names, autopct='%1.2f%%', startangle=140)
plt.axis('equal')
plt.show()

"""## Applying "Decision Tree" Algorithm"""

Model_DT = DecisionTreeClassifier()
Model_DT.fit(X_train_scaled, y_train)
y_pred_DT = Model_DT.predict(X_test_scaled)

# Quick evaluation
Train_Accuracy = Model_DT.score(X_train_scaled, y_train)
Test_Accuracy = Model_DT.score(X_test_scaled, y_test)
print(f'Training accuracy: {Train_Accuracy*100:.2f} %')
print(f'Testing accuracy: {Test_Accuracy*100:.2f} %')

"""## The Evaluation"""

# Confusion Matrix

CM = confusion_matrix(y_true=y_test, y_pred=y_pred_DT)
ConfusionMatrixDisplay(CM, display_labels=data['isFraud'].unique()).plot()
plt.title('Confusion Matrix Without Normalization')
plt.show()

print(classification_report(y_test, y_pred_DT))

# Accuracy = (TP + TN) / (TP + TN + FP + FN)
Accuracy_DT = accuracy_score(y_test, y_pred_DT)
print(f'➤➤➤ Accuracy Score : {Accuracy_DT * 100 : .2f} %\n')


# Precision = TP / (TP + FP)
Precision_DT = precision_score(y_test, y_pred_DT)
print(f'➤➤➤ Precision Score : {Precision_DT * 100 : .2f} %\n')


# Recall = TP / (TP + FN)
Recall_DT = recall_score(y_test, y_pred_DT)
print(f'➤➤➤ Recall Score : {Recall_DT * 100 : .2f} %\n')
# F1 Score = 2 × ((Precision * Recall) / (Precision + Recall))
F1_Score_DT = f1_score(y_test, y_pred_DT)
print(f'➤➤➤ F1 Score : {F1_Score_DT * 100 : .2f} %\n')


ROC_AUC_DT = roc_auc_score(y_test, y_pred_DT)
print(f'➤➤➤ AUC_ROC : {ROC_AUC_DT * 100 : .2f} %\n')

Scores = [Accuracy_DT, Precision_DT, Recall_DT, F1_Score_DT, ROC_AUC_DT]
Score_Names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']

# Plot
plt.figure(figsize=(7, 5))
plt.pie(Scores, labels=Score_Names, autopct='%1.2f%%', startangle=140)
plt.axis('equal')
plt.show()

"""## Applying "Naive Bayes" Algorithm"""

Model_NB = GaussianNB()
Model_NB.fit(X_train_scaled, y_train)
y_pred_NB = Model_NB.predict(X_test_scaled)


Train_Accuracy = Model_NB.score(X_train_scaled, y_train)
Test_Accuracy = Model_NB.score(X_test_scaled, y_test)
print(f'Training accuracy: {Train_Accuracy*100:.2f} %')
print(f'Testing accuracy: {Test_Accuracy*100:.2f} %')

"""## The Evaluation"""

CM = confusion_matrix(y_true=y_test, y_pred=y_pred_NB)
ConfusionMatrixDisplay(CM, display_labels=data['isFraud'].unique()).plot()
plt.title('Confusion Matrix Without Normalization')
plt.show()

print(classification_report(y_test, y_pred_NB))

# Accuracy = (TP + TN) / (TP + TN + FP + FN)
Accuracy_NB = accuracy_score(y_test, y_pred_NB)
print(f'➤➤➤ Accuracy Score : {Accuracy_NB * 100 : .2f} %\n')


# Precision = TP / (TP + FP)
Precision_NB = precision_score(y_test, y_pred_NB)
print(f'➤➤➤ Precision Score : {Precision_NB * 100 : .2f} %\n')


# Recall = TP / (TP + FN)
Recall_NB = recall_score(y_test, y_pred_NB)
print(f'➤➤➤ Recall Score : {Recall_NB * 100 : .2f} %\n')

# F1 Score = 2 × ((Precision * Recall) / (Precision + Recall))
F1_Score_NB = f1_score(y_test, y_pred_NB)
print(f'➤➤➤ F1 Score : {F1_Score_NB * 100 : .2f} %\n')


ROC_AUC_NB = roc_auc_score(y_test, y_pred_NB)
print(f'➤➤➤ AUC_ROC : {ROC_AUC_NB * 100 : .2f} %\n')

def predict_fraud(model, features):
    """
    Predicts whether the given transaction is fraud or not.

    Parameters:
    model: The trained model used for prediction.
    features: A numpy array of shape (1, n_features) containing the features of the transaction.

    Returns:
    str: 'Fraud' if the transaction is predicted to be fraud, 'Not-fraud' otherwise.
    """
    if  model.predict(features):
        return "Fraud"
    else:
        return "Not-fraud"



Scores = [Accuracy_NB, Precision_NB, Recall_NB, F1_Score_NB, ROC_AUC_NB]
Score_Names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']

# Plot
plt.figure(figsize=(7, 5))
plt.pie(Scores, labels=Score_Names, autopct='%1.2f%%', startangle=140)
plt.axis('equal')
plt.show()

evaluation = pd.DataFrame({'Model': ['Logistic Regression','Decision Tree', 'Naive Bayes'],
                           'Accuracy': [(Accuracy_LR*100), (Accuracy_DT*100), (Accuracy_NB*100)]})

evaluation

data["type"]=data["type"].map({"CASH_OUT": 1,"PAYMENT": 2,"CASH_IN": 3,"TRANSFER": 4,"DEBIT":5})
data["isFraud"]=data["isFraud"].map({0:"No Fraud",1:"Fraud"})
print (data.head())

# #prediction
# #features=[tyype,amount,oldvalanceorg,newbalanceorig]
# features=np.array([[5,10000.40,9000.60,0.0,0.1]])
# print(Model_NB.predict(features))

# #prediction
# #features=[tyype,amount,oldvalanceorg,newbalanceorig]
# #prediction
# #features=[tyype,amount,oldvalanceorg,newbalanceorig]
# features=np.array([[4,9000.60,20000.12,0.0,9000.60,0.1]])
# if(Model_NB.predict(features)):
#     result2="not-fraud";
def result2():
    return "not-fraud"

# Assuming Model_NB is already defined and trained for no 4 debit
##features = np.array([[4, 10000.20, 19000.20,9000.00,0.0,0.0]])
#result = predict_fraud(Model_NB, features)
#print(result)

def predict_fraud(model, features):
    """
    Predicts whether the given transaction is fraud or not.

    Parameters:
    model: The trained model used for prediction.
    features: A numpy array of shape (1, n_features) containing the features of the transaction.

    Returns:
    str: 'Fraud' if the transaction is predicted to be fraud, 'Not-fraud' otherwise.
    """
    features = features[:, :4]  # Select only the first 4 features
    if model.predict(features):  # Remove extra space before model.predict()
        return "Fraud"
    else:  # Align with the if statement
        return "Not-fraud"

features = np.array([[3, 7970.75, 372876.69,364905.94,118357.65,126328.4]])
result = predict_fraud(Model_NB, features)
print(result)

#for no 2 cashin

features = np.array([[1, 12163.16, 278963.72,266800.57,441307.73,453470.88]])
result = predict_fraud(Model_NB, features)
print(result2())



import pickle

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(Model_NB, file)

print("Model has been saved to 'model.pkl'")