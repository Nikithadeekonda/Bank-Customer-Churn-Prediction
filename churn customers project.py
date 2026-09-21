#!/usr/bin/env python
# coding: utf-8

# In[90]:


#1. import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')


# In[116]:


#2. Load data
df = pd.read_csv('BankChurners.csv')
# Preview data
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())


# In[100]:


#3.Data Cleaning & Preprocessing 
#Remove columns not useful for prediction (e.g., CLIENTNUM, and any columns with constant values)
if 'CLIENTNUM' in df.columns:
    df = df.drop('CLIENTNUM', axis=1)
if 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon' in df.columns:
    df = df.drop([col for col in df.columns if 'Naive_Bayes' in col], axis=1)

# Encode target variable
df['Attrition_Flag'] = df['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})

# Check categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns:", cat_cols)

# Fill missing/unknown values
for col in cat_cols:
    df[col] = df[col].replace('Unknown', np.nan)
    df[col] = df[col].fillna(df[col].mode()[0])

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Check for numerical columns with missing values
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
imputer = SimpleImputer(strategy='median')
df[num_cols] = imputer.fit_transform(df[num_cols])


# In[102]:


# 4. Exploratory Data Analysis
# Target variable distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Attrition_Flag', data=df)
plt.title('Churn Distribution')
plt.xticks([0,1], ['Existing', 'Attrited'])
plt.show()


# In[104]:


# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm', center=0)
plt.title('Feature Correlation')
plt.show()


# In[118]:


#5. Train-Test Split
X = df.drop('Attrition_Flag', axis=1)
y = df['Attrition_Flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# In[108]:


#6. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[110]:


#7. Model Training: Random Forest
# Hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [8, 12, 16],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42, class_weight='balanced')
grid = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
best_rf = grid.best_estimator_


# In[112]:


#8. Model Evaluation
# Predictions
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:,1]

# Evaluation metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label='Random Forest (AUC = {:.2f})'.format(roc_auc_score(y_test, y_proba)))
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[114]:


#9. Feature Importance
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices][:15], y=features[indices][:15])
plt.title('Top 15 Feature Importances')
plt.show()


# In[ ]:




