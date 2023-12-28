## necessary imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')

df = pd.read_csv("insurance_claims.csv")
df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], format='%d-%m-%Y', errors='coerce')


df.head()

# we can see some missing values denoted by '?' so lets replace missing values with np.nan

df.replace('?', np.nan, inplace = True)
df.describe()
df.info()
# missing values
df.isna().sum()

import missingno as msno

msno.bar(df)
plt.show()

df['collision_type'] = df['collision_type'].fillna(df['collision_type'].mode()[0])
df['property_damage'] = df['property_damage'].fillna(df['property_damage'].mode()[0])
df['police_report_available'] = df['police_report_available'].fillna(df['police_report_available'].mode()[0])

df.isna().sum()

# heatmap

plt.figure(figsize = (18, 12))

#corr = df.corr()
to_drop = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date',
           'incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year', '_c39']

df.drop(to_drop, inplace = True, axis = 1)
df.head()

#sns.heatmap(data = df.corr(), annot = True, fmt = '.2g', linewidth = 1)
#plt.show()

#df.nunique()

# dropping columns which are not necessary for prediction


# checking for multicollinearity

#plt.figure(figsize = (18, 12))

#corr = df.corr()
#mask = np.triu(np.ones_like(corr, dtype = bool))

#sns.heatmap(data = corr, mask = mask, annot = True, fmt = '.2g', linewidth = 1)
#plt.show()
#print("After dropping age and total_claim_amt:")
df.drop(columns = ['age', 'total_claim_amount'], inplace = True, axis = 1)
#df.head()
#df.info()

# separating the feature and target columns

X = df.drop('fraud_reported', axis = 1)
y = df['fraud_reported']

# extracting categorical columns
cat_df = X.select_dtypes(include = ['object'])

cat_df.head()
# printing unique values of each column
for col in cat_df.columns:
    print(f"{col}: \n{cat_df[col].unique()}\n")

cat_df = pd.get_dummies(cat_df, drop_first = True)
cat_df.head()

# extracting the numerical columns

num_df = X.select_dtypes(include = ['int64'])
num_df.head()

# combining the Numerical and Categorical dataframes to get the final dataset

X = pd.concat([num_df, cat_df], axis = 1)
X.head()

plt.figure(figsize = (25, 20))
plotnumber = 1

for col in X.columns:
    if plotnumber <= 24:
        ax = plt.subplot(5, 5, plotnumber)
        sns.distplot(X[col])
        plt.xlabel(col, fontsize = 15)

    plotnumber += 1

plt.tight_layout()
plt.show()

plt.figure(figsize = (20, 15))
plotnumber = 1

for col in X.columns:
    if plotnumber <= 24:
        ax = plt.subplot(5, 5, plotnumber)
        sns.boxplot(X[col])
        plt.xlabel(col, fontsize = 15)

    plotnumber += 1
plt.tight_layout()
plt.show()

# Splitting data into training set and test set
from sklearn.model_selection import train_test_split

# Splitting data into training set and test set
from sklearn.model_selection import train_test_split

# Set random_state to None for a random split each time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None)

# Record the split indices
split_indices = {'X_train': X_train.index, 'X_test': X_test.index, 'y_train': y_train.index, 'y_test': y_test.index}

# Print the split indices
print("Split indices:")
print(split_indices)

# Extracting numerical columns
num_df = X_train[['months_as_customer', 'policy_deductable', 'umbrella_limit',
                  'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
                  'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
                  'vehicle_claim']]

# Scaling the numeric values in the dataset
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(num_df)

scaled_num_df = pd.DataFrame(data=scaled_data, columns=num_df.columns, index=X_train.index)
X_train.drop(columns=num_df.columns, inplace=True)
X_train = pd.concat([scaled_num_df, X_train], axis=1)
# Models
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV

# Specify the hyperparameter grid to search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1, 10]
}

# Define a list of kernels to try
kernels_to_try = ['linear', 'rbf', 'poly', 'sigmoid']

# Variables to store the best model and accuracy
best_kernel = None
best_params = None
best_accuracy = 0

# Variables to store AUC-ROC values
best_auc_roc = 0
auc_roc_values = {}

# Iterate through each kernel and train an SVM model
for kernel in kernels_to_try:
    print(f"Training SVM with kernel: {kernel}")

    # Update the kernel in the parameter grid
    param_grid['kernel'] = [kernel]

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=SVC(), param_grid=param_grid, scoring='accuracy', cv=5)

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    # Retrieve the best hyperparameters and best model
    current_best_params = grid_search.best_params_
    svm_model = SVC(**current_best_params)

    # Train the model on the entire training set
    svm_model.fit(X_train, y_train)

    # Evaluate the model on the training set
    y_train_pred = svm_model.predict(X_train)
    accuracy_on_train = accuracy_score(y_train, y_train_pred)

    print(f"Training accuracy with {kernel} kernel: {accuracy_on_train}")

    # Evaluate the model on the test set
    y_test_pred = svm_model.predict(X_test)
    accuracy_on_test = accuracy_score(y_test, y_test_pred)

    print(f"Testing accuracy with {kernel} kernel: {accuracy_on_test}")

    # Calculate AUC-ROC on the test set
    auc_roc = roc_auc_score(y_test, svm_model.decision_function(X_test))
    print(f"AUC-ROC with {kernel} kernel: {auc_roc}")



    # Update the best kernel if the current model performs better
    if accuracy_on_test > best_accuracy:
        best_accuracy = accuracy_on_test
        best_kernel = kernel
        best_params = current_best_params

    # Update the best AUC-ROC if the current model performs better
    if auc_roc > best_auc_roc:
        best_auc_roc = auc_roc
        auc_roc_values = {'kernel': kernel, 'params': current_best_params}

# Print the best kernel and its corresponding hyperparameters
print(f"Best kernel: {best_kernel}")
print(f"Best hyperparameters: {best_params}")

# Print the best AUC-ROC and its corresponding kernel and hyperparameters
print(f"Best AUC-ROC: {best_auc_roc} (Kernel: {auc_roc_values['kernel']}, Hyperparameters: {auc_roc_values['params']})")

# Train the final model with the best kernel and hyperparameters on the entire training set
print(f"Training final SVM with the best kernel: {best_kernel}")
best_svm_model = SVC(kernel=best_kernel, C=best_params['C'], gamma=best_params['gamma'])
best_svm_model.fit(X_train, y_train)

# Evaluate the best model on the test set
y_test_pred = best_svm_model.predict(X_test)

# Accuracy score, confusion matrix, classification report, and AUC-ROC on the test set
svc_test_acc = accuracy_score(y_test, y_test_pred)
print(f"Test accuracy of Support Vector Classifier is: {svc_test_acc}")
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
print(f"AUC-ROC on Test Set: {roc_auc_score(y_test, best_svm_model.decision_function(X_test))}")


#other models:

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 30)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# accuracy_score, confusion_matrix and classification_report

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

knn_train_acc = accuracy_score(y_train, knn.predict(X_train))
knn_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of KNN is : {knn_train_acc}")
print(f"Test accuracy of KNN is : {knn_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#decision tree

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

# accuracy_score, confusion_matrix and classification_report

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))
dtc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Decision Tree is : {dtc_train_acc}")
print(f"Test accuracy of Decision Tree is : {dtc_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# hyper parameter tuning

from sklearn.model_selection import GridSearchCV

grid_params = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : [3, 5, 7, 10],
    'min_samples_split' : range(2, 10, 1),
    'min_samples_leaf' : range(2, 10, 1)
}

grid_search = GridSearchCV(dtc, grid_params, cv = 5, n_jobs = -1, verbose = 1)
grid_search.fit(X_train, y_train)

# best parameters and best score

print(grid_search.best_params_)
print(grid_search.best_score_)
# best estimator

dtc = grid_search.best_estimator_

y_pred = dtc.predict(X_test)

# accuracy_score, confusion_matrix and classification_report

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))
dtc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Decision Tree is : {dtc_train_acc}")
print(f"Test accuracy of Decision Tree is : {dtc_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



#Shap part ---> XAI 

import shap
background_summary = shap.sample(X_train, 100) 

explainer = shap.KernelExplainer(lambda x: best_svm_model.decision_function(x), background_summary)
shap_values = explainer.shap_values(X_test)

# Summary plot for the entire dataset
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Summary plot for an individual instance
shap.initjs()
sample_instance = X_test.sample(n=1, random_state=42)
shap.force_plot(explainer.expected_value, shap_values[0, :], sample_instance)

