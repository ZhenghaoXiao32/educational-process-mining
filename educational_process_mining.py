# In this .py file, we omitted some informative midway steps to control the output information size of this script
# The whole working procedure is illustrated more clearly in the jupyter notebook file
import requests
import zipfile
import io
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import timedelta

from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


from sklearn.cluster import KMeans

sns.set(style='whitegrid')

# PART 2 GATHERING DATA
# dowload data from machine learning database
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00346/EPMDataset%20.zip'
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall('../educational-process-mining')

# read all the session data and save them into a dictionary
sessions = {}
for x in range(1, 7):
    path = '../educational-process-mining/EPM Dataset 2/Data/Processes/Session {0}'.format(x)
    session = glob.glob(os.path.join(path, '*'))
    dataframes = (pd.read_csv(f, names=['session', 'student_Id', 'exercise', 'activity', 'start_time', 'end_time', 'idle_time', 'mouse_wheel', 'mouse_wheel_click', 'mouse_click_left', 'mouse_click_right', 'mouse_movement', 'keystroke']) for f in session)
    sessions['session{0}'.format(x)] = pd.concat(dataframes, ignore_index=True, sort=False)

# load logs data
logs = pd.read_csv('../educational-process-mining/EPM Dataset 2/Data/logs.txt', sep='\t')


# load final grades data
final_grades_1st = pd.read_excel('../educational-process-mining/EPM Dataset 2/Data/final_grades.xlsx', sheet_name='Exam (First time)')
final_grades_2nd = pd.read_excel('../educational-process-mining/EPM Dataset 2/Data/final_grades.xlsx', sheet_name='Exam (Second time)')

# load intermediate grades data
inter_grades = pd.read_excel('../educational-process-mining/EPM Dataset 2/Data/intermediate_grades.xlsx')

# PART 3 ACCESSING DATA
# check which of the students attended both final exams
attend_1st_id = np.asarray(final_grades_1st['Student ID'])
attend_2nd_id = np.asarray(final_grades_2nd['Student ID'])

def common_member(a, b):

    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        print(a_set & b_set)
    else:
        print('No common elements')


attend_both = common_member(attend_1st_id, attend_2nd_id)

# PART 4 CLEANING DATA
# keep the students who attended in all 6 sessions
ID_list = logs.loc[logs.iloc[:, 1:].sum(axis=1) == 6]

# clean the final grades dataset
# append the two final exam dataset
final = final_grades_1st.append(final_grades_2nd)

# calculate the mean of the final grades for those students who took the exam twice
dup_rows = final[final['Student ID'].duplicated(keep=False)]
final.drop(dup_rows.index, axis=0, inplace=True)
avg_grades = dup_rows.groupby('Student ID').mean()
avg_grades.insert(0, column='Student ID', value=avg_grades.index)
final = final.append(avg_grades).sort_values(by=['Student ID'])

# subset with only those students who attended in all 6 sessions
final = final[final['Student ID'].isin(ID_list['Student Id'])]

# create a pass or fail indicator for the final grades
df = final.copy()
df['pass_IND'] = (df['TOTAL\n(100 points)'] >= 60).astype(int)

# combine the intermediate grades and final grades datasets
df = pd.merge(inter_grades, df, how="inner", left_on="Student Id", right_on="Student ID")

# drop the duplicated column
df.drop(['Student ID'], axis=1, inplace=True)

# rename columns
df.rename(columns={"Student Id": 'ID', "TOTAL\n(100 points)": 'total'}, inplace=True)

# clean logs data
behavior = pd.DataFrame()
for i in range(1, 7):
    behavior = behavior.append(sessions['session{0}'.format(i)])

# convert the start and end time to work time
behavior['work_time'] = (pd.to_datetime(behavior['end_time']) - \
                         pd.to_datetime(behavior['start_time'])).dt.total_seconds()

behavior = behavior.reset_index(drop=True)

# find indexes of noise activities
noise_index = behavior.loc[behavior.activity.str.contains('Other|Blank', regex=True)].index

# drop rows with these indexes
behavior.drop(index=noise_index, inplace=True)

# filter session data based on student id: we need only those who attended all 6 sessions
behavior = behavior.loc[behavior['student_Id'].isin(df.ID)]
behavior = behavior.reset_index(drop=True)

# aggregate behavior data per student per session
sum_behv = behavior.groupby(['student_Id', 'session'], as_index=False).\
           agg({'work_time':'sum', 'idle_time': 'sum', 'mouse_wheel': 'sum', \
                'mouse_wheel_click': 'sum', 'mouse_click_left': 'sum', \
                'mouse_click_right': 'sum', 'mouse_movement': 'sum', 'keystroke': 'sum'})

# remove idle time from behavior data
sum_behv.drop(columns=['idle_time'], inplace=True)

# transform the dataset from long to wide so that a row contains all behavior data of a student per session
behv = sum_behv.pivot(index='student_Id', columns='session', values=['work_time', 'mouse_wheel', 'mouse_wheel_click', 'mouse_click_left',  'mouse_click_right', 'mouse_movement', 'keystroke'])

# rename columns to make it more clear
level_one = behv.columns.get_level_values(0).astype(str)
level_two = behv.columns.get_level_values(1).astype(str)
behv.columns = level_one + level_two

# reset index
behv.reset_index(inplace=True)

# copy datasets
df_clean = df.copy()
# drop uneccesary columns
df_clean.drop(['ES 1.1 \n(2 points)', 'ES 1.2 \n(3 points)', 'ES 2.1\n(2 points)', 'ES 2.2\n(3 points)', 'ES 3.1\n(1 points)', 'ES 3.2\n(2 points)', 'ES 3.3\n(2 points)', 'ES 3.4\n(2 points)', 'ES 3.5\n(3 points)', 'ES 4.1\n(15 points)', 'ES 4.2\n(10 points)', 'ES 5.1\n(2 points)', 'ES 5.2\n(10 points)', 'ES 5.3\n(3 points)', 'ES 6.1\n(25 points)', 'ES 6.2\n(15 points)', 'total'], axis=1, inplace=True)

# join datasets
df_clean = pd.merge(behv, df_clean, how='inner', left_on='student_Id', right_on='ID')
df_clean.drop(['ID'], axis=1, inplace=True)

# save the clean dataset for further analysis
df_clean.to_csv("epm_clean.csv", encoding='utf-8', index=False)


# PART 5 EXPLORATORY DATA ANALYSIS
# we omitted all the plots codes in this part
# for rendered plots, please check our jupyter notebook
# read the clean dataset
df_clean = pd.read_csv('epm_clean.csv')

# PART 6 CLASSIFICATION
# we omitted test models only kept those with best parameters
# generated by grid search with cross validation
# data separation with 70% training and 30% test
X = preprocessing.scale(df_clean.iloc[:, 1:43].values)
X_features = df_clean.iloc[:, 1:43].columns
y = df_clean.iloc[:, 48].values


# Split dataset into training set and test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

# k nearest neighbors with grid search cross validation
knn_param_grid = dict(n_neighbors=range(1, 16))
knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, knn_param_grid, cv=5, verbose=0, iid=True)
knn_grid.fit(X_train, y_train)
y_pred = knn_grid.predict(X_test)

# confusion matrix and classification report
def print_clf_report(model_name):

    print('Confusion matrix of {}:'.format(model_name))
    print();
    print(confusion_matrix(y_test, y_pred))
    print();
    print('Classification report of {}:'.format(model_name))
    print();
    print(classification_report(y_test, y_pred))


print_clf_report('KNN')

# cross validation result
CV_result = cross_val_score(knn_grid.best_estimator_, X_train, y_train, cv=5)

def print_cv_result(model_name):

    print(); print("Cross validation result of {}:".format(model_name))
    print(); print("Cross validation scores: {}".format(CV_result))
    print(); print("Mean of cross validation scores: {}".format(CV_result.mean()))
    print(); print("Standard deviation of cross validation scores: %0.2f" % CV_result.std())
    print(); print("Coefficient of Variation in CV result: %0.2f" % (CV_result.std()/CV_result.mean()))


print_cv_result(model_name='KNN')

# support vector machine with grid search cross validation
svm_param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ["linear", "poly", "rbf", "sigmoid"]}
svm_grid = GridSearchCV(SVC(random_state=1115), svm_param_grid, cv=5, verbose=0, iid=False)
svm_grid.fit(X_train, y_train)

# confusion matrix and classification report
y_pred = svm_grid.predict(X_test)
print_clf_report('SVM')

# cross validation result
CV_result = cross_val_score(svm_grid.best_estimator_, X_train, y_train, cv=5)
print_cv_result('SVM')

# logistic regression with grid search cross validation
C = np.logspace(0, 3, 10)

logr_param_grid = dict(C=C)
logr_grid = GridSearchCV(LogisticRegression(random_state=1115, solver='lbfgs'), logr_param_grid, cv=5, iid=False, verbose=0)
logr_grid.fit(X_train, y_train)
y_pred = logr_grid.predict(X_test)
# confusion matrix and classification report
print_clf_report('Logistic regression')
# cross validation result
CV_result = cross_val_score(logr_grid.best_estimator_, X_train, y_train, cv=5)
print_cv_result('Logsitic regression')

# decision tree with GridSearchCV
dtree = DecisionTreeClassifier(random_state=1115)
criterion = ['gini', 'entropy']
max_depth = [2, 4, 6, 8, 10, 12]
dtree_param_grid = dict(criterion=criterion, max_depth=max_depth)
dtree_grid = GridSearchCV(dtree, dtree_param_grid, cv=5, verbose=0, iid=False)
dtree_grid.fit(X_train, y_train)
# confusion matrix and classification report
y_pred = dtree_grid.predict(X_test)
print_clf_report('Decision tree')
# cross validation result
CV_result = cross_val_score(dtree_grid.best_estimator_, X_train, y_train, cv=5)
print_cv_result('Decision tree')

# random forest with GridSearchCV
rfc = RandomForestClassifier(random_state=1115)
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=500, num=5)]
rfc_param_grid = {'n_estimators': n_estimators}
rfc_grid = GridSearchCV(rfc, rfc_param_grid, cv=5, verbose=0, iid=False)
rfc_grid.fit(X_train, y_train)

# confusion matrix and classification report
y_pred = rfc_grid.predict(X_test)
print_clf_report('Random forest')
# cross validation result
CV_result = cross_val_score(rfc_grid.best_estimator_, X_train, y_train, cv=5)
print_cv_result('Random forest')

# naive bayes with GridSearchCV
gnb = GaussianNB()
gnb_param_grid = {'var_smoothing': [1e-09, 1e-08, 1e-07]}
gnb_grid = GridSearchCV(gnb, gnb_param_grid, cv=5, verbose=0, iid=False)
gnb_grid.fit(X_train, y_train)
# confusion matrix and classification report
y_pred = gnb_grid.predict(X_test)
print_clf_report('Naive Bayes')
# cross validation result
CV_result = cross_val_score(gnb_grid.best_estimator_, X_train, y_train, cv=5)
print_cv_result('Naive Bayes')

# XGBoost with GridSearchCV
xgb_param_grid = {'learning_rate': [1, 0.1, 0.01, 0.001, 0.0001],
                  'n_estimators': [50, 100, 200, 300]}
xgb = XGBClassifier(random_state=1115)
xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=5, verbose=0, iid=True)
xgb_grid.fit(X_train, y_train)
# confusion matrix and classification report
y_pred = xgb_grid.predict(X_test)
print_clf_report('XGBoost')
# cross validation result
CV = cross_val_score(xgb_grid.best_estimator_, X_train, y_train, cv=5)
print_cv_result('XGBoost')


# PART 7 CLUSTERING
# k-means clustering
km = KMeans(n_clusters=6, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=1115)
y_km = km.fit_predict(X)
y_km
