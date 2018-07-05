#Brest cancer recogintion -> classification for wore version
#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#Importing dataset
RawData = pd.read_csv('data.csv')

#First look into data
RawData.dtypes
RawData.describe()
RawData.hist(bins = 25)
RawData.skew()

#Removing unnesessery columns
RawData.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)

#Renaming values in Diagnosis column -> B(beginning) = 0 (cancer possible to treat)
                                    #-> M(malicious) = 1 (little chances to treat)
def Diagnosis():
    global RawData
    RawData['diagnosis'] = RawData['diagnosis'].map({'M':1, 'B':0})
    return RawData
RawData = Diagnosis()

#Visualisation of data
sns.stripplot(data = RawData, x = 'area_mean', y = 'smoothness_mean', hue = 'diagnosis')
sns.stripplot(data = RawData, x = 'area_mean', y = 'compactness_mean', hue = 'diagnosis')
sns.stripplot(data = RawData, x = 'compactness_mean', y = 'smoothness_mean', hue = 'diagnosis')
sns.stripplot(data = RawData, x = 'area_mean', y = 'concavity_mean', hue = 'diagnosis')
sns.stripplot(data = RawData, x = 'area_mean', y = 'symmetry_mean', hue = 'diagnosis')
sns.stripplot(data = RawData, x = 'area_mean', y = 'area_worst', hue = 'diagnosis')
sns.stripplot(data = RawData, x = 'area_mean', y = 'smoothness_worst', hue = 'diagnosis')
sns.stripplot(data = RawData, x = 'radius_mean', y = 'radius_worst', hue = 'diagnosis')
sns.stripplot(data = RawData, x = 'fractal_dimension_mean', y = 'fractal_dimension_worst', hue = 'diagnosis')
sns.stripplot(data = RawData, x = 'symmetry_mean', y = 'symmetry_worst', hue = 'diagnosis')
sns.stripplot(data = RawData, x = 'area_mean', y = 'concave points_mean', hue = 'diagnosis')
sns.stripplot(data = RawData, x = 'concave points_mean', y = 'concave points_worst', hue = 'diagnosis')

#Adding columns
RawData['distortion_area'] = (RawData['area_worst']/RawData['area_mean'])
RawData['distortion_radius'] = (RawData['radius_worst']/RawData['radius_mean'])

sns.stripplot(data = RawData, x = 'distortion_area', y = 'distortion_radius', hue = 'diagnosis')

#Splitting for Independent and Dependent variables
Y = RawData.iloc[:, 0]
X = RawData.iloc[:, 1:]

#Looking for nan values in dataset
nulls_summary = pd.DataFrame(X.isnull().any(), columns = ['Nulls'])
nulls_summary['Number of NaNs'] = pd.DataFrame(X.isnull().sum())
nulls_summary['Percentage'] = round((X.isnull().mean()*100),2)

#Scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler_1 = MinMaxScaler()
X = scaler_1.fit_transform(X)
X = pd.DataFrame(X)

#Splitting for Training, Validation and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state =0)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 0)

#Fitting first classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

LogReg = LogisticRegression()
RanFor = RandomForestClassifier()

LogReg.fit(X_train, Y_train)
RanFor.fit(X_train, Y_train)

LogRegPred = LogReg.predict(X_val)
RanForPred = RanFor.predict(X_val)

#Checking the accurancy of predictions for training set
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve
cm_LR = confusion_matrix(Y_val, LogRegPred)
cm_RF = confusion_matrix(Y_val, RanForPred)

report_LR = classification_report(Y_val, LogRegPred)
report_RF = classification_report(Y_val, RanForPred)

score_LR = accuracy_score(Y_val, LogRegPred)
score_RF = accuracy_score(Y_val, RanForPred)

x1_LR, x2_LR, _ = roc_curve(Y_val, LogRegPred)
x1_RF, x2_RF, _ = roc_curve(Y_val, RanForPred)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(x1_LR, x2_LR, label='LR')
plt.plot(x1_RF, x2_RF, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


#checking which values are important
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import RFE
acc_all = []
stab_all = []
for m in np.arange(0, 20):
    acc_loop = []
    stab_loop = []
    for n in np.arange(3, 20, 1):
        selector = RFE(RanFor, n, 1)
        cv = cross_val_score(RanFor, X_train.iloc[:, selector.fit(X_train, Y_train).support_], 
                                                  Y_train, cv = 10, scoring = 'accuracy')
        acc_loop.append(cv.mean())
        stab_loop.append(cv.std()*100/cv.mean())
    acc_all.append(acc_loop)
    stab_all.append(stab_loop)
acc = pd.DataFrame(acc_all, columns = np.arange(3, 20, 1))
stab = pd.DataFrame(stab_all, columns = np.arange(3, 20, 1))
print(acc.agg(['mean']))
print(stab.agg(['mean']))

selector = RFE(RanFor, 8, 1)
cols = X_train.iloc[:, selector.fit(X_train, Y_train).support_].columns
print(cols)

#Fitting most important values into model
RanFor2 = RandomForestClassifier()
RanFor2.fit(X_train[cols], Y_train)

RanFor2Pred = RanFor2.predict(X_val[cols])

RanFor2Score = accuracy_score(Y_val, RanFor2Pred)
print('Random Forest score: ',RanFor2Score)

x1_RF, x2_RF, _ = roc_curve(Y_val, RanFor2Pred)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(x1_RF, x2_RF, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

#Taking validation and train set together
def CombinedX():
    X_tr = X_train.append(X_val)
    X_tr.reset_index(inplace = True)
    return X_tr
X_train = CombinedX()

def CombinedY():
    Y_tr = Y_train.append(Y_val)
    return Y_tr
Y_train = CombinedY()

#Seaching for best parameters of the algorithm
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':np.arange(10,50,10),
               'max_depth':np.arange(1,6,1),
               'criterion':['gini', 'entropy'],
               'min_samples_split':np.arange(2,5,1),
               'min_samples_leaf':np.arange(1,3,1),
               'max_features':['auto', 'log2', 'sqrt']}
BestParams = GridSearchCV(RanFor2, scoring = 'accuracy', param_grid = parameters, cv = 10, 
                          verbose = 1)
BestParams.fit(X_train[cols], Y_train)
bests = BestParams.best_params_
print('Best score: ',BestParams.best_score_)
print('Best params: ',BestParams.best_params_)

#Checking the accuracy and stability
cv = cross_val_score(RandomForestClassifier(criterion = 'entropy', max_depth = 5), 
                     X = X_train[cols], y = Y_train, cv = 10, scoring = 'accuracy')
print('Accuracy: ' + str(cv.mean().round(3)))
print('Stability: ' + str((cv.std()*100/cv.mean()).round(3)) + '%')

#Final prediction
RanForFinal = RandomForestClassifier(**BestParams.best_params_)
RanForFinal.fit(X_train[cols], Y_train)
FinalPrediction = RanForFinal.predict(X_test[cols])

#Checking the final solution
FinalCM = confusion_matrix(Y_test, FinalPrediction)
FinalAccuracy = accuracy_score(Y_test, FinalPrediction)
#ROC curve
x1_FP, x2_FP, _ = roc_curve(Y_test, FinalPrediction)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(x1_FP, x2_FP, label='Final Random Forest Classifier')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()