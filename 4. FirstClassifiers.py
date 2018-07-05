import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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