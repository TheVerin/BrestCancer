import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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