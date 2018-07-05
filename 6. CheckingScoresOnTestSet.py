import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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