import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler_1 = MinMaxScaler()
X = scaler_1.fit_transform(X)
X = pd.DataFrame(X)

#Splitting for Training, Validation and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state =0)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 0)
