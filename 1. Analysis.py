#Brest cancer recogintion -> classification 0-1
#importing libraries
import pandas as pd
import seaborn as sns
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
