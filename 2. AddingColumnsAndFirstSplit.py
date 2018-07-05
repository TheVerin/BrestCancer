import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#Adding columns
RawData['distortion_area'] = (RawData['area_worst']/RawData['area_mean'])
RawData['distortion_radius'] = (RawData['radius_worst']/RawData['radius_mean'])

sns.stripplot(data = RawData, x = 'distortion_area', y = 'distortion_radius', hue = 'diagnosis')

#Splitting for Independent and Dependent variables
Y = RawData.iloc[:, 0]
X = RawData.iloc[:, 1:]