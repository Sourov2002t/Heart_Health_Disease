# Import Library

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/content/heart.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import yellowbrick
import pickle

from matplotlib.collections import PathCollection
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from yellowbrick.classifier import PrecisionRecallCurve, ROCAUC, ConfusionMatrix
from yellowbrick.style import set_palette
from yellowbrick.model_selection import LearningCurve, FeatureImportances
from yellowbrick.contrib.wrapper import wrap

# --- Libraries Settings ---
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi']=100
set_palette('dark')

"""# Color Palettes"""

# --- Create List of Color Palletes ---
red_grad = ['#FF0000', '#BF0000', '#800000', '#400000', '#000000']
pink_grad = ['#8A0030', '#BA1141', '#FF5C8A', '#FF99B9', '#FFDEEB']
purple_grad = ['#4C0028', '#7F0043', '#8E004C', '#A80059', '#C10067']
color_mix = ['#F38BB2', '#FFB9CF', '#FFD7D7', '#F17881', '#E7525B']
black_grad = ['#100C07', '#3E3B39', '#6D6A6A', '#9B9A9C', '#CAC9CD']

# --- Plot Color Palletes --
sns.palplot(red_grad)
sns.palplot(pink_grad)
sns.palplot(purple_grad)
sns.palplot(color_mix)
sns.palplot(black_grad)

# Reading Dataset
df = pd.read_csv("/content/heart.csv")

# --- Reading Dataset ---
df.head().style.background_gradient(cmap='Reds').set_properties(**{'font-family': 'Segoe UI'}).hide()

# --- Print Dataset Info ---
print('\033[1m'+'.: Dataset Info :.'+'\033[0m')
print('*' * 30)
print('Total Rows:'+'\033[1m', df.shape[0])
print('\033[0m'+'Total Columns:'+'\033[1m', df.shape[1])
print('\033[0m'+'*' * 30)
print('\n')

# --- Print Dataset Detail ---
print('\033[1m'+'.: Dataset Details :.'+'\033[0m')
print('*' * 30)
df.info(memory_usage = False)

"""
    👉 It can be seen that dataset has successfully imported.<br>
    👉 In the dataset, there are <mark><b>14 columns</b></mark> with <mark><b>1025 observations</b></mark>. Also, there are <mark><b>no null values</b></mark> in this dataset. The <b>details of each variables</b> also can be seen above.<br>
    👉 However, the <mark><b>data types for some columns are not matched</b></mark>. Below will <mark><b>fixed the data types for those column</b></mark> before analysis performed.
"""

# --- Fix Data Types ---
lst=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
df[lst] = df[lst].astype(object)

"""# Initial Data Exploration

# sex (Gender)
"""

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('*' * 25)
print('\033[1m'+'.: Sex (Gender) Total :.'+'\033[0m')
print('*' * 25)
df.sex.value_counts(dropna=False)

"""# CP (Chest Pain Type)"""

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('*' * 30)
print('\033[1m'+'.: Chest Pain Type Total :.'+'\033[0m')
print('*' * 30)
df.cp.value_counts(dropna=False)

"""# FBS (Fasting Blood Sugar)"""

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('*' * 32)
print('\033[1m'+'.: Fasting Blood Sugar Total :.'+'\033[0m')
print('*' * 32)
df.fbs.value_counts(dropna=False)

"""# restecg (Resting Electrocardiographic Results)"""

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('*' * 50)
print('\033[1m'+'.: Resting Electrocardiographic Results Total :.'+'\033[0m')
print('*' * 50)
df.restecg.value_counts(dropna=False)

"""# exang (Exercise Induced Angina)"""

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('*' * 35)
print('\033[1m'+'.: Exercise Induced Angina Total :.'+'\033[0m')
print('*' * 35)
df.exang.value_counts(dropna=False)

''' # slope (Slope of the Peak Exercise) '''

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('*' * 20)
print('\033[1m'+'.: Slope Total :.'+'\033[0m')
print('*' * 20)
df.slope.value_counts(dropna=False)

"""# ca (Number of Major Vessels)"""

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('*' * 40)
print('\033[1m'+'.: Number of Major Vessels Total :.'+'\033[0m')
print('*' * 40)
df.ca.value_counts(dropna=False)

""" # thal"""

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('*' * 20)
print('\033[1m'+'.: "thal" Total :.'+'\033[0m')
print('*' * 20)
df.thal.value_counts(dropna=False)

"""# target (Heart Diseases Status)"""

# --- Count Categorical Labels w/out Dropping Null Walues ---
print('*' * 45)
print('\033[1m'+'.: Heart Diseases Status (target) Total :.'+'\033[0m')
print('*' * 45)
df.target.value_counts(dropna=False)

""" # Numerical Variable
    👉 The second variable that will be explored is <b>numerical variable</b>.

# Descriptive Statistics

    👉 This section will show <b>descriptive statistics</b> of numerical variables.
"""

# --- Descriptive Statistics ---
df.select_dtypes(exclude='object').describe().T.style.background_gradient(cmap='PuRd').set_properties(**{'font-family': 'Segoe UI'})

"""### <div style="font-family: Trebuchet MS; background-color: #FF5C8A; color: #FFFFFF; padding: 12px; line-height: 1.5;"> Continuous Column Distribution 📈</div>
<div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
    👉 This section will show the <b>distribution of numerical variables</b> in histograms, boxplots, Q-Q Plots, skewness and kurtosis values.
</div>
"""

# --- Variable, Color & Plot Size ---
var = 'age'
color = color_mix[0]
fig=plt.figure(figsize=(12, 12))

# --- Skewness & Kurtosis ---
print('\033[1m'+'.: Age Column Skewness & Kurtosis :.'+'\033[0m')
print('*' * 40)
print('Skewness:'+'\033[1m {:.3f}'.format(df[var].skew(axis = 0, skipna = True)))
print('\033[0m'+'Kurtosis:'+'\033[1m {:.3f}'.format(df[var].kurt(axis = 0, skipna = True)))
print('\n')

# --- General Title ---
fig.suptitle('Age Column Distribution', fontweight='bold', fontsize=16,
             fontfamily='sans-serif', color=black_grad[0])
fig.subplots_adjust(top=0.9)

# --- Histogram ---
ax_1=fig.add_subplot(2, 2, 2)
plt.title('Histogram Plot', fontweight='bold', fontsize=14,
          fontfamily='sans-serif', color=black_grad[1])
sns.histplot(data=df, x=var, kde=True, color=color)
plt.xlabel('Total', fontweight='regular', fontsize=11,
           fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Age', fontweight='regular', fontsize=11, fontfamily='sans-serif',
           color=black_grad[1])

plt.show()

"""# trestbps (Resting Blood Pressure in mm Hg) """

# --- Variable, Color & Plot Size ---
var = 'trestbps'
color = color_mix[2]
fig=plt.figure(figsize=(12, 12))

# --- Skewness & Kurtosis ---
print('\033[1m'+'.: Resting Blood Pressure Column Skewness & Kurtosis :.'+'\033[0m')
print('*' * 55)
print('Skewness:'+'\033[1m {:.3f}'.format(df[var].skew(axis = 0, skipna = True)))
print('\033[0m'+'Kurtosis:'+'\033[1m {:.3f}'.format(df[var].kurt(axis = 0, skipna = True)))
print('\n')

# --- General Title ---
fig.suptitle('Resting Blood Pressure Column Distribution', fontweight='bold',
             fontsize=16, fontfamily='sans-serif', color=black_grad[0])
fig.subplots_adjust(top=0.9)

# --- Histogram ---
ax_1=fig.add_subplot(2, 2, 2)
plt.title('Histogram Plot', fontweight='bold', fontsize=14,
          fontfamily='sans-serif', color=black_grad[1])
sns.histplot(data=df, x=var, kde=True, color=color)
plt.xlabel('Total', fontweight='regular', fontsize=11,
           fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Resting Blood Pressure', fontweight='regular', fontsize=11,
           fontfamily='sans-serif', color=black_grad[1])

plt.show()

"""# chol (Serum Cholestoral in mg/dl) """

# --- Variable, Color & Plot Size ---
var = 'chol'
color = color_mix[4]
fig=plt.figure(figsize=(12, 12))

# --- Skewness & Kurtosis ---
print('\033[1m'+'.: Serum Cholestoral Column Skewness & Kurtosis :.'+'\033[0m')
print('*' * 45)
print('Skewness:'+'\033[1m {:.3f}'.format(df[var].skew(axis = 0, skipna = True)))
print('\033[0m'+'Kurtosis:'+'\033[1m {:.3f}'.format(df[var].kurt(axis = 0, skipna = True)))
print('\n')

# --- General Title ---
fig.suptitle('Serum Cholestoral Column Distribution', fontweight='bold',
             fontsize=16, fontfamily='sans-serif', color=black_grad[0])
fig.subplots_adjust(top=0.9)

# --- Histogram ---
ax_1=fig.add_subplot(2, 2, 2)
plt.title('Histogram Plot', fontweight='bold', fontsize=14,
          fontfamily='sans-serif', color=black_grad[1])
sns.histplot(data=df, x=var, kde=True, color=color)
plt.xlabel('Total', fontweight='regular', fontsize=11,
           fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Serum Cholestoral', fontweight='regular', fontsize=11,
           fontfamily='sans-serif', color=black_grad[1])

plt.show()

"""# thalach (Maximum Heart Rate) """

# --- Variable, Color & Plot Size ---
var = 'thalach'
color = purple_grad[1]
fig=plt.figure(figsize=(12, 12))

# --- Skewness & Kurtosis ---
print('\033[1m'+'.: Maximum Heart Rate Column Skewness & Kurtosis :.'+'\033[0m')
print('*' * 50)
print('Skewness:'+'\033[1m {:.3f}'.format(df[var].skew(axis = 0, skipna = True)))
print('\033[0m'+'Kurtosis:'+'\033[1m {:.3f}'.format(df[var].kurt(axis = 0, skipna = True)))
print('\n')

# --- General Title ---
fig.suptitle('Maximum Heart Rate Column Distribution', fontweight='bold',
             fontsize=16, fontfamily='sans-serif', color=black_grad[0])
fig.subplots_adjust(top=0.9)

# --- Histogram ---
ax_1=fig.add_subplot(2, 2, 2)
plt.title('Histogram Plot', fontweight='bold', fontsize=14,
          fontfamily='sans-serif', color=black_grad[1])
sns.histplot(data=df, x=var, kde=True, color=color)
plt.xlabel('Total', fontweight='regular', fontsize=11,
           fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('Maximum Heart Rate', fontweight='regular', fontsize=11,
           fontfamily='sans-serif', color=black_grad[1])

plt.show()

"""# oldpeak """

# --- Variable, Color & Plot Size ---
var = 'oldpeak'
color = red_grad[1]
fig=plt.figure(figsize=(12, 12))

# --- Skewness & Kurtosis ---
print('\033[1m'+'.: "oldpeak" Column Skewness & Kurtosis :.'+'\033[0m')
print('*' * 40)
print('Skewness:'+'\033[1m {:.3f}'.format(df[var].skew(axis = 0, skipna = True)))
print('\033[0m'+'Kurtosis:'+'\033[1m {:.3f}'.format(df[var].kurt(axis = 0, skipna = True)))
print('\n')

# --- General Title ---
fig.suptitle('"oldpeak" Column Distribution', fontweight='bold',
             fontsize=16, fontfamily='sans-serif', color=black_grad[0])
fig.subplots_adjust(top=0.9)

# --- Histogram ---
ax_1=fig.add_subplot(2, 2, 2)
plt.title('Histogram Plot', fontweight='bold', fontsize=14,
          fontfamily='sans-serif', color=black_grad[1])
sns.histplot(data=df, x=var, kde=True, color=color)
plt.xlabel('Total', fontweight='regular', fontsize=11,
           fontfamily='sans-serif', color=black_grad[1])
plt.ylabel('oldpeak', fontweight='regular', fontsize=11,
           fontfamily='sans-serif', color=black_grad[1])

plt.show()

"""# Dataset Pre-processing

# One-Hot Encoding
"""

# --- Creating Dummy Variables for cp, thal and slope ---
cp = pd.get_dummies(df['cp'], prefix='cp')
thal = pd.get_dummies(df['thal'], prefix='thal')
slope = pd.get_dummies(df['slope'], prefix='slope')

# --- Merge Dummy Variables to Main Data Frame ---
frames = [df, cp, thal, slope]
df = pd.concat(frames, axis = 1)

# --- Display New Data Frame ---
df.head().style.background_gradient(cmap='PuRd').hide().set_properties(**{'font-family': 'Segoe UI'})

"""# Dropping Unnecessary Variables

    👉 The <b>variables that unnecessary will be deleted</b>.

"""

# --- Drop Unnecessary Variables ---
df = df.drop(columns = ['cp', 'thal', 'slope'])

# --- Display New Data Frame ---
df.head().style.background_gradient(cmap='Reds').hide().set_properties(**{'font-family': 'Segoe UI'})

"""## <div style="font-family: Trebuchet MS; background-color: #FF5C8A; color: #FFFFFF; padding: 12px; line-height: 1.5;">Features Separating ➗</div>
<div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
    👉 In this section, <mark><b>the 'target' (dependent) column will be seperated</b></mark> from independent columns.
</div>
"""

# --- Seperating Dependent Features ---
x = df.drop(['target'], axis=1)
y = df['target']

"""# Data Normalization 
    👉 In this section, data normalization will be performed to normalize the range of independent variables or features of data.
    👉 Data normalization will use <mark><b>min-max normalization.
    
    📌 <b>Min-max normalization</b> is often known as feature scaling where <mark><b>the values of a numeric range of a feature of data, are reduced to a scale between 0 and 1</b></mark>.
    </blockquote>
"""

# --- Data Normalization using Min-Max Method ---
x = MinMaxScaler().fit_transform(x)

"""## <div style="font-family: Trebuchet MS; background-color: #FF5C8A; color: #FFFFFF; padding: 12px; line-height: 1.5;"> Splitting the Dataset 🪓</div>
<div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
    👉 The dataset will be splitted into <mark><b>80:20 ratio</b></mark> (80% training and 20% testing).
</div>
"""

# --- Splitting Dataset into 80:20 ---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

"""# <div style="font-family: Trebuchet MS; background-color: #BA1141; color: #FFFFFF; padding: 12px; line-height: 1.5;">Model Implementation 🛠</div>
<div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
    👉 This section will implement various machine learning models as mentioned in <b>Introduction</b> section. In addition, explanation for each models will be discussed.
</div>

## <div style="font-family: Trebuchet MS; background-color: #FF5C8A; color: #FFFFFF; padding: 12px; line-height: 1.5;"> Logistic Regression</div>
<div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
"""

# --- Applying Logistic Regression ---
LRclassifier = LogisticRegression(max_iter=1000, random_state=1, solver='liblinear', penalty='l1')
LRclassifier.fit(x_train, y_train)

y_pred_LR = LRclassifier.predict(x_test)

"""<div style="color:white;
            display:fill;
            border-radius:8px;
            background-color:#2b2b2b;
            font-size:120%;
            font-family:sans-serif;
            letter-spacing:0.5px">
    <p style="padding: 8px;color:white;"><b>👉 | Logistic Regression -84% acc</b></p>
</div>
"""

# --- LR Accuracy ---
LRAcc = accuracy_score(y_pred_LR, y_test)
print('.:. Logistic Regression Accuracy:'+'\033[1m {:.2f}%'.format(LRAcc*100)+' .:.')

"""## <div style="font-family: Trebuchet MS; background-color: #FF5C8A; color: #FFFFFF; padding: 12px; line-height: 1.5;"> K-Nearest Neighbour (KNN)</div>
<div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
   
"""

# --- Applying KNN ---
KNNClassifier = KNeighborsClassifier(n_neighbors=3)
KNNClassifier.fit(x_train, y_train)

y_pred_KNN = KNNClassifier.predict(x_test)

"""<div style="color:white;
            display:fill;
            border-radius:8px;
            background-color:#2b2b2b;
            font-size:120%;
            font-family:sans-serif;
            letter-spacing:0.5px">
    <p style="padding: 8px;color:white;"><b>👉 | KNearest Neighbors - 96% acc</b></p>
</div>
"""

# --- KNN Accuracy ---
KNNAcc = accuracy_score(y_pred_KNN, y_test)
print('.:. K-Nearest Neighbour Accuracy:'+'\033[1m {:.2f}%'.format(KNNAcc*100)+' .:.')

"""## <div style="font-family: Trebuchet MS; background-color: #FF5C8A; color: #FFFFFF; padding: 12px; line-height: 1.5;"> Support Vector Machine (SVM)</div>
<div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
"""

# --- Applying SVM ---
SVMclassifier = SVC(kernel='linear', max_iter=1000, C=10, probability=True)
SVMclassifier.fit(x_train, y_train)

y_pred_SVM = SVMclassifier.predict(x_test)

"""<div style="color:white;
            display:fill;
            border-radius:8px;
            background-color:#2b2b2b;
            font-size:120%;
            font-family:sans-serif;
            letter-spacing:0.5px">
    <p style="padding: 8px;color:white;"><b>👉 | SVM - 84% acc</b></p>
</div>
"""

# --- SVM Accuracy ---
SVMAcc = accuracy_score(y_pred_SVM, y_test)
print('.:. Support Vector Machine Accuracy:'+'\033[1m {:.2f}%'.format(SVMAcc*100)+' .:.')

"""## <div style="font-family: Trebuchet MS; background-color: #FF5C8A; color: #FFFFFF; padding: 12px; line-height: 1.5;">Gaussian Naive Bayes</div>
<div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
"""

# --- Applying Gaussian NB ---
GNBclassifier = GaussianNB(var_smoothing=0.1)
GNBclassifier.fit(x_train, y_train)

y_pred_GNB = GNBclassifier.predict(x_test)

"""<div style="color:white;
            display:fill;
            border-radius:8px;
            background-color:#2b2b2b;
            font-size:120%;
            font-family:sans-serif;
            letter-spacing:0.5px">
    <p style="padding: 8px;color:white;"><b>👉 |
Gaussian Naive Bayes - 83% acc</b></p>
</div>
"""

# --- GNB Accuracy ---
GNBAcc = accuracy_score(y_pred_GNB, y_test)
print('.:. Gaussian Naive Bayes Accuracy:'+'\033[1m {:.2f}%'.format(GNBAcc*100)+' .:.')

"""## <div style="font-family: Trebuchet MS; background-color: #FF5C8A; color: #FFFFFF; padding: 12px; line-height: 1.5;"> Decision Tree</div>
<div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
  
"""

# --- Applying Decision Tree ---
DTCclassifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, criterion='entropy', min_samples_split=5,
                                       splitter='random', random_state=1)

DTCclassifier.fit(x_train, y_train)
y_pred_DTC = DTCclassifier.predict(x_test)

"""<div style="color:white;
            display:fill;
            border-radius:8px;
            background-color:#2b2b2b;
            font-size:120%;
            font-family:sans-serif;
            letter-spacing:0.5px">
    <p style="padding: 8px;color:white;"><b>👉 | Decision Tree- 84% acc</b></p>
</div>
"""

# --- Decision Tree Accuracy ---
DTCAcc = accuracy_score(y_pred_DTC, y_test)
print('.:. Decision Tree Accuracy:'+'\033[1m {:.2f}%'.format(DTCAcc*100)+' .:.')

"""## <div style="font-family: Trebuchet MS; background-color: #FF5C8A; color: #FFFFFF; padding: 12px; line-height: 1.5;">Random Forest</div>
<div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
"""

# --- Applying Random Forest ---
RFclassifier = RandomForestClassifier(n_estimators=1000, random_state=1, max_leaf_nodes=20, min_samples_split=15)

RFclassifier.fit(x_train, y_train)
y_pred_RF = RFclassifier.predict(x_test)

"""<div style="color:white;
            display:fill;
            border-radius:8px;
            background-color:#2b2b2b;
            font-size:120%;
            font-family:sans-serif;
            letter-spacing:0.5px">
    <p style="padding: 8px;color:white;"><b>👉 | Random Forest - 89% acc</b></p>
</div>
"""

# --- Random Forest Accuracy ---
RFAcc = accuracy_score(y_pred_RF, y_test)
print('.:. Random Forest Accuracy:'+'\033[1m {:.2f}%'.format(RFAcc*100)+' .:.')

"""## <div style="font-family: Trebuchet MS; background-color: #FF5C8A; color: #FFFFFF; padding: 12px; line-height: 1.5;"> Gradient Boosting</div>
<div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
"""

# --- Applying Gradient Boosting ---
GBclassifier = GradientBoostingClassifier(random_state=1, n_estimators=100, max_leaf_nodes=3, loss='exponential',
                                          min_samples_leaf=20)

GBclassifier.fit(x_train, y_train)
y_pred_GB = GBclassifier.predict(x_test)

"""<div style="color:white;
            display:fill;
            border-radius:8px;
            background-color:#2b2b2b;
            font-size:120%;
            font-family:sans-serif;
            letter-spacing:0.5px">
    <p style="padding: 8px;color:white;"><b>👉 |Gradient Boosting - 87% acc</b></p>
</div>
"""

# --- Gradient Boosting Accuracy ---
GBAcc = accuracy_score(y_pred_GB, y_test)
print('.:. Gradient Boosting Accuracy:'+'\033[1m {:.2f}%'.format(GBAcc*100)+' .:.')

"""## <div style="font-family: Trebuchet MS; background-color: #FF5C8A; color: #FFFFFF; padding: 12px; line-height: 1.5;"> Model Comparison 👀</div>
<div style="font-family: Segoe UI; line-height: 2; color: #000000; text-align: justify">
    👉 After implementing 10 models, this section will <b>compare machine learning models</b>.
</div>
"""

# --- Create Accuracy Comparison Table ---
compare = pd.DataFrame({'Model': ['Logistic Regression', 'K-Nearest Neighbour', 'Support Vector Machine',
                                  'Gaussian Naive Bayes', 'Decision Tree', 'Random Forest', 'Gradient Boosting'
                                  ],
                        'Accuracy': [LRAcc*100, KNNAcc*100, SVMAcc*100, GNBAcc*100, DTCAcc*100, RFAcc*100, GBAcc*100,
                                     ]})

# --- Create Accuracy Comparison Table ---
compare.sort_values(by='Accuracy', ascending=False).style.background_gradient(cmap='PuRd').hide().set_properties(**{'font-family': 'Segoe UI'})

def calculate_sensitivity(y_true, y_predicted):
  confusion_matrix = confusion_matrix(y_true, y_predicted)
  tn, fp, fn, tp = confusion_matrix.ravel()
  sensitivity = tp / (tp + fn)
  return sensitivity

from sklearn.metrics import confusion_matrix

models = {
    "Logistic Regression": LogisticRegression(random_state=0),
    "K-Nearest Neighbour": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
}

# Define dictionaries to store the metrics
accuracy_scores = {}
sensitivity_scores = {}
specificity_scores = {}
precision_scores = {}

# --- Logistic Regression Accuracy ---
LRAcc = accuracy_score(y_pred_LR, y_test)
print('.:. Logistic Regression Accuracy: {:.2f}% .:. \n'.format(LRAcc*100)+' .:.')

# --- Logistic Regression Sensitivity ---
cm_LR = confusion_matrix(y_test, y_pred_LR)
sensitivity_LR = cm_LR[0, 0] / (cm_LR[0, 0] + cm_LR[0, 1])
print('.:. Logistic Regression Sensitivity: {:.2f}% .:. \n'.format(sensitivity_LR*100)+' .:.')

# --- Logistic Regression Specificity ---
specificity_LR = cm_LR[1, 1] / (cm_LR[1, 0] + cm_LR[1, 1])
print('.:. Logistic Regression Specificity: {:.2f}% .:. \n'.format(specificity_LR*100)+' .:.')

# --- Logistic Regression Precision ---
precision_LR = cm_LR[1, 1] / (cm_LR[0, 1] + cm_LR[1, 1])
print('.:. Logistic Regression Precision: {:.2f}% .:. \n'.format(precision_LR*100)+' .:.')

# --- Applying K-Nearest Neighbour ---
KNNClassifier = KNeighborsClassifier(n_neighbors=3)
KNNClassifier.fit(x_train, y_train)
y_pred_KN = KNNClassifier.predict(x_test)

# --- KNN Accuracy ---
KNNAcc = accuracy_score(y_pred_KN, y_test)
print('.:. K-Nearest Neighbour Accuracy:'+'\033[1m {:.2f}% .:. \n'.format(KNNAcc*100)+' .:.')

# --- K-Nearest Neighbour Sensitivity ---
cm_KN = confusion_matrix(y_test, y_pred_KN)
sensitivity_KN = cm_KN[0, 0] / (cm_KN[0, 0] + cm_KN[0, 1])
print('.:. K-Nearest Neighbour Sensitivity:'+'\033[1m {:.2f}% .:. \n'.format(sensitivity_KN*100)+' .:.')

# --- K-Nearest Neighbour Specificity ---
specificity_KN = cm_KN[1, 1] / (cm_KN[1, 0] + cm_KN[1, 1])
print('.:. K-Nearest Neighbour Specificity:'+'\033[1m {:.2f}% .:. \n'.format(specificity_KN*100)+' .:.')

# --- K-Nearest Neighbour Precision ---
precision_KN = cm_KN[1, 1] / (cm_KN[0, 1] + cm_KN[1, 1])
print('.:. K-Nearest Neighbour Precision:'+'\033[1m {:.2f}% .:. \n'.format(precision_KN*100)+' .:.')

# --- Applying Support Vector Machine ---
SVMclassifier = SVC(kernel='linear', max_iter=1000, C=10, probability=True)
SVMclassifier.fit(x_train, y_train)
y_pred_SVM = SVMclassifier.predict(x_test)

# --- Support Vector Machine Accuracy ---
SVMAcc = accuracy_score(y_pred_SVM, y_test)
print('.:. Support Vector Machine Accuracy:'+'\033[1m {:.2f}% .:. \n'.format(SVMAcc*100)+' .:.')

# --- Support Vector Machine Sensitivity ---
cm_SVM = confusion_matrix(y_test, y_pred_SVM)
sensitivity_SVM = cm_SVM[0, 0] / (cm_SVM[0, 0] + cm_SVM[0, 1])
print('.:. Support Vector Machine Sensitivity:'+'\033[1m {:.2f}% .:. \n'.format(sensitivity_SVM*100)+' .:.')

# --- Support Vector Machine Specificity ---
specificity_SVM = cm_SVM[1, 1] / (cm_SVM[1, 0] + cm_SVM[1, 1])
print('.:. Support Vector Machine Specificity:'+'\033[1m {:.2f}% .:. \n'.format(specificity_SVM*100)+' .:.')

# --- Support Vector Machine Precision ---
precision_SVM = cm_SVM[1, 1] / (cm_SVM[0, 1] + cm_SVM[1, 1])
print('.:. Support Vector Machine Precision:'+'\033[1m {:.2f}% .:. \n'.format(precision_SVM*100)+' .:.')

# --- Applying Gaussian Naive Bayes ---
GNBclassifier = GaussianNB(var_smoothing=0.1)
GNBclassifier.fit(x_train, y_train)
y_pred_GNB = GNBclassifier.predict(x_test)

# --- Gaussian Naive Bayes Accuracy ---
GNBAcc = accuracy_score(y_pred_GNB, y_test)
print('.:. Gaussian Naive Bayes Accuracy:'+'\033[1m {:.2f}% .:. \n'.format(GNBAcc*100)+' .:.')

# --- Gaussian Naive Bayes Sensitivity ---
cm_GNB = confusion_matrix(y_test, y_pred_GNB)
sensitivity_GNB = cm_GNB[0, 0] / (cm_GNB[0, 0] + cm_GNB[0, 1])
print('.:. Gaussian Naive Bayes Sensitivity:'+'\033[1m {:.2f}% .:. \n'.format(sensitivity_GNB*100)+' .:.')

# --- Gaussian Naive Bayes Specificity ---
specificity_GNB = cm_GNB[1, 1] / (cm_GNB[1, 0] + cm_GNB[1, 1])
print('.:. Gaussian Naive Bayes Specificity:'+'\033[1m {:.2f}% .:. \n'.format(specificity_GNB*100)+' .:.')

# --- Gaussian Naive Bayes Precision ---
precision_GNB = cm_GNB[1, 1] / (cm_GNB[0, 1] + cm_GNB[1, 1])
print('.:. Gaussian Naive Bayes Precision:'+'\033[1m {:.2f}% .:. \n'.format(precision_GNB*100)+' .:.')

# --- Applying Decision Tree ---
DTCclassifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, criterion='entropy', min_samples_split=5,
                                       splitter='random', random_state=1)
DTCclassifier.fit(x_train, y_train)
y_pred_DT = DTCclassifier.predict(x_test)

# --- Decision Tree Accuracy ---
DTCAcc = accuracy_score(y_pred_DTC, y_test)
print('.:. Decision Tree Accuracy:'+'\033[1m {:.2f}% .:. \n'.format(DTCAcc*100)+' .:.')

# --- Decision Tree Sensitivity ---
cm_DT = confusion_matrix(y_test, y_pred_DT)
sensitivity_DT = cm_DT[0, 0] / (cm_DT[0, 0] + cm_DT[0, 1])
print('.:. Decision Tree Sensitivity:'+'\033[1m {:.2f}% .:. \n'.format(sensitivity_DT*100)+' .:.')

# --- Decision Tree Specificity ---
specificity_DT = cm_DT[1, 1] / (cm_DT[1, 0] + cm_DT[1, 1])
print('.:. Decision Tree Specificity:'+'\033[1m {:.2f}% .:. \n'.format(specificity_DT*100)+' .:.')

# --- Decision Tree Precision ---
precision_DT = cm_DT[1, 1] / (cm_DT[0, 1] + cm_DT[1, 1])
print('.:. Decision Tree Precision:'+'\033[1m {:.2f}% .:. \n'.format(precision_DT*100)+' .:.')

# --- Applying Random Forest ---
RFclassifier = RandomForestClassifier(n_estimators=1000, random_state=1, max_leaf_nodes=20, min_samples_split=15)
RFclassifier.fit(x_train, y_train)
y_pred_RF = RFclassifier.predict(x_test)

# --- Random Forest Accuracy ---
RFAcc = accuracy_score(y_pred_RF, y_test)
print('.:. Random Forest Accuracy:'+'\033[1m {:.2f}% .:. \n'.format(RFAcc*100)+' .:.')


# --- Random Forest Sensitivity ---
cm_RF = confusion_matrix(y_test, y_pred_RF)
sensitivity_RF = cm_RF[0, 0] / (cm_RF[0, 0] + cm_RF[0, 1])
print('.:. Random Forest Sensitivity:'+'\033[1m {:.2f}% .:. \n'.format(sensitivity_RF*100)+' .:.')

# --- Random Forest Specificity ---
specificity_RF = cm_RF[1, 1] / (cm_RF[1, 0] + cm_RF[1, 1])
print('.:. Random Forest Specificity:'+'\033[1m {:.2f}% .:. \n'.format(specificity_RF*100)+' .:.')

# --- Random Forest Precision ---
precision_RF = cm_RF[1, 1] / (cm_RF[0, 1] + cm_RF[1, 1])
print('.:. Random Forest Precision:'+'\033[1m {:.2f}% .:. \n'.format(precision_RF*100)+' .:.')

# --- Applying Gradient Boosting ---
GBclassifier = GradientBoostingClassifier(random_state=1, n_estimators=100, max_leaf_nodes=3, loss='exponential',
                                          min_samples_leaf=20)
GBclassifier.fit(x_train, y_train)
y_pred_GB = GBclassifier.predict(x_test)

# --- Gradient Boosting Accuracy ---
GBAcc = accuracy_score(y_pred_GB, y_test)
print('.:. Gradient Boosting Accuracy:'+'\033[1m {:.2f}% .:. \n'.format(GBAcc*100)+' .:.')

# --- Gradient Boosting Sensitivity ---
cm_GB = confusion_matrix(y_test, y_pred_GB)
sensitivity_GB = cm_GB[0, 0] / (cm_GB[0, 0] + cm_GB[0, 1])
print('.:. Gradient Boosting Sensitivity:'+'\033[1m {:.2f}% .:. \n'.format(sensitivity_GB*100)+' .:.')

# --- Gradient Boosting Specificity ---
specificity_GB = cm_GB[1, 1] / (cm_GB[1, 0] + cm_GB[1, 1])
print('.:. Gradient Boosting Specificity:'+'\033[1m {:.2f}% .:. \n'.format(specificity_GB*100)+' .:.')

# --- Gradient Boosting Precision ---
precision_GB = cm_GB[1, 1] / (cm_GB[0, 1] + cm_GB[1, 1])
print('.:. Gradient Boosting Precision:'+'\033[1m {:.2f}% .:. \n'.format(precision_GB*100)+' .:.')

# --- Create Accuracy Comparison Table ---
compare = pd.DataFrame({'Model': ['Logistic Regression', 'K-Nearest Neighbour', 'Support Vector Machine',
                                  'Gaussian Naive Bayes', 'Decision Tree', 'Random Forest', 'Gradient Boosting'
                                  ],
                        'Accuracy': [LRAcc*100, KNNAcc*100, SVMAcc*100, GNBAcc*100, DTCAcc*100, RFAcc*100, GBAcc*100
                                     ]})

# --- Create Accuracy Comparison Table ---
compare.sort_values(by='Accuracy', ascending=False).style.background_gradient(cmap='PuRd').hide().set_properties(**{'font-family': 'Segoe UI'})

# --- Create Sensitivity Comparison Table ---
compare = pd.DataFrame({'Model': ['Logistic Regression', 'K-Nearest Neighbour', 'Support Vector Machine',
                                  'Gaussian Naive Bayes', 'Decision Tree', 'Random Forest', 'Gradient Boosting'
                                  ],
                        'Sensitivity': [sensitivity_LR*100, sensitivity_KN*100, sensitivity_SVM*100, sensitivity_GNB*100, sensitivity_DT*100, sensitivity_RF*100, sensitivity_GB*100
                                     ]})

# --- Create Sensitivity Comparison Table ---
compare.sort_values(by='Sensitivity', ascending=False).style.background_gradient(cmap='PuRd').hide().set_properties(**{'font-family': 'Segoe UI'})

# --- Create Specificity Comparison Table ---
compare = pd.DataFrame({'Model': ['Logistic Regression', 'K-Nearest Neighbour', 'Support Vector Machine',
                                  'Gaussian Naive Bayes', 'Decision Tree', 'Random Forest', 'Gradient Boosting'
                                  ],
                        'Specificity': [specificity_LR*100, specificity_KN*100, specificity_SVM*100, specificity_GNB*100, specificity_DT*100, specificity_RF*100, specificity_GB*100
                                     ]})

# --- Create Specificity Comparison Table ---
compare.sort_values(by='Specificity', ascending=False).style.background_gradient(cmap='PuRd').hide().set_properties(**{'font-family': 'Segoe UI'})

# --- Create Precision Comparison Table ---
compare = pd.DataFrame({'Model': ['Logistic Regression', 'K-Nearest Neighbour', 'Support Vector Machine',
                                  'Gaussian Naive Bayes', 'Decision Tree', 'Random Forest', 'Gradient Boosting'
                                  ],
                        'Precision': [precision_LR*100, precision_KN*100, precision_SVM*100, precision_GNB*100, precision_DT*100, precision_RF*100, precision_GB*100
                                     ]})

# --- Create Precision Comparison Table ---
compare.sort_values(by='Precision', ascending=False).style.background_gradient(cmap='PuRd').hide().set_properties(**{'font-family': 'Segoe UI'})

# Define the models and their corresponding metrics
models = ['Logistic Regression', 'K-Nearest Neighbour', 'Support Vector Machine',
          'Gaussian Naive Bayes', 'Decision Tree', 'Random Forest', 'Gradient Boosting']

accuracy = [83.902439, 95.609756, 83.902439, 82.439024, 83.902439, 88.780488, 86.829268]
sensitivity = [76.635514, 85.981308, 73.831776, 82.242991, 100.000000, 82.242991, 81.308411]
specificity = [91.836735, 84.693878, 94.897959, 82.653061, 100.000000, 95.918367, 92.857143]
precision = [78.260870, 84.693878, 76.859504, 81.000000, 100.000000, 83.185841, 81.981982]

# Set the width of the bars
bar_width = 0.1

# Set the position of the bars on the x-axis
r1 = np.arange(len(models))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Create the bars
plt.bar(r1, accuracy, color='skyblue', width=bar_width, edgecolor='grey', label='Accuracy')
plt.bar(r2, sensitivity, color='salmon', width=bar_width, edgecolor='grey', label='Sensitivity')
plt.bar(r3, specificity, color='lightgreen', width=bar_width, edgecolor='grey', label='Specificity')
plt.bar(r4, precision, color='orange', width=bar_width, edgecolor='grey', label='Precision')

# Add xticks on the middle of the group bars
plt.xlabel('Models', fontweight='bold')
plt.xticks([r + bar_width * 1.5 for r in range(len(models))], models, rotation=45, ha='right')

# Add y label
plt.ylabel('Percentage', fontweight='bold')

# Add a legend
plt.legend()

# Show the graph
plt.show()
