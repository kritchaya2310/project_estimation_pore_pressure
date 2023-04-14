# Estimation of Pore Pressure using Machine Learning
In this project, we aim to predict pore pressure using several independent variables with the help of machine learning. Pore pressure prediction is a crucial aspect of oil and gas exploration and production, as sudden changes in pore pressure can cause major drilling problems and reduce well stability. Accurate pore pressure prediction can help to reduce drilling risk/hazard, increase wellbore stability, optimize casing seat selection, and design mud programs.

## Timeline of the Project
The project was carried out in the following stages:

1. Importing Data: In this stage, we collected data from various sources related to pore pressure and other independent variables.
2. Data Analysis: After importing data, we performed exploratory data analysis to understand the data better.
3. Data Processing: In this stage, we processed the data to prepare it for the machine learning models. This involved dealing with missing values, outliers, and data normalization.
4. Model Building: We built machine learning models using several algorithms such as Random Forest, XGB Regressor, and SVM.
5. Hypertuning of Models: The models were tuned using hyperparameters to improve their performance.
## Results
We obtained the following results:

- Random Forest Model: 96% accuracy in predicting pore pressure.
- XGB Regressor: 94% accuracy in predicting pore pressure.
- SVM: Only around 60-70% accuracy in predicting pore pressure.
- Effective stress was also calculated using mathematical notions and predicted using ML algorithms with an accuracy of about 97%.
## Conclusion
The project demonstrated the effectiveness of machine learning algorithms in predicting pore pressure and effective stress. The results obtained from the Random Forest and XGB Regressor models were promising and could be used in the oil and gas industry to reduce drilling risk and increase wellbore stability.
## Installation

Install my-project with :

```bash
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import sklearn.metrics as metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn import svm

from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from pycaret.regression import *

```
    