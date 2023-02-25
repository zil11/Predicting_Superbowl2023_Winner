# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 19:49:38 2023

@author: jeffe
"""

# Import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from statistics import mean
 
# Read Excel File
df = pd.read_excel(r'C:\Users\jeffe\Desktop\PITT\ECON 2824 Big Data and Forcasting in Econ\Assignments\Assignment2\mydf.xlsx')

# Standardize the data
scaler = StandardScaler()
df[['winlose_perc_a', 'score_team_a','score_opp_a','points_diff_a','MoV_a','SoS_a','SRS_a','OSRS_a','DSRS_a','win_to_lose_perc', 'score_team','score_opp','points_diff','MoV','SoS','SRS','OSRS','DSRS']] = StandardScaler().fit_transform(df[['winlose_perc_a', 'score_team_a','score_opp_a','points_diff_a','MoV_a','SoS_a','SRS_a','OSRS_a','DSRS_a','win_to_lose_perc', 'score_team','score_opp','points_diff','MoV','SoS','SRS','OSRS','DSRS']])
df.head()

df2 = df[pd.notnull(df['afl_win'])]
row1 = df[['winlose_perc_a', 'score_team_a','score_opp_a','points_diff_a','MoV_a','SoS_a','SRS_a','OSRS_a','DSRS_a','win_to_lose_perc', 'score_team','score_opp','points_diff','MoV','SoS','SRS','OSRS','DSRS']].iloc[0].tolist()
df = df2

# Summary statistics
stat = df.describe()

# Check correlation
cor = df.corr()

# Check the counts of outcome variable (AFL wins = 1, 0 otherwise)
sns.countplot(x='afl_win', data=df, palette='hls')
plt.show()

# Split dataframe into train/test dataset

features = ['winlose_perc_a', 'score_team_a','score_opp_a','points_diff_a','MoV_a','SoS_a','SRS_a','OSRS_a','DSRS_a','win_to_lose_perc', 'score_team','score_opp','points_diff','MoV','SoS','SRS','OSRS','DSRS']
x = df.loc[:,features]
y = df.loc[:,['afl_win']]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0, train_size=0.8)

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# define model
from numpy import arange
model = LassoCV(alphas=arange(0, 1, 0.01), cv=cv, n_jobs=-1)
# fit model
model.fit(x, y)
# summarize chosen configuration
print('alpha: %f' % model.alpha_)

# make a prediction
yhat = model.predict([row1])

# summarize prediction
print('Predicted: %.3f' % yhat)
print(model.coef_)


# Second attempt using logistic regression with 10 fold cross validation 
model2 = LogisticRegressionCV(Cs = 10, cv=cv, penalty='l1', solver='saga', random_state=1)

model2.fit(x,y)

yhat2 = model2.predict_proba([row1])
print(model2.coef_)
print(yhat2)
model2.score(x,y)


