from requests import get
from io import BytesIO
from zipfile import ZipFile
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pylab import rcParams
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import os
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.holtwinters as ets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import chi2
from pylab import rcParams

from Final_toolbox import *

warnings.filterwarnings("ignore")
register_matplotlib_converters()




#=================================================================================================
# Question 6
# a. Pre-processing dataset: Dataset cleaning for missing observation. 
#    You must follow the data cleaning techniques for time series dataset.
# b. Plot of the dependent variable versus time.
# c. ACF/PACF of the dependent variable.
# d. Correlation Matrix with seaborn heatmap with the Pearson’s correlation coefficient.
# e. Split the dataset into train set (80%) and test set (20%).
#=================================================================================================

# if "AirQualityUCI" not in os.listdir():
#     request = get('https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip')
#     zip_file = ZipFile(BytesIO(request.content))
#     zip_file.extractall()
# print('\n')
df = pd.read_csv("AirQualityUCI.csv", sep = ';',infer_datetime_format=True)

#=================================================================================================
# Question 6
# a. Pre-processing dataset: Dataset cleaning for missing observation. 
#    You must follow the data cleaning techniques for time series dataset.
#=================================================================================================

# Print the head of the dataset
print(df.head(10))

# Description of the data
df.describe()
df.shape #9471 rows and 17 colums

# Summary of dataframe
print('Summary of Dataframe:\n',df.info)

# Changing the datatype from object to float: number that is not an integer
df['CO(GT)'] = df['CO(GT)'].str.replace(',', '.').astype(float)
df['C6H6(GT)'] = df['C6H6(GT)'].str.replace(',','.').astype(float)
df['T'] = df['T'].str.replace(',', '.').astype(float)
df['RH'] = df['RH'].str.replace(',', '.').astype(float)
df['AH'] = df['AH'].str.replace(',', '.').astype(float)

# Drop the Unnamed columns
df = df.drop(['Unnamed: 15','Unnamed: 16'], axis = 1)
# Null values per feature
print(df.isnull().sum())
# Removing null values
null_data = df[df.isnull().any(axis=1)] #all null values
print(null_data.head())
df= df.dropna()
print('Shape after null values:\n',df.shape) #(9357, 15)
# Replacing -200 with nan
df = df.replace(-200,np.nan)
print(df.isnull().sum())
# Appending date and time
print(df.index)
df.loc[:,'Datetime'] = df['Date'] + ' ' + df['Time']
DateTime = []
for x in df['Datetime']:
    DateTime.append(datetime.strptime(x,'%d/%m/%Y %H.%M.%S'))
datetime = pd.Series(DateTime)
df.index = datetime
# print(df.head())
# print('AFTER',df.dtypes)
df = df.replace(-200, np.nan)

# Process for NaN values: fill this NaN values with the mean
# Creating processed dataframe
print(df.isnull().sum())
print(df.head)
# SD = df['Date']
# ST = df['Time']
S0 = df['CO(GT)'].fillna(df['PT08.S1(CO)'].mean())
S1 = df['PT08.S1(CO)'].fillna(df['PT08.S1(CO)'].mean())
S2 = df['NMHC(GT)'].fillna(df['NMHC(GT)'].mean())
S3 = df['C6H6(GT)'].fillna(df['C6H6(GT)'].mean())
S4 = df['PT08.S2(NMHC)'].fillna(df['PT08.S1(CO)'].mean())
S5 = df['NOx(GT)'].fillna(df['NOx(GT)'].mean())
S6 = df['PT08.S3(NOx)'].fillna(df['PT08.S1(CO)'].mean())
S7 = df['NO2(GT)'].fillna(df['NO2(GT)'].mean())
S8 = df['PT08.S4(NO2)'].fillna(df['PT08.S1(CO)'].mean())
S9 = df['PT08.S5(O3)'].fillna(df['PT08.S1(CO)'].mean())
S10 = df['T'].fillna(df['T'].mean())
S11 = df['RH'].fillna(df['RH'].mean())
S12 = df['AH'].fillna(df['AH'].mean())
print('Handling nan with mean\n',df.isnull().sum())
print('\n')

#This values does not have any NaN values
df = pd.DataFrame({'CO(GT)':S0,'PT08.S1(CO)':S1,'NMHC(GT)':S2, 'C6H6(GT)':S3, 'PT08.S2(NMHC)':S4, 'NOx(GT)':S5,
                   'PT08.S3(NOx)':S6, 'NO2(GT)':S7,  'PT08.S4(NO2)':S8, 'PT08.S5(O3)':S9, 'T':S10, 'RH':S11, 'AH':S12 })


print("Shape after preprocessing:\n",df.shape) #(9357, 13)

#=================================================================================================
# Question 6
# b. Plot of the dependent variable versus time.
#=================================================================================================
# Created Dataframe for Dependent variable and time
df_rh = pd.DataFrame({'RH':S11})

# df.to_csv("AirQuality_processed_rh.csv")
print('Dataframe for Dependent variable and time\n',df_rh.head())

plt.figure(figsize=(15,10))
plt.plot(df_rh,  label = 'RH')
plt.xlabel('Time: March 2004- February 2005', fontsize=22)
plt.ylabel('Relative Humidity (RH)', fontsize=22)
plt.title('Relative humidity over time', fontsize=22)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
#plt.legend(loc='best')
plt.show()


#=================================================================================================
# Question 6
# c. ACF/PACF of the dependent variable.
#=================================================================================================

ACF_PACF_Plot(y = df_rh,
              title1 = 'ACF - Relative Humidity ',
              title2 = 'PACF -Relative Humidity ',
              nlags = 50)


#=================================================================================================
# Question 6
# d. Correlation Matrix with seaborn heatmap with the Pearson’s correlation coefficient.
#=================================================================================================

plt.figure()
fig, ax = plt.subplots(figsize=(12,12))
heatmap=sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True,cmap="mako",linewidth=0.3, linecolor='w')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=17);
plt.show()

#=================================================================================================
# Question 6
# e. Split the dataset into train set (80%) and test set (20%).
#=================================================================================================

# Always consider suffle=False, so the time dependency is not lost:
# Split the training and testing in 80% and 20%

train, test = train_test_split(df, shuffle=False, test_size = 0.2)

#Train and test Shape
print('Train Shape:',train.shape) #(7485, 13)
print('Test Shape:',test.shape) #(1872, 13)



#=================================================================================================
# Question 7
# Stationarity: Check for a need to make the dependent variable stationary.
# If the dependent variable is not stationary, you need to use the techniques discussed in class to make it stationary.
# Perform ACF/PACF analysis for stationarity.
# You need to perform ADF-test & kpss-test and plot the rolling mean and variance for the raw data and the transformed data.
# =================================================================================================

test_result = adfuller(df['RH'])

# ADF TEST
ADF_Cal(df['RH'])

# KPSS TEST
KPSS_test(df["RH"])

# ROLLING MEAN AND VAR
rolling_mean, rolling_var = cal_rolling_mean_var(df["RH"],start="2004-03-10 18:00:00", end="2005-04-04 14:00:00")

###### 1st Difference:
# DIFFERENCING RH variable
#difference_RH = differencing(df['RH'], 1)

#seasonal_diff_24
difference_RH_seasonal = differencing_l(df['RH'], 24)

# ADF TEST
#ADF_Cal(difference_RH)
ADF_Cal(difference_RH_seasonal)

# KPSS TEST
#KPSS_test(difference_RH)
KPSS_test(difference_RH_seasonal)

# ROLLING MEAN AND VAR
#rolling_mean, rolling_var = cal_rolling_mean_var(difference_RH,start="2004-03-10 18:00:00", end="2005-04-04 14:00:00")
rolling_mean, rolling_var = cal_rolling_mean_var(difference_RH_seasonal,start="2004-03-10 18:00:00", end="2005-04-04 14:00:00")


# ACF and PACF of DIFFERENCE_RH
#ACF_PACF_Plot(difference_RH, 'ACF Difference of RH variable', 'PACF Difference of RH variable' , 50)
ACF_PACF_Plot(y = difference_RH_seasonal,
              title1 = 'ACF - Seassonal Difference RH',
              title2 = 'PACF -Seassonal Difference RH',
              nlags = 50)

# ####### 2nd Difference
# # DIFFERENCING RH variable
# difference_2_RH = differencing(difference_RH, 1)
# # ADF TEST
# ADF_Cal(difference_2_RH)
#
# # KPSS TEST
# KPSS_test(difference_2_RH)
#
# # ROLLING MEAN AND VAR
# rolling_mean, rolling_var = cal_rolling_mean_var(difference_2_RH,start="2004-03-10 18:00:00", end="2005-04-04 14:00:00")
#
# # ACF and PACF of 1ST DIFFERENCE_RH
# ACF_PACF_Plot(difference_2_RH, 'ACF Difference of RH variable', 'PACF Difference of RH variable', 50)




#=================================================================================================
# Question 8
# Time series Decomposition: Approximate the trend and the seasonality and plot the detrended
# and the seasonally adjusted data set.
# Find the out the strength of the trend and seasonality.
# Refer to the lecture notes for different type of time series decomposition techniques.
# =================================================================================================

rcParams['figure.figsize'] = 16, 10
decomposition = sm.tsa.seasonal_decompose(train["RH"], model='additive')

fig = decomposition.plot()
plt.title('Additive Residuals')
plt.show()

# rcParams['figure.figsize'] = 16, 10
# decomposition = sm.tsa.seasonal_decompose(train["RH"], model='multiplicative')
# fig = decomposition.plot()
# plt.title('Multiplicative Residuals')
# plt.show()

y = df['RH'].astype(float)
print(y)
STL = STL(y)
res = STL.fit()
fig = res.plot()
# plt.fig(figsize=(16,10))
plt.show()

T = res.trend
S = res.seasonal
R = res.resid

plt.figure(figsize=(16, 10))
plt.plot(T, label='trend')
plt.plot(S, label='Seasonal')
plt.plot(R, label='residuals')
plt.xlabel('Year', fontsize=16)
plt.ylabel('Magnitude', fontsize=16)
plt.title('Trend, Seasonality, Residual components using STL Decomposition', fontsize=16)
plt.legend()
plt.show()

adjusted_seasonal = y - S
plt.figure(figsize=(16, 10))
plt.plot(y[:50], label='Original')
plt.plot(adjusted_seasonal[:50], label='Seasonally Adjusted')
plt.xlabel('Date', fontsize=16)
plt.ylabel('Magnitude', fontsize=16)
plt.title('Original vs Seasonally adjusted', fontsize=20)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.legend(loc='best', fontsize=15)
plt.show()

# Measuring strength of trend and seasonality
F = np.max([0, 1 - np.var(np.array(R)) / np.var(np.array(T + R))])
print('Strength of trend for Air quality dataset is', round(F, 3))

FS = np.max([0, 1 - np.var(np.array(R)) / np.var(np.array(S + R))])
print('Strength of seasonality for Air quality dataset is', round(FS, 3))

# =================================================================================================
# QUESTION 9 :
# Holt-Winters method: Using the Holt-Winters method try to find the best fit
# using the train dataset and make a prediction using the test set.
# =================================================================================================

# Holt's Winter Seasonal Trend
# ===============================
print('=' * 20, 'HOLT WINTERS METHOD', '=' * 20)

holt_winter_model = ExponentialSmoothing(train["RH"], seasonal='mul').fit()
# holt_winter_model = ExponentialSmoothing(train["RH"],  seasonal='multiplicative', trend='multiplicative').fit()

hw_train_pred = holt_winter_model.fittedvalues
hw_test_pred = list(holt_winter_model.forecast(len(test["RH"])))

# holt winter MSE and RESIDUAL ERROR
# =====================================
hw_residual_error_test = np.subtract(test["RH"].values, np.array(hw_test_pred))
hw_residual_error_train = np.subtract(train["RH"].values, np.array(hw_train_pred))

# Holt Winter mse
hw_mse_test = mean_squared_error(test["RH"].values, hw_test_pred)
hw_mse_train = mean_squared_error(train["RH"].values, hw_train_pred)

# holt winter rmse
hw_rmse_train = np.sqrt(hw_mse_train)
hw_rmse_test = np.sqrt(hw_mse_test)

# holt winter residual variance
hw_residual_variance_test = np.var(hw_residual_error_test)

# holt winter residual mean
hw_residual_mean_test = np.mean(hw_residual_error_test)

print("Holt Winter Method: MSE  of prediction errors (Train): ", hw_mse_train)
print("Holt Winter Method: RMSE of prediction errors (Train): ", hw_rmse_train)

print("Holt Winter Method: MSE  of forecast errors (Test): ", hw_mse_test)
print("Holt Winter Method: RMSE of forecast errors (Test): ", hw_rmse_test)

print("Holt Winter Method: Variance of Residual of forecast (Test) :", hw_residual_variance_test)
print("Holt Winter Method: Mean of Residual of forecast (Test):", hw_residual_mean_test)
print('\n')

# holt winter residual ACF


hw_residual_error_test_ACF = calc_acf(hw_residual_error_test, len(hw_test_pred))

# calculate ACF
# ===============================================
####  inbuilt function
# fig = plt.figure()
# plot_acf( hw_residual_error_test_ACF,  ax = plt.gca(), lags=len(hw_residual_error_test_ACF)-1)
# plt.title('ACF of Residuals Error', fontsize=15)
# plt.tick_params(axis='x', labelsize=12)
# plt.tick_params(axis='y', labelsize=12)
# plt.show()

plot_acfx(hw_residual_error_test_ACF, "ACF plot using Holt Winter Residuals")

# Calculate Q value for Residual Error
# Q = len(test['RH'])*np.sum(np.square(hw_residual_error_test_ACF[100:]))
# print('The Q value of Holt Winter forecast is ', Q)


# =================================================================================================
# Question 10
# Feature selection: You need to have a section in your report that explains how the feature
# selection was performed and whether the collinearity exits not.
# Backward stepwise regression along with SVD and condition number is needed.
# You must explain that which feature(s) need to be eliminated and why.
# You are welcome to use other methods like PCA or random forest for feature elimination.
# =================================================================================================


# Divide the dataset in features and target
x = df[['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)',
        'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'AH']]
y = df[['RH']]

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.2)

# ===============================
# Feature Selection
# ===============================
# simgular values
from numpy import linalg as LA

X = x_train.values
H = np.matmul(X.T, X)
s, d, v = np.linalg.svd(H)
print('SingularValues = ', d)  #

condition_num = LA.cond(X)
print('The condition number is ', condition_num)  # features are correlated in this dataset

# **************** All Variables ****************
model_0 = sm.OLS(y_train, x_train).fit()
print(model_0.summary())

# **************** Removing PT08.S5(O3) ****************
x_train.drop(columns='PT08.S5(O3)', axis=1, inplace=True)
model_1 = sm.OLS(y_train, x_train).fit()
print(model_1.summary())

# **************** Removing  NMHC(GT) ******************
x_train.drop(columns='NMHC(GT)', axis=1, inplace=True)
model_2 = sm.OLS(y_train, x_train).fit()
print(model_2.summary())

# =================================================================================================
# Question 11
# Base-models: average, naïve, drift, simple and exponential smoothing.
# You need to perform an h-step prediction based on the base models and compare the SARIMA model performance with
# the base model predication.
# =================================================================================================

# Performance for all the models
base_model_columns = ["Model", "MSE", "RMSE","Residual Mean","Residual Variance", 
                      "Train Residual Mean","Train Residual Variance", "Q Value"] 
base_model_results = pd.DataFrame(columns=base_model_columns)

# ===============================================#
# **************** AVERAGE METHOD ***************#
# ===============================================#
print(20 * "=" + " AVERAGE METHOD " + 20 * "=")


# Compute Prediction
average_pred_test = average_method(train["RH"], len(test["RH"]))
average_pred_train = average_method(train["RH"], len(train["RH"]))

# Compute Residual Error
average_residual_test = np.subtract(test["RH"].values, np.array(average_pred_test))
average_residual_train = np.subtract(train["RH"].values, np.array(average_pred_train))

# Compute Residual Variance 
average_residual_variance = np.var(average_residual_test)
average_residual_variance_train = np.var(average_residual_train)
# Compute Residual Mean
average_residual_mean = np.mean(average_residual_test)
average_residual_mean_train = np.mean(average_residual_train)

# Compute MSE
average_mse_test = mean_squared_error(test["RH"].values, average_pred_test)
# Compute RMSE
average_rmse_test = np.sqrt(average_mse_test)

# Average residual ACF
average_residual_acf = calc_acf(average_residual_test, len(average_pred_test))

k = len(train) + len(test)
average_Q_value = k * np.sum(np.array(average_residual_acf) ** 2)

print("The MSE for Average model is : ", round(average_mse_test, 4))
print("The RMSE for Average model is: ", round(average_rmse_test, 4))
print("The Variance of residual for Average model is: ", round(average_residual_variance, 4))
print("The Mean of residual for Average model is: ", round(average_residual_mean, 4))
print('The Q value of Residual for Average model is: ', round(average_Q_value, 4))

print(60 * "=")

plot_acfx(average_residual_acf, "ACF plot using Average Residuals")

# add the results to common dataframe
values = ["Average Model",
          average_mse_test,
          average_rmse_test,
          average_residual_mean,
          average_residual_variance,
          average_residual_mean_train,
          average_residual_variance_train,
          average_Q_value]

base_model_results = base_model_results.append(pd.DataFrame([values], columns=base_model_columns))

# plot the predicted vs actual data
average_df = test.copy(deep=True)
average_df["RH"] = average_pred_test

plot_multiline_chart_pandas_using_index([train, test, average_df], "RH",
                                        ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
                                        "Time", "RH",
                                        "Air Quality UCI Prediction Using Average Model",
                                        rotate_xticks=True)

# =====================================#
# *********** NAIVE METHOD ***********#
# =====================================#
print(20 * "=" + "NAIVE MODEL" + 20 * "=")


# Compute Prediction
naive_pred_test = naive_method(train["RH"], len(test["RH"]))
naive_pred_train = naive_method(train["RH"], len(train["RH"]))

# Compute Residual Error
naive_residual_test = np.subtract(test["RH"].values, np.array(naive_pred_test))
naive_residual_train = np.subtract(train["RH"].values, np.array(naive_pred_train))
# Compute Residual Variance 
naive_residual_variance = np.var(naive_residual_test)
naive_residual_variance_train = np.var(naive_residual_train)
# Compute Residual Mean
naive_residual_mean = np.mean(naive_residual_test)
naive_residual_mean_train = np.mean(naive_residual_train)# Compute MSE
naive_mse_test = mean_squared_error(test["RH"].values, naive_pred_test)
# Compute RMSE
naive_rmse_test = np.sqrt(naive_mse_test)

# Average residual ACF
naive_residual_acf = calc_acf(naive_residual_test, len(naive_pred_test))

k = len(train) + len(test)
naive_Q_value = k * np.sum(np.array(naive_residual_acf) ** 2)

print("The MSE for Naive Model is : ", round(naive_mse_test, 4))
print("The RMSE for Naive Model is: ", round(naive_rmse_test, 4))
print("The Variance of residual for Naive Model is: ", round(naive_residual_variance, 4))
print("The Mean of residual for Naive Model is: ", round(naive_residual_mean, 4))
print('The Q value of Residual for Naive Model is: ', round(naive_Q_value, 4))

print(60 * "=")

plot_acfx(naive_residual_acf, "ACF plot using Naive Residuals")

# add the results to common dataframe
values = ["Naive Model",
          naive_mse_test,
          naive_rmse_test,
          naive_residual_mean,
          naive_residual_variance,
          naive_residual_mean_train,
          naive_residual_variance_train,
          naive_Q_value]

base_model_results = base_model_results.append(pd.DataFrame([values], columns=base_model_columns))

# plot the predicted vs actual data
naive_df = test.copy(deep=True)
naive_df["RH"] = naive_pred_test

plot_multiline_chart_pandas_using_index([train, test, naive_df], "RH",
                                        ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
                                        "Time", "RH",
                                        "Air Quality UCI Prediction Using Naive Model",
                                        rotate_xticks=True)

# =====================================#
# *********** DRIFT METHOD ***********#
# =====================================#

print(20 * "=" + " DRIFT MODEL " + 20 * "=")

# Compute Prediction
drift_pred_test = drift_method(train["RH"], len(test["RH"]))
drift_pred_train = drift_method(train["RH"], len(train["RH"]))
# Compute Residual Error
drift_residual_test = np.subtract(test["RH"].values, np.array(drift_pred_test))
drift_residual_train = np.subtract(train["RH"].values, np.array(drift_pred_train))
# Compute Residual Variance 
drift_residual_variance = np.var(drift_residual_test)
drift_residual_variance_train = np.var(drift_residual_train)
# Compute Residual Mean
drift_residual_mean = np.mean(drift_residual_test)
drift_residual_mean_train = np.mean(drift_residual_train)
# Compute MSE
drift_mse_test = mean_squared_error( test["RH"].values, drift_pred_test)
# Compute RMSE
drift_rmse_test = np.sqrt(drift_mse_test)

# Average residual ACF
drift_residual_acf = calc_acf(drift_residual_test, len(drift_pred_test))

k = len(train) + len(test)
drift_Q_value = k * np.sum(np.array(drift_residual_acf) ** 2)

print("The MSE for Drift Model is : ", round(drift_mse_test, 4))
print("The RMSE for Drift Model is: ", round(drift_rmse_test, 4))
print("The Variance of residual for Drift Model is: ", round(drift_residual_variance, 4))
print("The Mean of residual for Drift Model is: ", round(drift_residual_mean, 4))
print('The Q value of Residual for Drift Model is: ', round(drift_Q_value, 4))

print(60 * "=")
plot_acfx(drift_residual_acf, "ACF plot using Drift Residuals")

# add the results to common dataframe
values = ["Drift Model",
          drift_mse_test,
          drift_rmse_test,
          drift_residual_mean,
          drift_residual_variance,
          drift_residual_mean_train,
          drift_residual_variance_train,
          drift_Q_value]

base_model_results = base_model_results.append(pd.DataFrame([values], columns=base_model_columns))

# plot the predicted vs actual data
drift_df = test.copy(deep=True)
drift_df["RH"] = drift_pred_test

plot_multiline_chart_pandas_using_index([train, test, drift_df], "RH",
                                        ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
                                        "Time", "RH",
                                        "Air Quality UCI Prediction Using Drift Model",
                                        rotate_xticks=True)

# ===========================================================#
# *********** SIMPLE EXPONENTIAL SMOTHING METHOD ***********#
# ===========================================================#

print(20 * "=" + "SIMPLE EXPONENTIAL SMOOTHING" + 20 * "=")

# Compute Prediction
ses_model = SimpleExpSmoothing(np.asarray(train["RH"])).fit(smoothing_level=0.6,optimized=False)
ses_pred_test = ses_model.forecast(len(test))
ses_pred_train = ses_model.fittedvalues
# Compute Residual Error
ses_residual_test = np.subtract(test["RH"].values, np.array(ses_pred_test))
ses_residual_train = np.subtract(train["RH"].values, np.array(ses_pred_train))
# Compute Residual Variance 
ses_residual_variance = np.var(ses_residual_test)
ses_residual_variance_train = np.var(ses_residual_train)
# Compute Residual Mean
ses_residual_mean = np.mean(ses_residual_test)
ses_residual_mean_train = np.mean(ses_residual_train)
# Compute MSE
ses_mse_test = mean_squared_error( test["RH"].values, ses_pred_test)
# Compute RMSE
ses_rmse_test = np.sqrt(ses_mse_test)


# Average residual ACF
ses_residual_acf = calc_acf(ses_residual_test, len(ses_pred_test))

k = len(train) + len(test)
ses_Q_value = k * np.sum(np.array(ses_residual_acf) ** 2)

print("The MSE for SES Model is : ", round(ses_mse_test, 4))
print("The RMSE for SES Model is: ", round(ses_rmse_test, 4))
print("The Variance of residual for SES Model is: ", round(ses_residual_variance, 4))
print("The Mean of residual for SES Model is: ", round(ses_residual_mean, 4))
print('The Q value of Residual for SES Model is: ', round(ses_Q_value, 4))

print(60 * "=")

plot_acfx(ses_residual_acf, "ACF plot using SES Residuals")

# add the results to common dataframe
values = ["Simple Exponential Smoothing Model",
          ses_mse_test,
          ses_rmse_test,
          ses_residual_mean,
          ses_residual_variance,
          ses_residual_mean_train,
          ses_residual_variance_train,
          ses_Q_value]

base_model_results = base_model_results.append(pd.DataFrame([values], columns=base_model_columns))

# plot the predicted vs actual data
ses_df = test.copy(deep=True)
ses_df["RH"] = ses_pred_test

plot_multiline_chart_pandas_using_index([train, test, ses_df], "RH",
                                        ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
                                        "Time", "RH",
                                        "Air Quality UCI Prediction Using SES Model",
                                        rotate_xticks=True)

# =====================================================#
# **************** HOLT WINTER METHOD ****************#
# =====================================================#

print(20 * "=" + " HOLT WINTER METHOD " + 20 * "=")

# Compute Prediction
holtwinter_model = ExponentialSmoothing(train["RH"],  seasonal='mul').fit()
holtwinter_pred_test = holtwinter_model.forecast(len(test))
holtwinter_pred_train = holtwinter_model.fittedvalues
# Compute Residual Error
holtwinter_residual_test = np.subtract(test["RH"].values, np.array(holtwinter_pred_test))
holtwinter_residual_train = np.subtract(train["RH"].values, np.array(holtwinter_pred_train))
# Compute Residual Variance 
holtwinter_residual_variance = np.var(holtwinter_residual_test)
holtwinter_residual_variance_train = np.var(holtwinter_residual_train)
# Compute Residual Mean
holtwinter_residual_mean = np.mean(holtwinter_residual_test)
holtwinter_residual_mean_train = np.mean(holtwinter_residual_train)
# Compute MSE
holtwinter_mse_test = mean_squared_error( test["RH"].values, holtwinter_pred_test)
# Compute RMSE
holtwinter_rmse_test = np.sqrt(holtwinter_mse_test)


# Average residual ACF
holtwinter_residual_acf = calc_acf(holtwinter_residual_test, len(holtwinter_pred_test))

k = len(train) + len(test)
holtwinter_Q_value = k * np.sum(np.array(holtwinter_residual_acf) ** 2)

print("The MSE for Holt Winter Model is : ", round(holtwinter_mse_test, 4))
print("The RMSE for Holt Winter Model is: ", round(holtwinter_rmse_test, 4))
print("The Variance of residual for Holt Winter Model is: ", round(holtwinter_residual_variance, 4))
print("The Mean of residual for Holt Winter Model is: ", round(holtwinter_residual_mean, 4))
print('The Q value of Residual for Holt Winter Model is: ', round(holtwinter_Q_value, 4))

print(60 * "=")
plot_acfx(holtwinter_residual_acf, "ACF plot using Holt Winter Residuals")

# add the results to common dataframe
values = ["Holt Winter Model",
          holtwinter_mse_test,
          holtwinter_rmse_test,
          holtwinter_residual_mean,
          holtwinter_residual_variance,
          holtwinter_residual_mean_train,
          holtwinter_residual_variance_train,
          holtwinter_Q_value]

base_model_results = base_model_results.append(pd.DataFrame([values], columns=base_model_columns))

# plot the predicted vs actual data
holtwinter_df = test.copy(deep=True)
holtwinter_df["RH"] = holtwinter_pred_test

plot_multiline_chart_pandas_using_index([train, test, holtwinter_df], "RH",
                                        ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
                                        "Time", "RH",
                                        "Air Quality UCI Prediction Using Holt Winter Model",
                                        rotate_xticks=True)

print(' ' * 45, 'BASE MODEL COMPARISON')
print('=' * 110)
print(base_model_results.to_string())
print('=' * 110)

#=================================================================================================
# Question 12
# Develop the multiple linear regression model that represent the dataset.
# Check the accuracy of the developed model.
# a. You need to include the complete regression analysis into your report.
# Perform one-step ahead prediction and compare the performance versus the test set.
#=================================================================================================

#Divide the dataset in features and target
x = df[['CO(GT)','PT08.S1(CO)','NMHC(GT)','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)',
        'PT08.S3(NOx)','NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)','T','AH']]
y = df[['RH']]
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.2)

# building the model
model = sm.OLS(y_train['RH'],x_train).fit()
# predict values for x_test
ml_pred_test = model.predict(x_test)
ml_pred_train = model.predict(x_train)

print(ml_pred_test)
print(model.summary())


print(20 * "=" + " MULTIPLE LINEAR REGRESSION " + 20 * "=")
# Compute Residual Error
ml_residual_test  = np.subtract(y_test[ 'RH'].values, np.array(ml_pred_test))
ml_residual_train = np.subtract(y_train["RH"].values, np.array(ml_pred_train))
# Compute Residual Variance 
ml_residual_variance_test  = np.var(ml_residual_test)
ml_residual_variance_train = np.var(ml_residual_train)
# Compute Residual Mean
ml_residual_mean_test  = np.mean(ml_residual_test)
ml_residual_mean_train = np.mean(ml_residual_train)
# Compute MSE
ml_mse_test  = mean_squared_error( y_test[ "RH"].values, ml_pred_test)
ml_mse_train = mean_squared_error( y_train["RH"].values, ml_pred_train)
# Compute RMSE
ml_rmse_test  = np.sqrt(ml_mse_test )
ml_rmse_train = np.sqrt(ml_mse_train)



# Average residual ACF
ml_residual_test_acf  = calc_acf(ml_residual_test , len(ml_pred_test))
# ml_residual_train_acf = calc_acf(ml_residual_train, len(ml_pred_train))

ml_test_Q_value  = len(x_test)  * np.sum(np.array( ml_residual_test_acf)**2)
# ml_train_Q_value = len(x_train) * np.sum(np.array( ml_residual_train_acf)**2)


print("The MSE for Multiple Linear Regression Model is : ", round( ml_mse_test , 4) )
print("The RMSE for Multiple Linear Regression Model is: ", round( ml_rmse_test, 4) )
print("The Variance of residual for Multiple Linear Regression Model is: ", round(ml_residual_variance_test, 4) )
print("The Mean of residual for Multiple Linear Regression Model is: ", round(ml_residual_mean_test, 4) )
print('The Q value of Residual for Multiple Linear Regression  Model is: ', round ( ml_test_Q_value, 4))
print(60 * "=" )

plot_acfx(ml_residual_test_acf, "ACF plot using Multiple Linear Regression Residuals (Test)")
# plot_acfx(ml_residual_train_acf, "ACF plot using Multiple Linear Regression Residuals (Train)")

# add the results to common dataframe
values = ["Multiple Linear Regression Model", 
          ml_mse_test,
          ml_rmse_test,
          ml_residual_mean_test,
          ml_residual_variance_test,
          ml_residual_mean_train,
          ml_residual_variance_train,
          ml_test_Q_value]                                       

base_model_results = base_model_results.append( pd.DataFrame([values], columns = base_model_columns ) )

print(' '* 45, 'BASE MODEL COMPARISON' )
print('=' * 170)
print( base_model_results.to_string() ) 
print('=' * 170)

# plot result

# TRAIN, TEST AND PREDICTED

dates_train = pd.date_range(start='2004-03-10 18:00:00', end='2005-01-16 14:00:00', periods=len(y_train))
dates_test  = pd.date_range(start='2005-01-16 15:00:00', end='2005-04-04 14:00:00', periods=len( y_test))

fig, ax = plt.subplots()
plt.title('Multiple Linear Regression:Traing vs Testing vs Forecast of RH', fontsize=22)
ax.plot(dates_train, y_train['RH'],label='Training data')
ax.plot(dates_test,  y_test[ 'RH'],label='Testing data')
ax.plot(dates_test,  ml_pred_test,   label='Forecast')
plt.xlabel('Date', fontsize=15)
plt.ylabel('RH', fontsize=15)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.legend(loc='best', fontsize=20)
plt.show()

# TRAIN AND PREDICTED 

fig, ax = plt.subplots()
plt.title('Multiple Linear Regression: Traing vs Predict of RH', fontsize=22)
ax.plot(dates_train,y_train['RH'],label='Training data')
ax.plot(dates_train,ml_pred_train,label='Predicition')
plt.xlabel('Date', fontsize=15)
plt.ylabel('RH', fontsize=15)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.legend(loc='best', fontsize=20)
plt.show()

# TEST AND PREDICTED

fig, ax = plt.subplots()
plt.title('Multiple Linear Regression Model: Test vs Predict of RH', fontsize=22)
ax.plot(dates_test, y_test['RH'] ,label='Testing data')
ax.plot(dates_test, ml_pred_test ,label='Forecast')
plt.xlabel('Date', fontsize=15)
plt.ylabel('RH', fontsize=15)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.legend(loc='best', fontsize=20)
plt.show()


#=================================================================================================
# Question 13
# ARMA and ARIMA and SARIMA model order determination:
# Develop an ARMA, ARIMA and SARIMA model that represent the dataset.
# a. Preliminary model development procedures and results.
#    (ARMA model order determination).
#    Pick at least two orders using GPAC table.
# b. Should include discussion of the autocorrelation function and the GPAC.
#    Include a plot of the autocorrelation function and the GPAC table within this section).
# c. Include the GPAC table in your report and highlight the estimated order.
#=================================================================================================

# ARMA (ARIMA or SARIMA) model

j = 10
k = 10
lags = j + k

y_mean = np.mean(train['RH'])
y = np.subtract(y_mean, df['RH'])
actual_output = np.subtract(y_mean, test['RH'])

# autocorrelation of RH
ry = auto_corr_cal(y, lags)

# create GPAC Table
gpac_table = create_gpac_table(j, k, ry)
print()
print("GPAC Table:")
print(gpac_table.to_string())
print()

plot_heatmap(gpac_table, "GPAC Table for RH")


# possible orders of the process
possible_order2 = [(1,0),(2, 1)]

print()
print("The possible orders identified from GPAC for ARMA process are:")
print(possible_order2)
print()
print("None of the identified ARMA orders pass the chi-squared test.")
print()

# checking which orders pass the GPAC test
print(gpac_order_chi_square_test(possible_order2, y, '2004-03-10 18:00:00', '2005-01-16 14:00:00',                                   lags,actual_output))


possible_order = [(1, 0)]
possible_order = [(2, 1)]


# checking which orders pass the chi-square test
gpac_order_chi_square_test(possible_order, y, '2004-03-10 18:00:00', '2005-01-16 14:00:00',
                                      lags, actual_output)


#******************************************************************************
# ARMA(1,0) model   
#******************************************************************************

n_a = 1
n_b = 0

model = sm.tsa.ARMA(y, (n_a, n_b)).fit(trend='nc', disp=0)
print(model.summary())

# ARMA predictions
arma_prediction = model.predict(start="2005-01-16 15:00:00", end="2005-04-04 14:00:00")
# arma_prediction = model.forecast(len(test['RH']))[1]

arma_prediction_train = model.fittedvalues[:len(train)]
# add the subtracted mean back into the predictions
arma_prediction = np.add(y_mean, arma_prediction)
arma_prediction_train = np.add(y_mean, arma_prediction_train)


# Compute Residual Error
arma_residual_test = np.subtract(test["RH"].values, np.array(arma_prediction))
arma_residual_train = np.subtract(train["RH"].values, np.array(arma_prediction_train))
# Compute Residual Variance 
arma_residual_variance = np.var(arma_residual_test)
arma_residual_variance_train = np.var(arma_residual_train)
# Compute Residual Mean
arma_residual_mean = np.mean(arma_residual_test)
arma_residual_mean_train = np.mean(arma_residual_train)
# Compute MSE
arma_mse_test = mean_squared_error( test["RH"].values, arma_prediction)
# Compute RMSE
arma_rmse_test = np.sqrt(arma_mse_test)

print(f"The MSE for ARMA({n_a}, {n_b}) model is: " , round(arma_mse_test, 4) )
print(f"The RMSE for ARMA({n_a}, {n_b}) model is: ", round(arma_rmse_test, 4))
print(f"The Variance of residual for ARMA({n_a}, {n_b}) model is:", round(arma_residual_variance, 4))
print(f"The Mean of residual for ARMA({n_a}, {n_b}) model is:", round(arma_residual_mean, 4))


# Average residual ACF
arma_residual_acf = calc_acf(arma_residual_test, len(arma_prediction))
plot_acfx(arma_residual_acf, f"ACF plot for ARMA({n_a},{n_b}) Residuals")

k=len(train) + len(test)
arma_Q_value  = k * np.sum(np.array( arma_residual_acf)**2)


print(f"Estimated covariance matrix for n_a = {n_a} and n_b = {n_b}: \n{model.cov_params()}\n")
print(f"Estimated variance of error for n_a = {n_a} and n_b = {n_b}: \n{model.sigma2}\n")


# add the results to common dataframe
values = [f"ARMA({n_a}, {n_b}) Model", 
          arma_mse_test,
          arma_rmse_test,
          arma_residual_mean,
          arma_residual_variance,
          arma_residual_mean_train,
          arma_residual_variance_train,
          arma_Q_value]                                       

base_model_results = base_model_results.append( pd.DataFrame([values], columns = base_model_columns ) )


print(' '* 45, 'BASE MODEL COMPARISON' )
print('=' * 170)
print( base_model_results.to_string() ) 
print('=' * 170)


# plot the predicted vs actual data
arma_df = test.copy(deep=True)
arma_df["RH"] = arma_prediction

plot_multiline_chart_pandas_using_index([train, test, arma_df], "RH",
                                             ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
                                             "Time", "RH",
                                             f"Air Quality UCI Prediction Using ARMA({n_a}, {n_b}) Model",
                                             rotate_xticks=True)



# chi square
#chi_square_test(arma_Q_value, lags, n_a, n_b, alpha=0.01)


#******************************************************************************
# ARMA(2,1) model   
#******************************************************************************

n_a = 2
n_b = 1

model = sm.tsa.ARMA(y, (n_a, n_b)).fit(trend='nc', disp=0)
print(model.summary())

# ARMA predictions
arma_prediction = model.predict(start="2005-01-16 15:00:00", end="2005-04-04 14:00:00")
arma_prediction_train = model.fittedvalues[:len(train)]
# add the subtracted mean back into the predictions
arma_prediction = np.add(y_mean, arma_prediction)
arma_prediction_train = np.add(y_mean, arma_prediction_train)


# Compute Residual Error
arma_residual_test = np.subtract(test["RH"].values, np.array(arma_prediction))
arma_residual_train = np.subtract(train["RH"].values, np.array(arma_prediction_train))
# Compute Residual Variance 
arma_residual_variance = np.var(arma_residual_test)
arma_residual_variance_train = np.var(arma_residual_train)
# Compute Residual Mean
arma_residual_mean = np.mean(arma_residual_test)
arma_residual_mean_train = np.mean(arma_residual_train)
# Compute MSE
arma_mse_test = mean_squared_error( test["RH"].values, arma_prediction)
# Compute RMSE
arma_rmse_test = np.sqrt(arma_mse_test)

print(f"The MSE for ARMA({n_a}, {n_b}) model is: " , round(arma_mse_test, 4) )
print(f"The RMSE for ARMA({n_a}, {n_b}) model is: ", round(arma_rmse_test, 4))
print(f"The Variance of residual for ARMA({n_a}, {n_b}) model is:", round(arma_residual_variance, 4))
print(f"The Mean of residual for ARMA({n_a}, {n_b}) model is:", round(arma_residual_mean, 4))

# Average residual ACF
arma_residual_acf = calc_acf(arma_residual_test, len(arma_prediction))
plot_acfx(arma_residual_acf, f"ACF plot for ARMA({n_a},{n_b}) Residuals")

k=len(train) + len(test)
arma_Q_value  = k * np.sum(np.array( arma_residual_acf)**2)


print(f"Estimated covariance matrix for n_a = {n_a} and n_b = {n_b}: \n{model.cov_params()}\n")
print(f"Estimated variance of error for n_a = {n_a} and n_b = {n_b}: \n{model.sigma2}\n")


# add the results to common dataframe
values = [f"ARMA({n_a}, {n_b}) Model", 
          arma_mse_test,
          arma_rmse_test,
          arma_residual_mean,
          arma_residual_variance,
          arma_residual_mean_train,
          arma_residual_variance_train,
          arma_Q_value]                                       

base_model_results = base_model_results.append( pd.DataFrame([values], columns = base_model_columns ) )




print(' '* 45, 'BASE MODEL COMPARISON' )
print('=' * 170)
print( base_model_results.to_string() ) 
print('=' * 170)


# plot the predicted vs actual data
arma_df = test.copy(deep=True)
arma_df["RH"] = arma_prediction

plot_multiline_chart_pandas_using_index([train, test, arma_df], "RH",
                                             ["Train", "Test", "Prediction"], ["Blue", "Orange", "Green"],
                                             "Time", "RH",
                                             f"Air Quality UCI Prediction Using ARMA({n_a}, {n_b}) Model",
                                             rotate_xticks=True)

# chi square
#chi_square_test(arma_Q_value, lags, n_a, n_b, alpha=0.01)


#=================================================================================================
# Question 14
# Estimate ARMA model parameters using the Levenberg Marquardt algorithm. 
# Display the parameter estimates, the standard deviation of the parameter 
# estimates and confidence intervals.
#=================================================================================================

#####################
#### ARMA (1,0) #####
#####################

na, nb = (1, 0)

# ================================================
# LM ALGORITM
# ================================================
lm_params, ro2, cov_theta, iterations = LM_algoritmh(y=train['RH'],
                                                     n_a=na,
                                                     n_b=nb,
                                                     num_iter=30,
                                                     delta=1e-6,
                                                     flip_val=True)

# parameter estimated
print(' ' * 27, ' PARAMETER ESTIMATED ')
print('=' * 80)

lm_den = [1.] + [lm_params[i] for i in range(na)]
lm_num = [1.] + [lm_params[i + na] for i in range(nb)]
for i in range(na):
    print('LM - The AR coefficient a{}'.format(i), 'is:', lm_params[i])
for i in range(nb):
    print('LM - The MA coefficient b{}'.format(i), 'is:', lm_params[i + na])

# ARMA MODEL to compare
model = sm.tsa.ARMA(train["RH"], (na, nb)).fit(trend='nc', disp=0)  # hacer del train

for i in range(na):
    print('The AR coefficient a{}'.format(i), 'is:', model.params[i])
for i in range(nb):
    print('The MA coefficient b{}'.format(i), 'is:', model.params[i + na])

# Estimated covariance matrix of the estimated parameters.
print(' ' * 25, 'Estimated Covariance Matrix')
print(' ' * 30, 'LM algorithm')
print('=' * 80)
print(np.matrix(cov_theta))
print('=' * 80)
# Estimated variance of error.
print(' ' * 23, 'Estimated Variance of Error')
print(' ' * 30, 'LM algorithm')
print('=' * 80)
print(np.matrix(ro2))
print('=' * 80)
# Confidence interval for each estimated parameter(s).
# θi ± 2*sqrt(cov(θ)ii)
print(' ' * 23, 'Confidence Interval for Theta')
print('=' * 80)
print('   θi-2*sqrt(cov(θ)ii)   |            θi            |   θi ± 2*sqrt(cov(θ)ii)')
print('=' * 80)
for i in range(len(lm_params)):
    sqrt_ro = np.sqrt(cov_theta[i, i])
    theta_i = lm_params[i]
    theta_lower = theta_i - 2 * sqrt_ro
    theta_upper = theta_i + 2 * sqrt_ro
    print('  ', str(theta_lower).ljust(26),
          str(theta_i).ljust(26),
          str(theta_upper))

print('=' * 80)

#=================================================================================================
# Question 15
# Diagnostic Analysis: Make sure to include the followings:
# a. Diagnostic tests (confidence intervals, zero/pole cancellation, chi-square test).
# b. Display the estimated variance of the error and the estimated covariance of the estimated parameters.
# c. Is the derived model biased or this is an unbiased estimator?
# d. Check the variance of the residual errors versus the variance of the forecast errors.
# e. If you find out that the ARIMA or SARIMA model may better represents the dataset, then you can find the model accordingly. You are not constraint only to use of ARMA model. Finding an ARMA model is a minimum requirement and making the model better is always welcomed.
#=================================================================================================

# Confidence interval for each estimated parameter(s).
# θi ± 2*sqrt(cov(θ)ii)
print(' '*23,'Confidence Interval for Theta')
print('=' * 80)
print('   θi-2*sqrt(cov(θ)ii)   |            θi            |   θi ± 2*sqrt(cov(θ)ii)')
print('=' * 80)
for i in range( len(lm_params) ):
    sqrt_ro = np.sqrt(cov_theta[i,i])
    theta_i = lm_params[i]
    theta_lower = theta_i - 2 * sqrt_ro
    theta_upper = theta_i + 2 * sqrt_ro
    print('  ', str(theta_lower).ljust(26),
          str(theta_i).ljust(26),
          str(theta_upper) )
print('=' * 80)

# Zero/Pole Cancelation
#******************************************************
zeros, poles, _ = zero_pole_plot(lm_num, lm_den)

print(' '*23,'Zero / Pole Cancelation')
print('=' * 80)
print('Root of numerator "Zeros" : ', zeros)
print('Root of denominator "Poles" : ', poles)

# Estimated covariance matrix of the estimated parameters.
print(' '*25,'Estimated Covariance Matrix')
print(' '*30, 'LM algorithm' )
print('=' * 80)
print(np.matrix( cov_theta ) )
print('=' * 80)
# Estimated variance of error.
print(' '*23,'Estimated Variance of Error')
print(' '*30, 'LM algorithm' )
print('=' * 80)
print(np.matrix( ro2 ) )
print('=' * 80)

#####################
#### ARMA (2,1) #####
#####################

na, nb = (2,1)

#================================================
# LM ALGORITM 
#================================================
lm_params, ro2, cov_theta, iterations = LM_algoritmh(y = train['RH'],
                                                     n_a = na,
                                                     n_b = nb,
                                                     num_iter = 30,
                                                     delta = 1e-6,
                                                     flip_val=True)



# parameter estimated
print(' '*27,' PARAMETER ESTIMATED ')
print('=' * 80)

lm_den = [1.] + [ lm_params[i] for i in range(na)]
lm_num = [1.] + [ lm_params[i+na] for i in range(nb)]
for i in range(na):
    print('LM - The AR coefficient a{}'.format(i), 'is:', lm_params[i])
for i in range(nb):
    print('LM - The MA coefficient b{}'.format(i), 'is:', lm_params[i+na])


# ARMA MODEL to compare
model = sm.tsa.ARMA(train["RH"],(na,nb)).fit(trend='nc',disp=0) # hacer del train

for i in range(na):
    print('The AR coefficient a{}'.format(i), 'is:', model.params[i])
for i in range(nb):
    print('The MA coefficient b{}'.format(i), 'is:', model.params[i+na])


# Estimated covariance matrix of the estimated parameters.
print(' '*25,'Estimated Covariance Matrix')
print(' '*30, 'LM algorithm' )
print('=' * 80)
print(np.matrix( cov_theta ) )
print('=' * 80)
# Estimated variance of error.
print(' '*23,'Estimated Variance of Error')
print(' '*30, 'LM algorithm' )
print('=' * 80)
print(np.matrix( ro2 ) )
print('=' * 80)
# Confidence interval for each estimated parameter(s). 
# θi ± 2*sqrt(cov(θ)ii)
print(' '*23,'Confidence Interval for Theta')
print('=' * 80)
print('   θi-2*sqrt(cov(θ)ii)   |            θi            |   θi ± 2*sqrt(cov(θ)ii)')
print('=' * 80)
for i in range( len(lm_params) ):
    sqrt_ro = np.sqrt(cov_theta[i,i])
    theta_i = lm_params[i] 
    theta_lower = theta_i - 2 * sqrt_ro
    theta_upper = theta_i + 2 * sqrt_ro
    print('  ', str(theta_lower).ljust(26), 
          str(theta_i).ljust(26),
          str(theta_upper) )
    
print('=' * 80)

#=================================================================================================
# Question 15
# Diagnostic Analysis: Make sure to include the followings:
# a. Diagnostic tests (confidence intervals, zero/pole cancellation, chi-square test).
# b. Display the estimated variance of the error and the estimated covariance of the estimated parameters.
# c. Is the derived model biased or this is an unbiased estimator?
# d. Check the variance of the residual errors versus the variance of the forecast errors.
# e. If you find out that the ARIMA or SARIMA model may better represents the dataset, then you can find the model accordingly. You are not constraint only to use of ARMA model. Finding an ARMA model is a minimum requirement and making the model better is always welcomed.
#=================================================================================================

# Confidence interval for each estimated parameter(s). 
# θi ± 2*sqrt(cov(θ)ii)
print(' '*23,'Confidence Interval for Theta')
print('=' * 80)
print('   θi-2*sqrt(cov(θ)ii)   |            θi            |   θi ± 2*sqrt(cov(θ)ii)')
print('=' * 80)
for i in range( len(lm_params) ):
    sqrt_ro = np.sqrt(cov_theta[i,i])
    theta_i = lm_params[i] 
    theta_lower = theta_i - 2 * sqrt_ro
    theta_upper = theta_i + 2 * sqrt_ro
    print('  ', str(theta_lower).ljust(26), 
          str(theta_i).ljust(26),
          str(theta_upper) )
print('=' * 80)

# Zero/Pole Cancelation
#******************************************************
zeros, poles, _ = zero_pole_plot(np.array(lm_num) , np.array(lm_den) )

print(' '*23,'Zero / Pole Cancelation')
print('=' * 80)
print('Root of numerator "Zeros" : ', zeros)
print('Root of denominator "Poles" : ', poles)

# Estimated covariance matrix of the estimated parameters.
print(' '*25,'Estimated Covariance Matrix')
print(' '*30, 'LM algorithm' )
print('=' * 80)
print(np.matrix( cov_theta ) )
print('=' * 80)
# Estimated variance of error.
print(' '*23,'Estimated Variance of Error')
print(' '*30, 'LM algorithm' )
print('=' * 80)
print(np.matrix( ro2 ) )
print('=' * 80)

#=================================================================================================
# Question 18
# Forecast function: Once the final mode is picked (SARIMA), the forecast 
# function needs to be developed and included in your report.
#=================================================================================================

#******************************************************************************
# SARIMA MODEL (0,0,0)  (n_a, d , nb)
#******************************************************************************

order_x = (0,0,0)
seasonal_order_x = (0,1,1,24)

sarima= sm.tsa.statespace.SARIMAX(train["RH"], 
                                  order = order_x, seasonal_order = seasonal_order_x ).fit()

print(sarima.summary())

sarima_pred_test = sarima.forecast(len(test['RH']))
sarima_pred_train = sarima.fittedvalues
# Compute Residual Error
sarima_residual_test = np.subtract(test["RH"].values, np.array(sarima_pred_test))
sarima_residual_train = np.subtract(train["RH"].values, np.array(sarima_pred_train))
# Compute Residual Variance 
sarima_residual_variance = np.var(sarima_residual_test)
sarima_residual_variance_train = np.var(sarima_residual_train)
# Compute Residual Mean
sarima_residual_mean=np.mean(sarima_residual_test)
sarima_residual_mean_train = np.mean(sarima_residual_train)
# Compute MSE
sarima_mse_test = mean_squared_error( test["RH"].values, sarima_pred_test)
# Compute RMSE
sarima_rmse_test = np.sqrt(sarima_mse_test)



# Average residual ACF
sarima_residual_acf = calc_acf(sarima_residual_test, len(sarima_pred_test))


k = len(train) + len(test)
sarima_Q_value = k * np.sum(np.array( sarima_residual_acf)**2)


print("The MSE for SARIMA Model is : ", round( sarima_mse_test , 4) )
print("The RMSE for SARIMA Model is: ", round( sarima_rmse_test, 4) )
print("The Variance of residual for SARIMA Model is: ", round(sarima_residual_variance, 4) )
print("The Mean of residual for SARIMA Model is: ", round(sarima_residual_mean, 4) )
print("The Variance of residual for SARIMA Model (Train) is: ", round(sarima_residual_variance_train, 4) )
print("The Mean of residual for SARIMA Model (Train) is: ", round(sarima_residual_mean_train, 4) )
print('The Q value of Residual for SARIMA Model is: ', round ( sarima_Q_value, 4))


print(60 * "=" )
plot_acfx(sarima_residual_acf, "ACF plot using SARIMA Residuals")

# add the results to common dataframe
values = [f"SARIMA {order_x} {seasonal_order_x} Model", 
          sarima_mse_test,
          sarima_rmse_test,
          sarima_residual_mean,
          sarima_residual_variance,
          sarima_residual_mean_train,
          sarima_residual_variance_train,
          sarima_Q_value]                                       

base_model_results = base_model_results.append( pd.DataFrame([values], columns = base_model_columns ) )


dates_train = pd.date_range(start='2004-03-10 18:00:00', end='2005-01-16 14:00:00', periods=len(y_train))
dates_test  = pd.date_range(start='2005-01-16 15:00:00', end='2005-04-04 14:00:00', periods=len( y_test))

# TEST AND PREDICTED
start = 1000
range_x = range(start, start + 400)
fig, ax = plt.subplots()
plt.title('SARIMA: Train vs Fitted Values of RH', fontsize=22)
ax.plot(dates_train[range_x], y_train['RH'].values[range_x] ,label='Training data')
ax.plot(dates_train[range_x], sarima_pred_train.values[range_x] ,label='Fitted Values')
plt.xlabel('Date', fontsize=15)
plt.ylabel('RH', fontsize=15)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.legend(loc='best', fontsize=20)
plt.show()



# TEST AND PREDICTED
start = 0
range_x = range(start, start + 400)
fig, ax = plt.subplots()
plt.title('SARIMA: Test vs Predict of RH', fontsize=22)
ax.plot(dates_test[range_x], y_test['RH'].values[range_x] ,label='Testing data')
ax.plot(dates_test[range_x], sarima_pred_test.values[range_x] ,label='Forecast')
plt.xlabel('Date', fontsize=15)
plt.ylabel('RH', fontsize=15)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.legend(loc='best', fontsize=20)
plt.show()



print(' '* 80, 'BASE MODEL COMPARISON' )
print('=' * 170)
print( base_model_results.to_string() ) 
print('=' * 170)

#=================================================================================================
# Question 19
# h-step ahead Predictions: You need to make a multiple step ahead prediction 
# for the duration of the test data set. Then plot the predicted values 
# versus the true value (test set) and write down your observations.
#=================================================================================================

# h-step ahead prediction - using Multiple Linear Regression 


# *******************************************************************************
# MULTIPLE LINEAR REGRESSION
# *******************************************************************************
# TRAIN, TEST AND PREDICTED

dates_train = pd.date_range(start='2004-03-10 18:00:00', end='2005-01-16 14:00:00', periods=len(y_train))
dates_test  = pd.date_range(start='2005-01-16 15:00:00', end='2005-04-04 14:00:00', periods=len( y_test))


# TRAIN AND PREDICTED 
h_ahead = 1700
range_x = range( h_ahead)


# TEST AND PREDICTED

fig, ax = plt.subplots()
plt.title('Multiple Linear Regression Model: Test vs Predict of RH', fontsize=22)
ax.plot( dates_test[range_x] ,y_test['RH'].values[range_x],label='Testing data')
ax.plot( dates_test[range_x], ml_pred_test.values[range_x],label='Forecast')
plt.xlabel('Date', fontsize=15)
plt.ylabel('RH', fontsize=15)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.legend(loc='best', fontsize=20)
plt.show()

#***********************************************************
# SARIMA OBSERVATIONS
#***********************************************************

dates_train = pd.date_range(start='2004-03-10 18:00:00', end='2005-01-16 14:00:00', periods=len(y_train))
dates_test  = pd.date_range(start='2005-01-16 15:00:00', end='2005-04-04 14:00:00', periods=len( y_test))

# TEST AND PREDICTED
h_ahead = 100

range_x = range(0, h_ahead)
fig, ax = plt.subplots()
plt.title('SARIMA: Train vs Fitted Values of RH', fontsize=22)
ax.plot(dates_train[range_x], y_train['RH'].values[range_x] ,label='Training data')
ax.plot(dates_train[range_x], sarima_pred_train.values[range_x] ,label='Fitted Values')
plt.xlabel('Date', fontsize=15)
plt.ylabel('RH', fontsize=15)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.legend(loc='best', fontsize=20)
plt.show()



# TEST AND PREDICTED
fig, ax = plt.subplots()
plt.title('SARIMA: Test vs Predict of RH', fontsize=22)
ax.plot(dates_test[range_x], y_test['RH'].values[range_x] ,label='Testing data')
ax.plot(dates_test[range_x], sarima_pred_test.values[range_x] ,label='Forecast')
plt.xlabel('Date', fontsize=15)
plt.ylabel('RH', fontsize=15)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.legend(loc='best', fontsize=20)
plt.show()


