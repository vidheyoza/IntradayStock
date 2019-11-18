import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_maker import data_to_indicators, normalize, normalize_test

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error


def error(y_pred, y_true):
    """
    Give an array of squared errors

    :param y_pred: Predicted Y
    :param y_true: True Y
    :return: Data-point wise squared error
    """
    return (y_pred - y_true) ** 2


def plot_error(model_name, y_pred, y_true):
    """
    Plot squared errors for each test data point

    :param model_name: Name of model to be evaluated
    :param y_pred: Predicted Y
    :param y_true: True Y
    """
    e = error(y_pred, y_true)
    plt.plot(e)
    plt.title(model_name + ' Evaluation')
    plt.show()


# ===============
# DATA EXTRACTION
# ===============

data_path = '/Users/vidhey/coding/Python/IntradayStock/data/'
stock = 'AAPL'

data = pd.read_csv(data_path + stock + '_180101_191111.csv')
data.drop(['<TICKER>', '<PER>', '<DATE>', '<TIME>'], axis=1, inplace=True)

data = data.head(1000)

# TODO Adjusted Close?

c = 'close'
o = 'open'
h = 'high'
l = 'low'
v = 'vol'

data = pd.DataFrame({
    c: data['<CLOSE>'],
    o: data['<OPEN>'],
    l: data['<LOW>'],
    h: data['<HIGH>'],
    v: data['<VOL>']
})

data = data_to_indicators(data, o, h, l, c, v, window=10)
data = data[100:-100]

# ===============
# DATA EXTRACTION
# ===============

X = data.iloc[:, 5:].values
y = data.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

X_train, X_mean, X_std = normalize(X_train)
X_test = normalize_test(X_test, X_mean, X_std)

y_train, y_mean, y_std = normalize(y_train)
y_test = normalize_test(y_test, y_mean, y_std)

# =========================
# PRELIMINARY DATA ANALYSIS
# =========================


# ==============
# MODEL CREATION
# ==============

svr_model = SVR()
rf_model = RandomForestRegressor()
xgb_model = XGBRegressor()

svr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# TODO ANN MODEL

# ================
# MODEL EVALUATION
# ================

# SUPPORT VECTOR REGRESSOR
y_pred_svr = svr_model.predict(X_test)
svr_mse = mean_squared_error(y_pred_svr, y_test)
plot_error('SVR', y_pred_svr, y_test)

# RANDOM FOREST
y_pred_rf = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_pred_rf, y_test)
plot_error('Random Forest', y_pred_rf, y_test)

# XGBOOST
y_pred_xgb = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_pred_xgb, y_test)
plot_error('XGBoost', y_pred_xgb, y_test)
