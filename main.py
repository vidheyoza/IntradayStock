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


def plot_error(model_name, y_pred, y_true, path):
    """
    Plot squared errors for each test data point

    :param model_name: Name of model to be evaluated
    :param y_pred: Predicted Y
    :param y_true: True Y
    """
    e = error(y_pred, y_true)
    plt.plot(e)
    plt.title(model_name + ' Evaluation')
    plt.savefig(path + model_name + '_errors')
    plt.show()


# ===============
# DATA EXTRACTION
# ===============

path = '/Users/vidhey/coding/Python/IntradayStock/'
stock = 'AAPL'

data = pd.read_csv(path + 'data/' + stock + '_180101_191111.csv')
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

minutes = 390
day = 1
plt.plot(data[c].values[day * minutes:(day + 1) * minutes])
plt.title('Minute-wise closing price of AAPL for 2018-01-03')
plt.savefig(path + 'plots/AAPL_20180103')
plt.show()

plt.plot(data[v].values[day * minutes:(day + 1) * minutes])
plt.title('Minute-wise traded volume of AAPL for 2018-01-03')
plt.savefig(path + 'plots/AAPL_vol_20180103')
plt.show()

plt.plot(data['macd_10'].values[day * minutes:(day + 1) * minutes])
plt.plot(np.zeros(data['macd_10'].values[day * minutes:(day + 1) * minutes].shape), linestyle=':', c='black')
plt.title('Minute-wise MACD of AAPL for 2018-01-03')
plt.savefig(path + 'plots/AAPL_macd_20180103')
plt.show()

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
plot_error('SVR', y_pred_svr, y_test, path=path + 'plots/')

plt.plot(y_test, c='black', label='Real Price')
plt.plot(y_pred_svr, c='blue', label='Predicted Price')
plt.legend()
plt.savefig(path + 'plots/SVR_pred')
plt.show()

# RANDOM FOREST
y_pred_rf = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_pred_rf, y_test)
plot_error('RandomForest', y_pred_rf, y_test, path=path + 'plots/')

plt.plot(y_test, c='black', label='Real Price')
plt.plot(y_pred_rf, c='blue', label='Predicted Price')
plt.legend()
plt.savefig(path + 'plots/RF_pred')
plt.show()

# XGBOOST
y_pred_xgb = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_pred_xgb, y_test)
plot_error('XGBoost', y_pred_xgb, y_test, path=path + 'plots/')

plt.plot(y_test, c='black', label='Real Price')
plt.plot(y_pred_xgb, c='blue', label='Predicted Price')
plt.legend()
plt.savefig(path + 'plots/XGB_pred')
plt.show()
