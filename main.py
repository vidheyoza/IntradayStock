import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_maker import data_to_indicators, normalize, normalize_test, inverse_normalize
from helper import plot_results

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

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

# =========================
# PRELIMINARY DATA ANALYSIS
# =========================

minutes = 390
day = 1
plt.plot(data[c].values[day * minutes:(day + 1) * minutes])
plt.title('Minute-wise closing price of ' + stock + ' for 2018-01-03')
plt.savefig(path + 'plots/' + stock + '_20180103')
plt.show()

plt.plot(data[v].values[day * minutes:(day + 1) * minutes])
plt.title('Minute-wise traded volume of ' + stock + ' for 2018-01-03')
plt.savefig(path + 'plots/' + stock + '_vol_20180103')
plt.show()

plt.plot(data['macd_10'].values[day * minutes:(day + 1) * minutes])
plt.plot(np.zeros(data['macd_10'].values[day * minutes:(day + 1) * minutes].shape), linestyle=':', c='black')
plt.title('Minute-wise MACD of ' + stock + ' for 2018-01-03')
plt.savefig(path + 'plots/' + stock + '_macd_20180103')
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
# y_test = normalize_test(y_test, y_mean, y_std)

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
y_pred_svr = inverse_normalize(data=svr_model.predict(X_test), m=y_mean, s=y_std)
plot_results('XGBoost', y_test, y_pred_svr, path=path)

# RANDOM FOREST
y_pred_rf = inverse_normalize(data=rf_model.predict(X_test), m=y_mean, s=y_std)
plot_results('XGBoost', y_test, y_pred_rf, path=path)

# XGBOOST
y_pred_xgb = inverse_normalize(data=xgb_model.predict(X_test), m=y_mean, s=y_std)
plot_results('XGBoost', y_test, y_pred_xgb, path=path)
