import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.externals import joblib

from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import mape, mae

from data_maker import data_to_indicators, normalize, normalize_test, inverse_normalize
from helper import print_results, increase_data_interval, K_rmse

# ===============
# DATA EXTRACTION
# ===============

path = '/Users/vidhey/coding/Python/IntradayStock/'
# stock = 'AAPL'
data_interval = 10
for stock in ['AAPL', 'AMZN', 'MSFT', 'TSLA', 'WMT']:
    data = pd.read_csv(path + 'data/' + stock + '_180101_191111.csv')
    data.drop(['<TICKER>', '<PER>', '<DATE>', '<TIME>'], axis=1, inplace=True)

    data = data.head(100000)

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

    data = increase_data_interval(data, new_interval=data_interval)

    data = data_to_indicators(data, o, h, l, c, v, window=10)
    data = data[100:-100]

    # =========================
    # PRELIMINARY DATA ANALYSIS
    # =========================

    minutes = int(390 / data_interval)
    # day = 1
    for day in [0, 1]:
        plt.plot(data[c].values[day * minutes:(day + 1) * minutes])
        plt.title('Minute-wise closing price of ' + stock + ' for 2018-01-0' + str(day + 2))
        plt.savefig(path + 'plots/' + str(data_interval) + 'min/' + stock + '_2018010' + str(day + 2))
        plt.show()

        plt.plot(data[v].values[day * minutes:(day + 1) * minutes])
        plt.title('Minute-wise traded volume of ' + stock + ' for 2018-01-0' + str(day + 2))
        plt.savefig(path + 'plots/' + str(data_interval) + 'min/' + stock + '_vol_2018010' + str(day + 2))
        plt.show()

        plt.plot(data['macd_10'].values[day * minutes:(day + 1) * minutes])
        plt.plot(np.zeros(data['macd_10'].values[day * minutes:(day + 1) * minutes].shape), linestyle=':', c='black')
        plt.title('Minute-wise MACD of ' + stock + ' for 2018-01-0' + str(day + 2))
        plt.savefig(path + 'plots/' + str(data_interval) + 'min/' + stock + '_macd_2018010' + str(day + 2))
        plt.show()

    # ===============
    # DATA EXTRACTION
    # ===============

    all_features = [
        'close', 'open', 'high', 'low', 'vol',
        'macd_10', 'rsi_10', 'wr_10', 'mfi_10', 'stochk_10', 'stochd_10', 'roc_10',
        'sma_10', 'wma_10', 'ema_10', 'hma_10',
        'cci_10', 'adl', 'cmf_10', 'obv', 'emv_10',
        'atr_10', 'mass_ind_10', 'ichimoku_a', 'ichimoku_b', 'aroon_ind_10', 'adx_10'
    ]

    selected_features = [
        'close', 'open', 'high', 'low', 'vol',
        'macd_10', 'rsi_10', 'wr_10', 'mfi_10', 'stochk_10', 'stochd_10', 'roc_10',
        'sma_10', 'wma_10', 'cci_10', 'atr_10', 'mass_ind_10',
        'ichimoku_a', 'ichimoku_b', 'aroon_ind_10', 'adx_10'
    ]

    data = data[selected_features]

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
    rf_model = RandomForestRegressor(n_estimators=100)
    adb_model = AdaBoostRegressor(n_estimators=100)
    xgb_model = XGBRegressor()

    svr_model.fit(X_train, y_train)
    joblib.dump(svr_model, path + 'models/' + str(data_interval) + 'min/svr_' + stock + '.pkl')
    # svr_model = joblib.load(path+'models/'+str(data_interval)+'min/svr_'+stock+'.pkl')

    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, path + 'models/' + str(data_interval) + 'min/rf_' + stock + '.pkl')
    # rf_model = joblib.load(path+'models/'+str(data_interval)+'min/rf_'+stock+'.pkl')

    adb_model.fit(X_train, y_train)
    joblib.dump(adb_model, path + 'models/' + str(data_interval) + 'min/adb_' + stock + '.pkl')
    # adb_model = joblib.load(path+'models/'+str(data_interval)+'min/adb_'+stock+'.pkl')

    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, path + 'models/' + str(data_interval) + 'min/xgb_' + stock + '.pkl')
    # xgb_model = joblib.load(path+'models/'+str(data_interval)+'min/xgb_'+stock+'.pkl')

    ann_model = Sequential()
    ann_model.add(Dense(100, activation='relu', input_dim=X_train.shape[1]))
    ann_model.add(Dense(50, activation='relu'))
    ann_model.add(Dense(1, activation='linear'))

    ann_model.compile(optimizer='adam', loss='mse', metrics=[mape, mae, K_rmse])

    ann_model.fit(X_train, y_train,
                  batch_size=8,
                  epochs=50,
                  # validation_data=(X_test, y_test),
                  verbose=0)
    ann_model.save_weights(path + 'models/' + str(data_interval) + 'min/ann_' + stock + '.h5')
    # ann_model.load_weights(path+'models/'+str(data_interval)+'min/ann_'+stock+'.h5')

    # ================
    # MODEL EVALUATION
    # ================

    print('Stock name: ', stock)

    # SUPPORT VECTOR REGRESSOR
    y_pred_svr = inverse_normalize(data=svr_model.predict(X_test), m=y_mean, s=y_std)
    print_results('SVR', stock, data_interval, y_test, y_pred_svr, path=path)

    # RANDOM FOREST
    y_pred_rf = inverse_normalize(data=rf_model.predict(X_test), m=y_mean, s=y_std)
    print_results('RandomForest', stock, data_interval, y_test, y_pred_rf, path=path)

    # ADABOOST
    y_pred_adb = inverse_normalize(data=adb_model.predict(X_test), m=y_mean, s=y_std)
    print_results('AdaBoost', stock, data_interval, y_test, y_pred_adb, path=path)

    # XGBOOST
    y_pred_xgb = inverse_normalize(data=xgb_model.predict(X_test), m=y_mean, s=y_std)
    print_results('XGBoost', stock, data_interval, y_test, y_pred_xgb, path=path)

    # NEURAL NETWORK
    y_pred_ann = inverse_normalize(data=ann_model.predict(X_test), m=y_mean, s=y_std)
    print_results('ANN', stock, data_interval, y_test, y_pred_ann, path=path)
