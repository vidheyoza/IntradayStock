import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
from tqdm import tqdm
from keras import backend as K
import numpy as np


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
    :param path: Path of plot to be saved
    """
    e = error(y_pred, y_true)
    plt.plot(e)
    plt.title(model_name + ' Evaluation')
    plt.xlabel('MSE Value')
    plt.ylabel('Test data points')
    plt.savefig(path + model_name + '_errors')
    plt.show()


def print_results(model_name, stock_name, data_interval, y_true, y_pred, path):
    """
    Plot errors and prices predicted by the given model

    :param model_name: String name of model
    :param stock_name: String name of stock used for training model
    :param data_interval: Interval of data min minutes (only for specifying path)
    :param y_true: True values of y
    :param y_pred: Predicted values of y
    :param path: Path of plots to be saved
    """
    # y_pred = model.predict(X_test)
    model_mse = mean_squared_error(y_true, y_pred)
    model_rmse = rmse(y_true, y_pred)
    model_mape = mape(y_true, y_pred)

    print(model_name + ' Results: ')
    print('RMSE:', model_rmse)
    print('MSE:', model_mse)
    print('MAPE:', model_mape)

    # plot_error(model_name, y_pred, y_true, path=path + 'plots/'+str(data_interval)+'min/')

    plt.plot(y_true, c='black', label='Real Price')
    plt.plot(y_pred, c='blue', label='Predicted Price')
    plt.xlabel('Price ($)')
    plt.ylabel('Test data points')
    plt.legend()
    plt.title(model_name + 'Results')
    plt.savefig(path + 'plots/' + str(data_interval) + 'min/' + stock_name + '_' + model_name + '_pred')
    plt.show()


def increase_data_interval(data, new_interval: int = 5):
    """
    Returns data in new interval after adjusting values of columns

    :param data: Data containing C-OHLV values only (in 1 min interval)
    :param new_interval: new interval of data in minutes (default = 5 mins)
    :return: Data with new interval and adjusted values
    """
    ret = pd.DataFrame(columns=['close', 'open', 'high', 'low', 'vol'])

    for i in tqdm(range(int(data.shape[0] / new_interval))):
        window = data.iloc[i * new_interval:(i + 1) * new_interval].values
        ret = ret.append(
            {'close': window[4, 0],
             'open': window[0, 3],
             'high': max(window[:, 1]),
             'low': min(window[:, 2]),
             'vol': sum(window[:, 4])},
            ignore_index=True)

    return ret


def K_rmse(y_true, y_pred):
    """
    Custom RMSE function for Keras model

    :param y_true: True Y
    :param y_pred: Predicted Y
    :return: RMSE metric for given predictions
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def K_mape(y_true, y_pred):
    """
    Custom MAPE function for Keras model

    :param y_true: True Y
    :param y_pred: Predicted Y
    :return: MAPE metric for given predictions
    """
    return K.mean(K.abs((y_true - y_pred) / y_true), axis=-1) * 100


def rmse(y_true, y_pred):
    """
    Custom RMSE function

    :param y_true: True Y
    :param y_pred: Predicted Y
    :return: RMSE metric for given predictions
    """
    return np.sqrt(((y_pred - y_true) ** 2).mean())


def mape(y_true, y_pred):
    """
    Custom MAPE function

    :param y_true: True Y
    :param y_pred: Predicted Y
    :return: MAPE metric for given predictions
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
