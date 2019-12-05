import matplotlib.pyplot as plt
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
    :param path: Path of plot to be saved
    """
    e = error(y_pred, y_true)
    plt.plot(e)
    plt.title(model_name + ' Evaluation')
    plt.xlabel('MSE Value')
    plt.ylabel('Test data points')
    plt.savefig(path + model_name + '_errors')
    plt.show()


def plot_results(model_name, y_true, y_pred, path):
    """
    Plot errors and prices predicted by the given model

    :param model_name: String name of model
    :param y_true: True values of y
    :param y_pred: Predicted values of y
    :param path: Path of plots to be saved
    """
    # y_pred = model.predict(X_test)
    model_mse = mean_squared_error(y_pred, y_true)
    plot_error(model_name, y_pred, y_true, path=path + 'plots/')

    plt.plot(y_true, c='black', label='Real Price')
    plt.plot(y_pred, c='blue', label='Predicted Price')
    plt.xlabel('Price ($)')
    plt.ylabel('Test data points')
    plt.legend()

    plt.savefig(path + 'plots/' + model_name + '_pred')
    plt.show()
