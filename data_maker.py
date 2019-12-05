from ta import *
from pyti import simple_moving_average, weighted_moving_average, exponential_moving_average, hull_moving_average


def data_to_indicators(data, o, h, l, c, v, window) -> pd.DataFrame:
    """
    Given a DataFrame of stock prices, return the data along with the corresponding indicators

    :param data: Data in C-OHLV format
    :param window: default value of window for indicators (only for indicators that support this)
    :param o: name of column containing 'OPEN' values
    :param h: name of column containing 'HIGH' values
    :param l: name of column containing 'LOW' values
    :param c: name of column containing 'CLOSE' values
    :param v: name of column containing 'VOLUME' values
    :return: DataFrame with all indicators
    """
    df = data.reset_index(drop=True)

    # Momentum
    # TODO PMO
    df['macd_' + str(window)] = macd(df[c], n_fast=int(window / 2), n_slow=window)
    df['rsi_' + str(window)] = rsi(df[c], n=window)
    df['wr_' + str(window)] = wr(df[h], df[l], df[c], lbp=window)
    df['mfi_' + str(window)] = money_flow_index(df[h], df[l], df[c], df[v], n=window)

    # ROC
    df['roc_' + str(window)] = [(df[c][i] - df[c][i - window]) / df[c][i - window] if i >= window else np.nan
                                for i in range(0, df[c].size)]

    # Smoothing
    df['sma_' + str(window)] = simple_moving_average.simple_moving_average(df[c], period=window)
    df['wma_' + str(window)] = weighted_moving_average.weighted_moving_average(df[c], period=window)
    df['ema_' + str(window)] = exponential_moving_average.exponential_moving_average(np.array(df[c]), period=window)
    df['hma_' + str(window)] = hull_moving_average.hull_moving_average(np.array(df[c]), period=window)
    # df['cma_' + str(window)] = [df[c][i - int(window / 2): i + int(window / 2)].mean()
    #                             if (i >= int(window / 2) and (i + int(window / 2)) < df[c].size) else np.nan
    #                             for i in range(0, df[c].size)]

    # Overbought/Oversold Signals
    df['cci_' + str(window)] = cci(df[h], df[l], df[c], n=window, c=0.015)

    # Volume
    df['adl'] = acc_dist_index(df[h], df[l], df[c], df[v])
    df['cmf_' + str(window)] = chaikin_money_flow(df[h], df[l], df[c], df[v], n=window)
    df['obv'] = on_balance_volume(df[c], df[v])
    df['emv_' + str(window)] = ease_of_movement(df[h], df[l], df[c], df[v], n=window)

    # Volatility
    df['atr_' + str(window)] = average_true_range(df[h], df[l], df[c], n=window)
    df['mass_ind_' + str(window)] = mass_index(df[h], df[l], n=window / 2, n2=window)

    df['bollinger_hband'] = bollinger_hband_indicator(df[c], n=20, ndev=2)
    df['bollinger_lband'] = bollinger_lband_indicator(df[c], n=20, ndev=2)

    # Trends
    # How will model know b/w Ichmoku A and B?
    df['ichimoku_a'] = ichimoku_a(df[h], df[l], n1=9, n2=26)
    df['ichimoku_b'] = ichimoku_b(df[h], df[l], n2=26, n3=52)

    series_aroon_up = aroon_up(df[c], n=window)
    series_aroon_down = aroon_down(df[c], n=window)
    df['aroon_ind_' + str(window)] = series_aroon_up - series_aroon_down

    df['adx_' + str(window)] = adx(df[h], df[l], df[c], n=window)

    return df


def adjust_ratio(data, name, ac, c):
    """
    Given a DataFrame of stock prices a column, return the Series with the adjusted values of that column

    :param data: DataFrame containing stock prices
    :param name: Column to be adjusted
    :param ac: ADJUSTED CLOSE column
    :param c: CLOSE column
    :return: Series of adjusted NAME column
    """
    x = [data[name][i] * (data[ac][i] / data[c][i]) for i in range(len(data))]
    return pd.Series(x)


def normalize(data):
    """
    Performs Z-score normalization

    :param data: Array to be normalized
    :return: Normalized data, mean of data, standard deviation of data
    """
    m, s = np.mean(data), np.std(data)
    data = (data - m) / s
    return data, m, s


def normalize_test(data, m, s):
    """
    Performs Z-score normalization for test data

    :param data: Array to be normalized
    :param m: Mean of data
    :param s: Standard deviation
    :return: Normalized data
    """
    return (data - m) / s


def inverse_normalize(data, m, s):
    """
    Performs inverse Z-score normalization

    :param data: Normalized array
    :param m: Original mean of data
    :param s: Original standard deviation
    :return: De-normalized data
    """
    return (data * s) + m
