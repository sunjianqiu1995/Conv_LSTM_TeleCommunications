"""
http://xtf615.com/2017/03/08/Python%E5%AE%9E%E7%8E%B0%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E5%88%86%E6%9E%90/

http://www.36dsj.com/archives/44065
"""
import requests, pandas as pd, numpy as np
from pandas import DataFrame
from io import StringIO
import time, json
from datetime import date
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA, ARIMAResults
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import os, sklearn
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import sys


# 注意这里面使用的ts_log_diff是经过合适阶数的差分之后的数据
# 上文中提到ARIMA该开源库，不支持3阶以上的#差分。所以我们需要提前将数据差分好再传入
# 求解最佳模型参数p,q
def _proper_model(ts_log_diff, maxLag):
    best_p = 0
    best_q = 0
    best_bic = sys.maxsize
    best_model = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            try:
                model = ARMA(ts_log_diff, order=(p, q))
                results_ARMA = model.fit(disp=-1)
            except:
                continue
            bic = results_ARMA.bic
            # print(bic, best_bic)
            if bic < best_bic:
                best_p = p
                best_q = q
                best_bic = bic
                best_model = results_ARMA
    return best_p, best_q, best_model


print(os.getcwd())

Sin = np.load('../data/data_5features_narry/SMSIn_9999*8784.npy')  # 9999 * timecount
Sout = np.load('../data/data_5features_narry/SMSOut_9999*8784.npy')
Cin = np.load('../data/data_5features_narry/CallIn_9999*8784.npy')
Cout = np.load('../data/data_5features_narry/CallOut_9999*8784.npy')
Itra = np.load('../data/data_5features_narry/InterTra_9999*8784.npy')
print(Sin.shape)

data_test = np.load('../data_test/test_total.npy')  # (144,100,100,5)
data_test = data_test.reshape(144, 10000, 5)
print(data_test.shape)

timeseries = [Sin, Sout, Cin, Cout, Itra]
timeseriesname = ['Sin', 'Sout', 'Cin', 'Cout', 'Itra']


def pre_data(timeseries_fea, feaID, GridID):
    """
    prepare timeSeries and timeSeeries_diff
    """
    ts = pd.Series(timeseries_fea[GridID][-2016:])  # [8784]
    ts_diff = ts - ts.shift()
    ts_diff.dropna(inplace=True)
    X = ts_diff.values
    ts_diff = X.astype('float32')  # numpy array
    X = ts.values
    ts = X.astype('float32')  # numpy array

    y_true = data_test[:, GridID, feaID]  # [144]

    return ts, ts_diff, y_true


# monkey patch around bug in ARIMA class for save model
def __getnewargs__(self):
    return ((self.endog), (self.k_lags, self.k_diff, self.k_ma))


# train model and save model
def train_save_arima(ts, save_path, best_p, best_q):
    a = 0
    try:
        model = ARIMA(ts, order=(best_p, 1, best_q)) # 一阶差分
        results_ARIMA = model.fit(disp=-1)
        # Save Models
        ARIMA.__getnewargs__ = __getnewargs__
        # save model
        results_ARIMA.save(save_path)
    except:
        model = ARIMA(ts, order=(best_p, 1, best_q)) # 一阶差分
        results_ARIMA = model.fit(transparams=False, disp=-1)
        # Save Models
        ARIMA.__getnewargs__ = __getnewargs__
        # save model
        results_ARIMA.save(save_path)
        a = a+1
        print('pass')

    return results_ARIMA,a


# predict and eval model
def pred_eval_model(results_ARIMA, ts, forecast_n, y_true, fig_save_path):
    # forecast方法会自动进行差分还原，当然仅限于支持的1阶和2阶差分
    # forecast_n = 144  # 预测未来12个月走势
    forecast_ARIMA_log = results_ARIMA.forecast(forecast_n)
    forecast_ARIMA_log = forecast_ARIMA_log[0]
    # print(forecast_ARIMA_log[:144])

    MSE = sklearn.metrics.mean_squared_error(y_true, forecast_ARIMA_log)
    # diff = y_true - forecast_ARIMA_log  # [144]
    y_true_mean = np.mean(y_true)
    acc = MSE / y_true_mean

    forecast_ARIMA_log = pd.Series(forecast_ARIMA_log,
                                   index=np.arange(len(ts[-2016:]) + 1, len(ts[-2016:]) + len(forecast_ARIMA_log) + 1,
                                                   1))
    y_true = pd.Series(y_true, index=np.arange(2016 + 1, 2016 + 144 + 1, 1))
    return MSE, acc


# predict model
for feaID in range(len(timeseriesname)):
    filename = open('./log_test_%s.log'%timeseriesname[feaID],'w')
    MSE_arr = []
    acc_arr = []
    for gridID in range(0,len(Sin),2500):
        # Prepare train data and test data
        ts, ts_diff, y_true = pre_data(timeseries[feaID], feaID, gridID)

        # optimize the best model parameters
        best_p, best_q, model_ama = _proper_model(ts_diff, 10)  # 对一阶差分求最优p和q
        print(best_p, best_q)
        filename.write('Grid %d predict stat Info:\n'%gridID)
        filename.write('Best_p:%s, Best_q:%s\n'%(str(best_p),str(best_q)))

        # train and save model
        save_path = './train_model_arima_save/train_model_save_%s/ARIMA_model_%s_grid%d.pkl' % (
            timeseriesname[feaID], timeseriesname[feaID], gridID)
        results_ARIMA, a  = train_save_arima(ts, save_path, best_p, best_q)
        if a == 1:
            filename.write('###########################################parameters not converge very weill. \n')

        fig_save_path = './predict_figure_save/predict_figure_save_%s/ARIMA_fig_%s_grid%d.png' % (timeseriesname[feaID], timeseriesname[feaID], gridID)
        MSE, acc = pred_eval_model(results_ARIMA,ts=ts, forecast_n=144,y_true=y_true,fig_save_path=fig_save_path)
        filename.write('MSE: %s, acc: %s\n'%(str(MSE),str(acc)))
        MSE_arr.append(MSE)
        acc_arr.append(acc)

    MSE_mean = np.mean(MSE_arr)
    acc_mean = np.mean(acc_arr)
    print(MSE_mean)
    filename.write('MSE_mean: %s, acc_mean: %s\n'%(str(MSE_mean),str(acc_mean)))
    filename.close()
    MSE_ts = pd.Series(MSE_arr)
    acc_ts = pd.Series(acc_arr)
