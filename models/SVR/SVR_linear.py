import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.externals import joblib



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

# x axis
X = np.arange(0, 2016, 1)  # [8784,1]
X = X.reshape(2016, 1)

X_future = np.arange(0,2016+144,1).reshape(2016+144,1) #[8784+144,1]

print(__doc__)

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

def train_save_eval_model(timeseries_fea,feaID,gridID,y_true,save_fig_path,save_model_path):
    # Generate sample data
    y = timeseries_fea[gridID][-2016:] #[8784]
    # #############################################################################
    # Fit regression model
    svr_rbf = SVR(kernel='linear', C=1e3, gamma=0.1)
    # svr_lin = SVR(kernel='linear', C=1e3)
    # svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    results_SVR = svr_rbf.fit(X, y)
    # results_SVR.save(save_model_path)
    joblib.dump(results_SVR,save_model_path)
    # results_SVR = joblib.load('filename.pkl')
    y_rbf = svr_rbf.fit(X, y).predict(X_future[-144:]) #[144]
    print('finish linear')
    MSE = sklearn.metrics.mean_squared_error(y_true, y_rbf[-144:])
    y_true_mean = np.mean(y_true)
    acc = MSE / y_true_mean

    y_rbf = pd.Series(y_rbf,index=np.arange(2016+1,2016+144+1,1))
    y_true = pd.Series(y_true,index=np.arange(2016+1,2016+144+1,1))

    plt.plot(y[-2016:], color="blue", label='Original')
    plt.plot(y_true, color="navy", label='y_true')
    plt.plot(y_rbf, color='red', label='Predicted')
    plt.legend(loc='best')
    plt.title('SVR linear MSE: %.4f ACC: %.4f' % (MSE, acc))
    plt.xlim([0, 2016 + 144])
    # show the biggest figure
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    fig = plt.gcf()
    # plt.show()
    fig.savefig(save_fig_path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    return y_rbf.values,MSE,acc

for feaID in range(len(timeseriesname)):
    filename = open('./log_test_svr_linear_%s.log'%timeseriesname[feaID],'w')
    MSE_arr = []
    acc_arr = []
    for gridID in range(0,len(Sin),250):
        y_true = data_test[:, gridID, feaID]
        save_fig_path = './svr_linear_predict_figure_save/svr_predict_figure_save_%s/SVR_%s_grid%d.png' % (timeseriesname[feaID], timeseriesname[feaID], gridID)
        save_model_path = './svr_linear_train_model_store/svr_train_model_store_%s/SVR_model_%s_grid%d.pkl' % (timeseriesname[feaID], timeseriesname[feaID], gridID)

        y_pred, MSE, acc = train_save_eval_model(timeseries[feaID],feaID,gridID,y_true,save_fig_path,save_model_path)
        filename.write('SVR - Grid %d predict stat Info:\n' % gridID)
        filename.write('MSE: %s, acc: %s\n' % (str(MSE), str(acc)))
        filename.write(y_pred)
        filename.write('\n')
        MSE_arr.append(MSE)
        acc_arr.append(acc)

    MSE_mean = np.mean(MSE_arr)
    acc_mean = np.mean(acc_arr)
    print(MSE_mean)
    filename.write('MSE_mean: %s, acc_mean: %s\n'%(str(MSE_mean),str(acc_mean)))
    filename.close()
    MSE_ts = pd.Series(MSE_arr)
    acc_ts = pd.Series(acc_arr)
    plt.plot(MSE_ts, color="blue", label='MSE_timeSeries')
    plt.plot(acc_ts, color='red', label='Acc_timeSeries')
    plt.legend(loc='best')
    plt.title('%s linear MSE_mean: %s, acc_mean: %s\n'%(timeseriesname[feaID],str(MSE_mean),str(acc_mean)))
    mse_acc_save_path = './linear_MSE_ACC_%s.png'%timeseriesname[feaID]
    plt.xlim([0, len(MSE_arr)])
    # show the biggest figure
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    fig = plt.gcf()
    #plt.show()
    fig.savefig(mse_acc_save_path, bbox_inches='tight',dpi=100)
    plt.close(fig)





#
# # #############################################################################
# # Look at the results
# lw = 2
# plt.scatter(X, y, color='darkorange', label='data')
# plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
# # plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
# plt.xlabel('data')
# plt.ylabel('target')
# plt.title('Support Vector Regression')
# plt.legend()
# plt.show()
