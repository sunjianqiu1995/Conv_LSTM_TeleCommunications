# -*- coding: utf-8 -*-
"""
Python3.6
Get Communication Data's statistical information, 
like autocorrelation, mean and deviation, spatial information, 
temporal information, etc. Save the figure in the folder ./Figures

"""
from __future__ import print_function
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

print(os.getcwd())
path = '../data'

timecount = 8784
print(timecount)
'''
# load data from dictionary to narray , and save as narry
Sin = []
Sout = []
Cin = []
Cout = []
Itra = []
for i in range(1, 10000):
    if os.path.isfile('../data/%d.pickle' % i):
        with open('../data/%d.pickle' % i, 'rb') as handle:
            dicta = pickle.load(handle)
        smsin = dicta['SMSin'] # array
        smsout = dicta['SMSout']
        callin = dicta['Callin']
        callout = dicta['Callout']
        internettra = dicta['InternetTra']

        Sin.append(smsin)  # 9999 * timecount
        Sout.append(smsout)  # 9999 * timecount
        Cin.append(callin)  # 9999 * timecount
        Cout.append(callout)  # 9999 * timecount
        Itra.append(internettra)  # 9999 * timecount


# list to array
Sin = np.array(Sin) # 9999 * timecount
Sout = np.array(Sout)
Cin = np.array(Cin)
Cout = np.array(Cout)
Itra = np.array((Itra))

np.save('../data/data_5features_narry/SMSIn_9999*8784.npy',Sin)
np.save('../data/data_5features_narry/SMSOut_9999*8784.npy',Sout)
np.save('../data/data_5features_narry/CallIn_9999*8784.npy',Cin)
np.save('../data/data_5features_narry/CallOut_9999*8784.npy',Cout)
np.save('../data/data_5features_narry/InterTra_9999*8784.npy',Itra)
'''

Sin = np.load('../data/data_5features_narry/SMSIn_9999*8784.npy')  # 9999 * timecount
Sout = np.load('../data/data_5features_narry/SMSOut_9999*8784.npy')
Cin = np.load('../data/data_5features_narry/CallIn_9999*8784.npy')
Cout = np.load('../data/data_5features_narry/CallOut_9999*8784.npy')
Itra = np.load('../data/data_5features_narry/InterTra_9999*8784.npy')
print(Sin.shape)

'''
# x axis length
xlen = len(Sin[0])
# Figure, x axis
xtimes = np.arange(1, xlen + 1)  # [1,xlen]

######################################################
# Stat Info 1: For Each grid, plot 5 features.
def difffeatures(xtimes, xlen, savepath, figurenum):
    """
    For Each grid, plot 5 features.
    :param xtimes: Figure, x axis.
    :param xlen: the length of x. xlen = len(xtimes)
    :param savepath: Figures saving paths.
    :param figurenum: Figure number.
    :return:
    """
    # create a figure
    #plt.figure()
    f, axarr = plt.subplots(5, sharex=True, sharey=False)
    f.suptitle('Grid %d'%figurenum) # titles

    # SMSIn
    plt.subplot(511)
    plt.plot(xtimes,Sin[figurenum-1])
    #plt.axis([1,xlen,0,2])
    plt.xlim([1,xlen])
    plt.ylabel('SMSIn')
    axarr[0].xaxis.grid(True, which='major') # set x grid
    #plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

    # SMSOut
    plt.subplot(512)
    plt.plot(xtimes,Sout[figurenum-1])
    #plt.axis([1,xlen])
    plt.xlim([1,xlen])
    plt.ylabel('SMSout')
    axarr[1].xaxis.grid(True, which='major') # set x grid

    # CallIn
    plt.subplot(513)
    plt.plot(xtimes,Cin[figurenum-1])
    #plt.axis([1,xlen])
    plt.xlim([1,xlen])
    plt.ylabel('CallIn')
    axarr[2].xaxis.grid(True, which='major') # set x grid

    # CallOut
    plt.subplot(514)
    plt.plot(xtimes,Cout[figurenum-1])
    #plt.axis([1,xlen])
    plt.xlim([1,xlen])
    plt.ylabel('Callout')
    axarr[3].xaxis.grid(True, which='major') # set x grid

    # InternationalTraffic
    plt.subplot(515)
    plt.plot(xtimes,Itra[figurenum-1])
    #plt.axis([1,xlen])
    plt.xlim([1,xlen])
    plt.ylabel('InterTra')
    axarr[4].xaxis.grid(True, which='major') # set x grid

    # Bring subplots close to each other.
    f.subplots_adjust(hspace=0)
    # Hide x labels and tick labels for all but bottom plot.
    for ax in axarr:
        ax.label_outer()
    # set x steps
    plt.xticks(np.arange(1,xlen+1,288))
    # show the biggest figure
    # Option 1
    # QT backend
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    # plt.show(block=False) # non-interactive mode
    # timer = f.canvas.new_timer(interval=3000)  # creating a timer object and setting an interval of 3000 milliseconds
    # timer.add_callback(plt.close(f))
    # timer.start()
    plt.show()
    f.savefig(savepath, bbox_inches='tight')
    plt.close(f)
    # plt.clf()

# create figures
savepath = '../Figures/Figures_two_weeks/'
for i in range(0, 9999, 75):
    difffeatures(xtimes = xtimes, xlen = xlen, savepath=savepath+'%d_5features_Twoweeks.png'%(i+1), figurenum = i+1)


######################################################
# Stat Info 2: For each feature, plot comparision between grids, (every 5 grids).
def comparegrids(xtimes, xlen, feaarray, feaname, savepath, figurenum):
    """
    For Each grid, plot 5 features.
    :param feaarray: Sin, Sout, Cin, Cout, Itra.
    :param xlen: the length of x. xlen = len(xtimes)
    :param savepath: Figures saving paths.
    :param feaname: Feature name.
    :param figurenum: Step: every 75 grids -->figurenum is the number of steps.
    :return:
    """
    # create a figure
    f, axarr = plt.subplots(10, sharex=True, sharey=False)
    f.suptitle('%s'%feaname) # titles
    if figurenum + 75*10 < 10000:
        for i in range(10):
            axarr[i].plot(xtimes,feaarray[figurenum + 75*i -1])
            axarr[i].set_xlim([1,xlen])
            axarr[i].set_ylabel('%d'%(figurenum + 75*i))
            axarr[i].xaxis.grid(True, which='major') # set x grid
    else:
        step = (10000-figurenum)//10
        for i in range(10):
            axarr[i].plot(xtimes,feaarray[figurenum + step*i -1])
            axarr[i].set_xlim([1,xlen])
            axarr[i].set_ylabel('%d'%(figurenum + step*i))
            axarr[i].xaxis.grid(True, which='major') # set x grid

    # Bring subplots close to each other.
    f.subplots_adjust(hspace=0)
    # Hide x labels and tick labels for all but bottom plot.
    for ax in axarr:
        ax.label_outer()
    # set x steps
    plt.xticks(np.arange(1,xlen+1,288))

    # show the biggest figure
    # QT backend
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.show()
    f.savefig(savepath, bbox_inches='tight')
    plt.close(f)
    # plt.clf()

# create figures: SMSIn
savepath = '../Figures/SMSIn_1/'
for i in range(0, 9999, 75*10):
    comparegrids(xtimes = xtimes, xlen = xlen, feaarray=Sin, feaname='SMSIn', savepath=savepath+'SMSIn_%d-%d.png'%(i+1,(i)+75*10), figurenum = i+1)

# create figures: SMSOut
savepath = '../Figures/SMSOut_1/'
for i in range(0, 9999, 75*10):
    comparegrids(xtimes = xtimes, xlen = xlen, feaarray=Sout, feaname='SMSOut', savepath=savepath+'SMSOut_%d-%d.png'%(i+1,(i)+75*10), figurenum = i+1)

# create figures: CallIn
savepath = '../Figures/CallIn_1/'
for i in range(0, 9999, 75*10):
    comparegrids(xtimes = xtimes, xlen = xlen, feaarray=Cin, feaname='CallIn', savepath=savepath+'CallIn_%d-%d.png'%(i+1,(i)+75*10), figurenum = i+1)

# create figures: CallOut
savepath = '../Figures/CallOut_1/'
for i in range(0, 9999, 75*10):
    comparegrids(xtimes = xtimes, xlen = xlen, feaarray=Cout, feaname='CallOut', savepath=savepath+'CallOut_%d-%d.png'%(i+1,(i)+75*10), figurenum = i+1)
# create figures: InternetTra
savepath = '../Figures/InternetTra_1/'
for i in range(0, 9999, 75*10):
    comparegrids(xtimes = xtimes, xlen = xlen, feaarray=Itra, feaname='InternetTra', savepath=savepath+'InternetTra_%d-%d.png'%(i+1,(i)+75*10), figurenum = i+1)
'''

# Two weeks
# x axis length
xlen_twoweeks = len(Sin[0][:6 * 24 * 14])
# Figure, x axis
xtimes_twoweeks = np.arange(1, xlen_twoweeks + 1)  # [1,xlen]

'''
######################################################
# Stat Info 3: For Each grid, plot 5 features.- two weeks
def difffeatures_twoweeks(xtimes, xlen, savepath, figurenum):
    """
    For Each grid, plot 5 features.
    :param xtimes: Figure, x axis.
    :param xlen: the length of x. xlen = len(xtimes)
    :param savepath: Figures saving paths.
    :param figurenum: Figure number.
    :return:
    """
    # create a figure
    f, axarr = plt.subplots(5, sharex=True, sharey=False)
    f.suptitle('Grid %d' % figurenum)  # titles

    # plt.subplot(511).xaxis.grid(True, which='major')  # set x grid
    # plt.plot(xtimes, Sin[figurenum - 1][:xlen])
    # # plt.axis([1,xlen,0,2])
    # plt.xlim([1, xlen])
    # plt.ylabel('SMSIn')
    feaarray = [Sin[figurenum - 1][:xlen], Sout[figurenum - 1][:xlen], Cin[figurenum - 1][:xlen],
                Cout[figurenum - 1][:xlen], Itra[figurenum - 1][:xlen]]
    feaname = ['SMSIn', 'SMSOut', 'CallIn', 'CallOut', 'InternetTra']
    for i in range(5):
        axarr[i].plot(xtimes, feaarray[i])
        axarr[i].set_xlim([1, xlen])
        axarr[i].set_ylabel(feaname[i])
        axarr[i].xaxis.grid(True, which='major')  # set x grid

    # Bring subplots close to each other.
    f.subplots_adjust(hspace=0)
    # Hide x labels and tick labels for all but bottom plot.
    for ax in axarr:
        ax.label_outer()
    # set x steps
    plt.xticks(np.arange(1, xlen + 1, 72))
    # show the biggest figure
    # Option 1
    # QT backend
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    # plt.show(block=False) # non-interactive mode
    # timer = f.canvas.new_timer(interval=3000)  # creating a timer object and setting an interval of 3000 milliseconds
    # timer.add_callback(plt.close(f))
    # timer.start()
    plt.show()
    f.savefig(savepath, bbox_inches='tight')
    plt.close(f)
    # plt.clf()


# two weeks
savepath = '../Figures/Figures_Grids_two_weeks/'
for i in range(0, 9999, 75):
    difffeatures_twoweeks(xtimes=xtimes_twoweeks, xlen=xlen_twoweeks,
                          savepath=savepath + '%d_5features_Twoweeks.png' % (i + 1), figurenum=i + 1)


# Stat Info 4: For each feature, plot comparision between grids, (every 5 grids). - Two weeks
def comparegrids_two_weeks(xtimes, xlen, feaarray, feaname, savepath, figurenum):
    """
    For Each grid, plot 5 features.
    :param feaarray: Sin, Sout, Cin, Cout, Itra.
    :param xlen: the length of x. xlen = len(xtimes)
    :param savepath: Figures saving paths.
    :param feaname: Feature name.
    :param figurenum: Step: every 75 grids -->figurenum is the number of steps.
    :return:
    """

    # create a figure
    f, axarr = plt.subplots(10, sharex=True, sharey=False)
    f.suptitle('%s_Twoweeks_1-2016'%feaname) # titles
    if figurenum + 75*10 < 10000:
        for i in range(10):
            axarr[i].plot(xtimes,feaarray[figurenum + 75*i -1][:xlen])
            axarr[i].set_xlim([1,xlen])
            axarr[i].set_ylabel('%d'%(figurenum + 75*i))
            axarr[i].xaxis.grid(True, which='major') # set x grid
    else:
        step = (10000-figurenum)//10
        for i in range(10):
            axarr[i].plot(xtimes,feaarray[figurenum + step*i -1][:xlen])
            axarr[i].set_xlim([1,xlen])
            axarr[i].set_ylabel('%d'%(figurenum + step*i))
            axarr[i].xaxis.grid(True, which='major') # set x grid

    # Bring subplots close to each other.
    f.subplots_adjust(hspace=0)
    # Hide x labels and tick labels for all but bottom plot.
    for ax in axarr:
        ax.label_outer()
    # set x steps
    plt.xticks(np.arange(1,xlen+2,72))

    # show the biggest figure
    # QT backend
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.show()
    f.savefig(savepath, bbox_inches='tight')
    plt.close(f)
    # plt.clf()

# create figures: SMSIn
savepath = '../Figures/Figures_Features_Two_Weeks/'
for i in range(0, 9999, 75*10):
    comparegrids_two_weeks(xtimes = xtimes_twoweeks, xlen = xlen_twoweeks, feaarray=Sin, feaname='SMSIn', savepath=savepath+'SMSIn_%d-%d_Two_weeks.png'%(i+1,(i)+75*10), figurenum = i+1)

# create figures: SMSOut
for i in range(0, 9999, 75*10):
    comparegrids_two_weeks(xtimes = xtimes_twoweeks, xlen = xlen_twoweeks, feaarray=Sout, feaname='SMSOut', savepath=savepath+'SMSOut_%d-%d_Two_weeks.png'%(i+1,(i)+75*10), figurenum = i+1)

# create figures: CallIn
for i in range(0, 9999, 75*10):
    comparegrids_two_weeks(xtimes = xtimes_twoweeks, xlen = xlen_twoweeks, feaarray=Cin, feaname='CallIn', savepath=savepath+'CallIn_%d-%d_Two_weeks.png'%(i+1,(i)+75*10), figurenum = i+1)

# create figures: CallOut
for i in range(0, 9999, 75*10):
    comparegrids_two_weeks(xtimes = xtimes_twoweeks, xlen = xlen_twoweeks, feaarray=Cout, feaname='CallOut', savepath=savepath+'CallOut_%d-%d_Two_weeks.png'%(i+1,(i)+75*10), figurenum = i+1)
# create figures: InternetTra
for i in range(0, 9999, 75*10):
    comparegrids_two_weeks(xtimes = xtimes_twoweeks, xlen = xlen_twoweeks, feaarray=Itra, feaname='InternetTra', savepath=savepath+'InternetTra_%d-%d_Two_weeks.png'%(i+1,(i)+75*10), figurenum = i+1)

'''
# Two Days
# x axis length
xlen_twodays = len(Sin[0][:288])
# Figure, x axis
xtimes_twodays = np.arange(1, xlen_twodays + 1)  # [1,xlen]
'''
# Stat Info 5: For each feature, plot comparision between grids, (every 5 grids). - Two Days
def comparegrids_two_days(xtimes, xlen, feaarray, feaname, savepath, figurenum):
    """
    For Each grid, plot 5 features.
    :param feaarray: Sin, Sout, Cin, Cout, Itra.
    :param xlen: the length of x. xlen = len(xtimes)
    :param savepath: Figures saving paths.
    :param feaname: Feature name.
    :param figurenum: Step: every 75 grids -->figurenum is the number of steps.
    :return:
    """

    # create a figure
    f, axarr = plt.subplots(10, sharex=True, sharey=False)
    f.suptitle('%s_TwoDays_1-288'%feaname) # titles
    if figurenum + 75*10 < 10000:
        for i in range(10):
            axarr[i].plot(xtimes,feaarray[figurenum + 75*i -1][:xlen])
            axarr[i].set_xlim([1,xlen])
            axarr[i].set_ylabel('%d'%(figurenum + 75*i))
            axarr[i].xaxis.grid(True, which='major') # set x grid
            axarr[i].yaxis.grid(True, which='minor') # set y grid
    else:
        step = (10000-figurenum)//10
        for i in range(10):
            axarr[i].plot(xtimes,feaarray[figurenum + step*i -1][:xlen])
            axarr[i].set_xlim([1,xlen])
            axarr[i].set_ylabel('%d'%(figurenum + step*i))
            axarr[i].xaxis.grid(True, which='major') # set x grid

    # Bring subplots close to each other.
    f.subplots_adjust(hspace=0)
    # Hide x labels and tick labels for all but bottom plot.
    for ax in axarr:
        ax.label_outer()
    # set x steps
    plt.xticks(np.arange(1,xlen+1,12))

    # show the biggest figure
    # QT backend
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.show()
    f.savefig(savepath, bbox_inches='tight')
    plt.close(f)
    # plt.clf()

# create figures: SMSIn
savepath = '../Figures/Figures_Features_Two_Days/'
for i in range(0, 9999, 75*10):
    comparegrids_two_days(xtimes = xtimes_twodays, xlen = xlen_twodays, feaarray=Sin, feaname='SMSIn', savepath=savepath+'SMSIn_%d-%d_Two_Days.png'%(i+1,(i)+75*10), figurenum = i+1)

# create figures: SMSOut
savepath = '../Figures/SMSOut_1/'
for i in range(0, 9999, 75*10):
    comparegrids_two_days(xtimes = xtimes_twodays, xlen = xlen_twodays, feaarray=Sout, feaname='SMSOut', savepath=savepath+'SMSOut_%d-%d_Two_Days.png'%(i+1,(i)+75*10), figurenum = i+1)

# create figures: CallIn
savepath = '../Figures/CallIn_1/'
for i in range(0, 9999, 75*10):
    comparegrids_two_days(xtimes = xtimes_twodays, xlen = xlen_twodays, feaarray=Cin, feaname='CallIn', savepath=savepath+'CallIn_%d-%d_Two_Days.png'%(i+1,(i)+75*10), figurenum = i+1)

# create figures: CallOut
savepath = '../Figures/CallOut_1/'
for i in range(0, 9999, 75*10):
    comparegrids_two_days(xtimes = xtimes_twodays, xlen = xlen_twodays, feaarray=Cout, feaname='CallOut', savepath=savepath+'CallOut_%d-%d_Two_Days.png'%(i+1,(i)+75*10), figurenum = i+1)
# create figures: InternetTra
savepath = '../Figures/InternetTra_1/'
for i in range(0, 9999, 75*10):
    comparegrids_two_days(xtimes = xtimes_twodays, xlen = xlen_twodays, feaarray=Itra, feaname='InternetTra', savepath=savepath+'InternetTra_%d-%d_Two_Days.png'%(i+1,(i)+75*10), figurenum = i+1)
'''

# # SMSIn
# plt.subplot(511).xaxis.grid(True, which='major')  # set x grid
# plt.plot(xtimes, Sin[figurenum - 1][:xlen])
# # plt.axis([1,xlen,0,2])
# plt.xlim([1, xlen])
# plt.ylabel('SMSIn')
# # plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

######################################################
######################################################
# Stat Info 6: Python - time series - rolling stat与ADF共同检验平稳性
# http://www.36dsj.com/archives/44065
# https://www.leiphone.com/news/201703/6rVkgxvxUumnv5mm.html
# http://xtf615.com/2017/03/08/Python%E5%AE%9E%E7%8E%B0%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E5%88%86%E6%9E%90/
# Stat Info 6: Python - time series - Rolling Statistics: Means & Standards Deviation -
# rolling stat与ADF共同检验平稳性
# http://xtf615.com/2017/03/08/Python%E5%AE%9E%E7%8E%B0%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E5%88%86%E6%9E%90/
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def rolling_statistics_adf_test(timeseries, window, savepath, ii):
    """
    # 输出如何解读?
    # Test statistic：代表检验统计量
    # p-value：代表p值检验的概率
    # Lags used：使用的滞后k，autolag=AIC时会自动选择滞后
    # Number of Observations Used：样本数量
    # Critical Value(5%) : 显著性水平为5%的临界值。

    # ADF检验
    # 假设是存在单位根，即不平稳；
    # 显著性水平，1%：严格拒绝原假设；5%：拒绝原假设，10%类推。
    # 看P值和显著性水平a的大小，p值越小，小于显著性水平的话，就拒绝原假设，认为序列是平稳的；大于的话，不能拒绝，认为是不平稳的
    # 看检验统计量和临界值，检验统计量小于临界值的话，就拒绝原假设，认为序列是平稳的；大于的话，不能拒绝，认为是不平稳的
    # 根据上文提到的p值检验以及上面的结果，我们可以发现p=0.99>10%>5%>1%, 并且检验统计量0.815>>-2.58>-2.88>-3.48，因此可以认定原序列不平稳。
    #
    # 在telecom数据中，应该大部分是平稳的。

    :param timeseries:时间序列
    :param window:window代表多少值进行平均
    :param savepath:figure保存路径
    :param ii:时间序列array中的index
    :return:
    """
    # def rolling_statistics(timeseries, window, savepath):
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=window)  # window代表多少值进行平均
    rolstd = pd.rolling_std(timeseries, window=window)
    # Plot rolling statistics:
    plt.figure(figsize=(38.40 * 1.2, 24.00 * 1.2))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation_%d' % (ii + 1))
    plt.xlim([0, len(timeseries)])

    # show the biggest figure
    # QT backend
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.show(block=False)
    manager.savefig(savepath, bbox_inches='tight')
    plt.close()

    # def adf_test(timeseries, window, savepath):
    print('Results of Augment Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    # print(dfoutput)
    if (dfoutput['Test Statistic'] > dfoutput['Critical Value (1%)']) or (dfoutput['p-value'] > 0.01):
        print('%d不平稳' % ii)


timeseries = [Sin, Sout, Cin, Cout, Itra]
timeseriesname = ['Sin', 'Sout', 'Cin', 'Cout', 'Itra']

# for id in range(len(timeseriesname)):
#     for i in range(0,8784,1500):
#         savepath =  '../Figures/Rolling Mean&Standard Deviation/RollingMean&StadDeviation_%s[:,%d]_spatial.png'%(timeseriesname[id],i)
#         rolling_statistics_adf_test(timeseries[id][:,i],window=2,savepath=savepath,ii=i)
#     for j in range(0,9999,750):
#         savepath =  '../Figures/Rolling Mean&Standard Deviation/RollingMean&StadDeviation_%s[%d]_temporal.png'%(timeseriesname[id],j)
#         rolling_statistics_adf_test(timeseries[id][j],window=6,savepath=savepath,ii=j)


######################################################
# Stat Info 7: Python - time series - autocorrelation 用于对平稳性进行再次验证
# 这里我们使用自相关图和偏自相关图对数据平稳性再次进行验证，一阶差分如下图：
# 蓝色区域中的点表示统计学显着性。滞后值为 0 相关性为 1 的点表示观察值与其本身 100% 正相关。
# 可以看到，图中在 1,2,12 和 17 个月显示出了显著的滞后性。
# 这个分析为后续的比较过程提供了一个很好的基准。

# 自相关函数（ACF）：这是时间序列和它自身滞后版本之间的相关性的测试。比如在自相关函数可以比较时间的瞬间‘t1’…’t2’以及序列的瞬间‘t1-5’…’t2-5’ (t1-5和t2 是结束点)。
# 部分自相关函数(PACF):这是时间序列和它自身滞后版本之间的相关性测试，但是是在预测（已经通过比较干预得到解释）的变量后。如：滞后值为5，它将检查相关性，但是会删除从滞后值1到4得到的结果。
from pandas import Series
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot


def acf_pacf_plot(ts_log_diff,savepath,ii,lags):
    """

    :param ts_log_diff: 时间序列array
    :param savepath: figure保存路径
    :param ii: 时间序列array的index
    :param lags:滞后值
    :return:
    """
    fig = plot_acf(ts_log_diff, lags=lags, title='Autocorrelation_%d'%ii)  # ARIMA,q
    # plot_pacf(ts_log_diff, lags=2016) #ARIMA,p
    #plt.figsize=(38.40 * 1.2, 24.00 * 1.2)
    plt.xlim([0, len(ts_log_diff)])

    # show the biggest figure
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    fig = plt.gcf()
    plt.show()
    fig.savefig(savepath, bbox_inches='tight',dpi=100)
    plt.close(fig)

    # pyplot.show()

# temporal analysis
for id in range(len(timeseriesname[0])):
    for j in range(0, 9999, 750):
        savepath = '../Figures/Autocorrelation/Autocorrelation_%s[%d]_temporal.png' % (
            timeseriesname[id], j)
        acf_pacf_plot(timeseries[id][j],savepath,ii=j,lags=8784-144)

# # spacial analysis
# for id in range(len(timeseriesname)):
#     for j in range(0, 8784, 2000):
#         savepath = '../Figures/Autocorrelation_Spatial/Autocorrelation_%s[:,%d]_spacial.png' % (
#             timeseriesname[id], j)
#         acf_pacf_plot(timeseries[id][:,j],savepath,ii=j,lags=9995)



# 计算自相关性的值
def autocorrelation(x, lags):  # 计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
    n = len(x)
    x = np.array(x)
    result = [np.correlate(x[i:] - x[i:].mean(), x[:n - i] - x[:n - i].mean())[0] \
              / (x[i:].std() * x[:n - i].std() * (n - i)) \
              for i in range(1, lags + 1)]
    return result

# print(autocorrelation(Sin[0], lags=40))

#########################################################################
# Stat Info 8: 对spatial数据进行平稳性处理

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    """
    The same as rolling_statistics_adf_test()
    :param timeseries:
    :return:
    """
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    # Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


# 差分操作,d代表差分序列，比如[1,1,1]可以代表3阶差分。  [12,1]可以代表第一次差分偏移量是12，第二次差分偏移量是1
def diff_ts(ts, d):
    global shift_ts_list
    #  动态预测第二日的值时所需要的差分序列
    global last_data_shift_list #这个序列在恢复过程中需要用到
    shift_ts_list = []
    last_data_shift_list = []
    tmp_ts = ts
    for i in d:
        last_data_shift_list.append(tmp_ts[-i])
        print(last_data_shift_list)
        shift_ts = tmp_ts.shift(i)
        shift_ts_list.append(shift_ts)
        tmp_ts = tmp_ts - shift_ts
    tmp_ts.dropna(inplace=True)
    return tmp_ts
# 还原操作
def predict_diff_recover(predict_value, d):
    if isinstance(predict_value, float):
        tmp_data = predict_value
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    elif isinstance(predict_value, np.ndarray):
        tmp_data = predict_value[0]
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    else:
        tmp_data = predict_value
        for i in range(len(d)):
            try:
                tmp_data = tmp_data.add(shift_ts_list[-i-1])
            except:
                raise ValueError('What you input is not pd.Series type!')
        tmp_data.dropna(inplace=True)
    return tmp_data # return np.exp(tmp_data)也可以return到最原始，tmp_data是对原始数据取对数的结果

#
#
# ts = pd.Series(Sin[0][:2016])
# #print(ts)
# # ts = Sin[0][:2016]
# d=[1] # 定义差分序列
# ts_log = np.log(ts+1)
# # diffed_ts = diff_ts(ts_log, d)
# diffed_ts = ts_log - ts_log.shift(periods=1)
# from statsmodels.tsa.arima_model import ARIMA
# #ARIMA
# model = ARIMA(ts_log, order=(1, 1, 1))
# results_ARIMA = model.fit(disp=-1)  #fit
# predict_ts = model.predict() #对训练数据进行预测
# #还原
# diff_recover_ts = predict_diff_recover(predict_ts, d=[1])#恢复数据
# log_recover = np.exp(diff_recover_ts-1)#还原对数前数据
# #绘图
# #ts = ts[log_recover.index]#排除空的数据
# plt.plot(ts,color="blue",label='Original')
# plt.plot(log_recover,color='red',label='Predicted')
# plt.legend(loc='best')
# plt.title('RMSE: %.4f'% np.sqrt(sum((log_recover-ts)**2)/len(ts)))#RMSE,残差平方和开根号，即标准差
#



#
# # 画ACF图像
#
# #注意这里面使用的ts_log_diff是经过合适阶数的差分之后的数据，上文中提到ARIMA该开源库，不支持3阶以上的#差分。所以我们需要提前将数据差分好再传入
# import sys
# from statsmodels.tsa.arima_model import ARMA
# def _proper_model(ts_log_diff, maxLag):
#     best_p = 0
#     best_q = 0
#     best_bic = sys.maxsize
#     best_model=None
#     for p in np.arange(maxLag):
#         for q in np.arange(maxLag):
#             model = ARMA(ts_log_diff, order=(p, q))
#             try:
#                 results_ARMA = model.fit(disp=-1)
#             except:
#                 continue
#             bic = results_ARMA.bic
#             print(bic, best_bic)
#             if bic < best_bic:
#                 best_p = p
#                 best_q = q
#                 best_bic = bic
#                 best_model = results_ARMA
#     return best_p,best_q,best_model
#



# _proper_model(ts_log_diff, 10) #对一阶差分求最优p和q
