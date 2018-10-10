# -*- coding: utf8 -*-

import os
import numpy as np
import pickle
import time
import shutil

path = (os.getcwd()+'/data2')
try:
    os.path.exists(path)
except OSError:
    print("Path not found.")
# combine the month and path, in files_month:
# ['/home/weimin/Workspaces/Conv_LSTM/data/Nov', '/home/weimin/Workspaces/Conv_LSTM/data/Dec']
files_month = [path+'/0Nov',path+'/1Dec']
filespath = []
for month in files_month:
    files = os.listdir(month)
    for file in files:
        if not os.path.isdir(file):  # judge file
            filespath.append(month + "/" + file)
filespath.sort()
print(filespath)
print(len(filespath))

def txt2array2(filename):

    start = time.time()
    cov = {}
    for i in range(3,8):
        cov[i] = lambda s: 0 if not s else float(s)

    a = np.loadtxt(filename, converters=cov, delimiter='\t')
    print(time.time()-start)
    return a

# create 9999 files in folder: squareid
def createID():
    if os.path.isdir('squareid'):
        shutil.rmtree('squareid')
    os.mkdir('squareid')

    for i in range(9):
        file = open('./squareid/ID%d.txt'%(i+1),'a')
        file.close()


filenum=1
for filename in filespath:
    print(filename)
    a=txt2array2(filename)
    np.save('./data2/rawarray/%d.npy'%filenum,a)
    filenum+=1
    del a
#
# path = './data_sample/rawarray'
# files = os.listdir(path)
# for filename in files:
#     array = np.load(os.path.join(path,filename))
#     for i in range(len(array)):
#         int(array[i][0])
#         #np.where(int(array[i][0])==1,print(i),print('a'))
#







# path = (os.getcwd()+'/data_sample')
# try:
#     os.path.exists(path)
# except OSError:
#     print("Path not found.")
# # combine the month and path, in files_month:
# # ['/home/weimin/Workspaces/Conv_LSTM/data/Nov', '/home/weimin/Workspaces/Conv_LSTM/data/Dec']
# files_month = [path+'/0Nov',path+'/1Dec']
# filespath = []
# for month in files_month:
#     files = os.listdir(month)
#     for file in files:
#         if not os.path.isdir(file):  # judge file
#             filespath.append(month + "/" + file)
# filespath.sort()
# print(filespath)
# print(len(filespath))
#
#
#
#
#
# oneday = np.load()
# def onegrid(filename,squareid,dict):
#   '''
#   open a specific file and read the context
#   :param : filename-files need to be open
#   :param : squareid - the id of the square that is part of the Milano grid, type: numeric
#   '''
#   try:
#     with open(filename, 'r') as file:
#         for readline in file.readlines():
#           data=readline.strip('\n').split('\t')
#           if int(data[0])==squareid:
#             # print(int(data[0]))
#             timeid = int((int(data[1])-1383260400000)//600000) + 1 - 1
#             if data[3]=='':
#               data[3]=0
#             if data[4]=='':
#               data[4]=0
#             if data[5]=='':
#               data[5]=0
#             if data[6]=='':
#               data[6]=0
#             if data[7]=='':
#               data[7]=0
#
#             dict['SMSin'][timeid] += float(data[3])
#             dict['SMSout'][timeid] += float(data[4])
#             dict['Callin'][timeid] += float(data[5])
#             dict['Callout'][timeid] += float(data[6])
#             dict['InternetTra'][timeid] += float(data[7])
#     file.close()
#   except OSError:
#     # 'File not found' error message.
#     print("File not found.")
#
#   return dict
#
# timestart=time.time()
# for squareid in range(1):
#   dict = {}
#   dict['SMSin'] = np.zeros(6*24*61)
#   dict['SMSout'] = np.zeros(6*24*61)
#   dict['Callin'] = np.zeros(6*24*61)
#   dict['Callout'] = np.zeros(6*24*61)
#   dict['InternetTra'] = np.zeros(6*24*61)
#   for filename in filespath:
#     print(filename)
#     onegrid(filename,squareid+1,dict) # onegrid
#
#   with open('%d.pickle'%(squareid+1), 'wb') as handle:
#     pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#   del dict
#
# print(time.time()-timestart)
#
