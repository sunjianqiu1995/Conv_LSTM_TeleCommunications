# -*- coding: utf8 -*-

import os
import numpy as np
import pickle
import time

os.chdir('./data')
path = os.getcwd()
try:
	os.path.exists(path)
except OSError:
	print("Path not found.")
files= os.listdir(path) #get all the files names
# combine the month and path, in files_month:
# ['/home/weimin/Workspaces/Conv_LSTM/data/Nov', '/home/weimin/Workspaces/Conv_LSTM/data/Dec']
files_month=[]
for file in files: #遍历文件夹  
     if os.path.isdir(file): #judge folder
          files_month.append(path+'/'+file)
print(files_month)
filespath = []
for month in files_month:
  files= os.listdir(month)
  for file in files: 
    if not os.path.isdir(file): #judge file      

      # file = open(month+"/"+file); #
      # for readline in file.readlines():
      #   readline.strip('\n').split('\t')
      # file.close()
      filespath.append(month+"/"+file)
filespath.sort()
print(filespath)
print(len(filespath))


def onegrid(filename,squareid,dict):
  '''
  open a specific file and read the context
  :param : filename-files need to be open
  :param : squareid - the id of the square that is part of the Milano grid, type: numeric
  '''
  try:
    with open(filename, 'r') as file:
        for readline in file.readlines():
          data=readline.strip('\n').split('\t')
          if int(data[0])==squareid:
            # print(int(data[0]))
            timeid = int((int(data[1])-1383260400000)//600000) + 1 - 1
            
            if data[3]=='':
              data[3]=0
            if data[4]=='':
              data[4]=0            
            if data[5]=='':
              data[5]=0
            if data[6]=='':
              data[6]=0
            if data[7]=='':
              data[7]=0

            dict['SMSin'][timeid] += float(data[3])
            dict['SMSout'][timeid] += float(data[4])
            dict['Callin'][timeid] += float(data[5])
            dict['Callout'][timeid] += float(data[6])
            dict['InternetTra'][timeid] += float(data[7])
    file.close()
  except OSError:
    # 'File not found' error message.
    print("File not found.")

  return dict

timestart=time.time()
for squareid in range(533,1000):
  dict = {}
  dict['SMSin'] = np.zeros(6*24*61)
  dict['SMSout'] = np.zeros(6*24*61)
  dict['Callin'] = np.zeros(6*24*61)
  dict['Callout'] = np.zeros(6*24*61)
  dict['InternetTra'] = np.zeros(6*24*61)
  for filename in filespath:
    print(filename)
    onegrid(filename,squareid+1,dict) # onegrid

  with open('%d.pickle'%(squareid+1), 'wb') as handle:
    pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
  del dict

print(time.time()-timestart)
# with open('1.pickle', 'rb') as handle:
#     dict = pickle.load(handle)



'''
# open a specific file and read the context
os.chdir('./data')
try:
    with open('sms-call-internet-mi-2013-11-01.txt', 'r') as file:
        for readline in file.readlines():
        	data.append(readline.strip('\n').split('\t'))
    file.close()
except OSError:
    # 'File not found' error message.
    print("File not found.")
'''




