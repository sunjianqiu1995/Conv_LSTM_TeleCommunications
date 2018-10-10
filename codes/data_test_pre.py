# -*- coding: utf8 -*-

import os
import numpy as np
import pickle
import time

path = './data_test'
try:
	os.path.exists(path)
except OSError:
	print("Path not found.")
filename =path + "/sms-call-internet-mi-2014-01-01.txt"


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
            timeid = (int((int(data[1])-1383260400000)//600000) + 1 - 1) - 8784
            
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

# Pre Test data
timecount = 6*24
print(timecount)

Sin = []
Sout = []
Cin = []
Cout = []
Itra = []

for squareid in range(1, 10000):
  dict = {}
  dict['SMSin'] = np.zeros(timecount)
  dict['SMSout'] = np.zeros(timecount)
  dict['Callin'] = np.zeros(timecount)
  dict['Callout'] = np.zeros(timecount)
  dict['InternetTra'] = np.zeros(timecount)

  onegrid(filename,squareid,dict) # onegrid
  if squareid % 1000 == 0:
      print(squareid)

  smsin = dict['SMSin']  # array
  smsout = dict['SMSout']
  callin = dict['Callin']
  callout = dict['Callout']
  internettra = dict['InternetTra']

  Sin.append(smsin)  # 9999 * timecount
  Sout.append(smsout)  # 9999 * timecount
  Cin.append(callin)  # 9999 * timecount
  Cout.append(callout)  # 9999 * timecount
  Itra.append(internettra)  # 9999 * timecount

  del dict

extra = np.zeros(timecount)
Sin.append(extra)  # 10000 * timecount
Sout.append(extra)  # 10000 * timecount
Cin.append(extra)  # 10000 * timecount
Cout.append(extra)  # 10000 * timecount
Itra.append(extra)  # 10000 * timecount
print(type(Sin))
# list to array
Sin = np.array(Sin)
Sout = np.array(Sout)
Cin = np.array(Cin)
Cout = np.array(Cout)
Itra = np.array((Itra))

Sin = Sin.T  # timecount * 10000
print(Sin.shape)
print(type(Sin))
Sout = Sout.T  # timecount * 10000
Cin = Cin.T  # timecount * 10000
Cout = Cout.T  # timecount * 10000
Itra = Itra.T  # timecount * 10000

teledata = []  # combine all the time
for times in range(timecount):
    telefea = [] # combine feature
    telefea.append(Sin[times])
    telefea.append(Sout[times])
    telefea.append(Cin[times])
    telefea.append(Cout[times])
    telefea.append(Itra[times])
    telefea = np.array(telefea)
    # telefea.shape: array [5 * 10000]
    telefea = telefea.T # [10000, 5]
    telefea = np.reshape(telefea, (100, 100, 5)) # 100*100*5

    teledata.append(telefea)

teledata = np.array(teledata) # timecount * 100*100*5
print(teledata.shape)
# total: timecount8784 * 100*100*5
np.save('./data_test/test_total.npy',teledata)
x = np.reshape(teledata,(timecount//12, 12, 100, 100, 5))

# test_one_batch: (timecount//12, 12, 100, 100, 5)
np.save('./data_test/test_one_batch.npy',x)

