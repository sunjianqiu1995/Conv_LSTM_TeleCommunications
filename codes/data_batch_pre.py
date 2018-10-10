# -*- coding: utf8 -*-

import os
import numpy as np
import pickle
import time
import tensorflow as tf
import numpy as np
import pickle

#os.chdir('./data')
path = os.getcwd()+'/data'
print(path)

timecount = 8784
print(timecount)

# translate raw data into dictionary, saved in %d.pickle
Sin = []
Sout = []
Cin = []
Cout = []
Itra = []
for i in range(1, 10000):
    if os.path.isfile('./data/%d.pickle' % i):
        with open('./data/%d.pickle' % i, 'rb') as handle:
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
np.save('./batch_data/total.npy',teledata)
x = np.reshape(teledata,(timecount//12, 12, 100, 100, 5))

# batch_timestamp_3d: (timecount//12, 12, 100, 100, 5)
np.save('./batch_data/batch_timestamp_3d.npy',x)
# split into batch
#dat = np.load('./batch_data/total_sample.npy')


# generate batches from data loaded from dictionarys
# timestamp = 12
# batch = 12
# # ###########################################
# # # Method1: every file: (batch_size, timestamp, 100, 100, 5)
# # # for bt in range(1,timecount//(timestamp*batch)+2):
# # # 	if bt*timestamp*batch < timecount:
# # # 		x = teledata[(bt-1)*timestamp*batch:bt*timestamp*batch] # [timestamp*batch_Size, 100, 100, 5]
# # # 		x = x.reshape(batch, timestamp, 100, 100, 5)
#
# # # 		np.save('./batch_data/batch_%d.npy'%bt,x)
# # 		#print(len(teledata[(bt-1)*timestamp*batch:bt*timestamp*batch]))
# # 	# else:
# # 	# 	x = teledata[(bt-1)*timestamp*batch:timecount]
# # 	# 	x = x.reshape(batch, timestamp, 10, 10, 5)
# # 	# 	np.save('./batch_data/batch_%d.npy'%bt,x)
# # 	# 	print(len(x))
# # 	# 	print(timecount%(timestamp*batch))
#
# # #######################################
# # # Method2: every file: batch_data_random/batch_%d.npy : (timestamp, 100, 100, 5)
# # print(int(timecount//timestamp)+1)
# # for bt in range(1,int(timecount//timestamp)+1):
# # 	x = teledata[(bt-1)*timestamp:bt*timestamp] # [timestamp, 100, 100, 5]
# # 	np.save('./batch_data_random/batch_%d.npy'%bt,x)
#
# ################################################
# # # Method3: every file: ./batch_data_3.0/batch_%d.npy: (timestamp, 100, 100, 5), files number: 8784 - 12 + 1;
# teledata = np.load('./batch_data/total.npy')
# train_dir = './batch_data_3.0'
# if tf.gfile.Exists(train_dir):
#     tf.gfile.DeleteRecursively(train_dir)
# tf.gfile.MakeDirs(train_dir)
# # files: 8784 - 12 + 1; every file: ./batch_data_3.0/batch_%d.npy : (timestamp, 100, 100, 5)
# for bt in range(timecount - timestamp + 1):
#     x = teledata[(bt):(bt+timestamp)]
#     print(len(x))
#     np.save('./batch_data_3.0/batch_%d.npy'%(bt+1),x)

##############################################################
##############################################################
