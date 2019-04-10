import numpy as np
import pylab as pl

data_9952=np.abs(np.fromfile('all_data/cc1310_data_9952', dtype=np.complex64)) ##this is sensor1, naming is wrong

criteria=len(data_9952)##SENSOR1

import matplotlib.pyplot as plt
d=0
i=0
mean2=np.arange(18906.)
std2=np.arange(18906.)
var2=np.arange(18906.)
while i<=criteria:

##################################
    if data_9952[i]>=0.05:
        mean2[d]=np.mean(data_9952[i:i+10172])
        std2[d]=np.std(data_9952[i:i+10172])
        var2[d]=np.var(data_9952[i:i+10172])
        d=d+1
        i=i+10800
        #plt.figure()
        #plt.plot(mean1)
        #plt.savefig('plot' + str(n) + '.png')
##################################        
    i=i+1


dataset={'Mean2':mean2, 'Standard_Deviation2':std2, 'Variance2':var2}

import pandas as pd
df2=pd.DataFrame(data=dataset)

df2.to_csv('new_02_2019.csv')
