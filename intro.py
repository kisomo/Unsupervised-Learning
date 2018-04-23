import numpy as np
import pandas as pd


#+++++++++++++++++++++++++++ Echo State NetWork (Reservoir Computing) ESN ++++++++++++++++++++++++++++++++++++++++++++++++
#https://www.quantamagazine.org/machine-learnings-amazing-ability-to-predict-chaos-20180418/

#http://strategy.doubledoji.com/how-to-use-echo-state-network-in-forex-trading/


#import ESN module
from pyESN import ESN
#import matplotlib
import matplotlib.pyplot as plt

#read the open, high, low and closing price from the csv files
o, h, l, c=np.loadtxt("E:/MarketData/GBPUSD30.csv", delimiter=',', 
                      usecols=(2,3,4,5), unpack=True)

##build an Echo State Network
esn = ESN(n_inputs = 1,
          n_outputs = 1,
          n_reservoir = 500,
          spectral_radius = 1.5,
          random_state=42)
#choose the training set

trainlen = 500
future = 20
#start training the model
pred_training = esn.fit(np.ones(trainlen),c[len(c)-trainlen: len(c)])
#make the predictions
prediction = esn.predict(np.ones(future))
print("test error: \n"+str(np.sqrt(np.mean((prediction.flatten() \
- c[trainlen:trainlen+future])**2))))

#print the predicted values of the closing price
prediction
#plot the predictions
plt.figure(figsize=(11,1.5))
plt.plot(range(0,trainlen),c[len(c)-trainlen:len(c)],'k',label="target system")
plt.plot(range(trainlen,trainlen+future),prediction,'r', label="free running ESN")
lo,hi = plt.ylim()


plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
plt.legend(loc=(0.61,1.1),fontsize='x-small')









#++++++++++++++++++++++++++++++++++++++ Self Organizing Maps (SOM) +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



#+++++++++++++++++++++++++++++++++++++++++ Boltzman Machines (BM) ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#++++++++++++++++++++++++++++++++++++++ Restricted Boltzman Machines (RBM) +++++++++++++++++++++++++++++++++++++++++++++++++



#++++++++++++++++++++++++++++++++++++ Deep Belief Networks (DBN) ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



#+++++++++++++++++++++++++++++++++++++++++++++++++++++ K-MEANS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++





#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ Mixture Models ++++++++++++++++++++++++++++++++++++++++++++++++++

