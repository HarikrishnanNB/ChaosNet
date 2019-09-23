#!/usr/bin/env python
# coding: utf-8

# ### Method - I
# Fixing the SNR for both TT-SS method and DL

# In[371]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_data import get_data
from parameterfile import parameter_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
import logging


# In[372]:


### power of signal is computed
def power(x):
    return np.sum(np.multiply(x,x))


# In[373]:


### SNR ratio is computed
### Output: SNR in dB and SNR is returned
def snr(Psig, Pnoi):
    SNR = Psig.astype(np.float)/Pnoi
    SNR_dB = 10 * np.log10(SNR)
    return SNR_dB, SNR


# In[374]:


#### Fixing the SNR in dB and computing the noise power
#### VAL_dB = SNR_in_dB
def noise_power_for_fixed_SNR_dB(VAL_dB, Ps):
    Pn  = np.float(Ps)/np.float(10.0 ** np.float(VAL_dB/10.0))
    return Pn


# In[375]:


def find_noise_for_SNR(length1, Pn, Ps):
    ### Gaussain distribution
    noise = np.zeros((1, length1))
    noise[0, 0:length1-1] =  0.001 * (Pn/Ps) * np.random.randn(1,length1-1)
    noise[0, -1] = np.sqrt(Pn - power(noise[0, 0:length1-1]))
    print("power of noise = ", power(noise))
    return noise


# In[376]:


import matplotlib.pyplot as plt
from Codes import (skew_tent, iterations, firingtime_calculation, probability_calculation, class_avg_distance, cosine_similar_measure, class_wise_data, test_split_generator,chaos_method, CHAOSNET)


# In[378]:


def noise_analysis(data_name, method, var_num , SNR_in_dB):

    import os
    folder_path = os.getcwd() + "/"+ "chaosnet-results/" + data_name + "/" + method + "/"+ data_name + "-test_proba.csv"
    print(" Accessing Test Proba " , folder_path)
    test_proba = np.array(pd.read_csv(folder_path, header = None))
    try:
        assert np.min(test_proba) >= 0 and np.max(test_proba <= 1)
    except AssertionError:
        logging.error("ERROR-Check test_proba.csv and check the codes properly", exc_info=True)


    X_train, y_train, X_test, y_test = get_data(data_name, var_num)

    a, b, c, q, length, num_classes, samples_per_class, check, details, var, method, epsilon = parameter_file(data_name)
    

        # define the name of the directory to be created
    path = os.getcwd()
    resultpath = path + '/noise-results-fixed-snr/'  + data_name + '/'+ method

    ACC = np.zeros((samples_per_class, 1))
    PRECISION = np.zeros((samples_per_class, 1))
    RECALL = np.zeros((samples_per_class, 1))
    F1SCORE = np.zeros((samples_per_class, 1))
    
    for i in range(0, samples_per_class):
        print("Number of samples_per_class = ", i+1)
        rep_path = os.getcwd() + "/"+ "chaosnet-results/" + data_name + "/" + method +"/"+  data_name + "-" + method + "-representation_vector_last" + str(i) + ".csv"
        representation_vectors = np.array(pd.read_csv(rep_path, header = None))

        Ps = power(representation_vectors)
        length1 = representation_vectors.shape[0] * representation_vectors.shape[1]
        Pn = noise_power_for_fixed_SNR_dB(SNR_in_dB, Ps)
        noise = find_noise_for_SNR(length1, Pn, Ps)


        print("Adding noise")
        noise_rep_vectors = representation_vectors + np.reshape(noise, (representation_vectors.shape[0] , representation_vectors.shape[1]))
        print(" Calculated SNR : = ", snr(Ps, power(noise))[0], " in  dB")
        
        try:
            assert np.abs(np.sum(representation_vectors - noise_rep_vectors)) > 0
        except AssertionError:
            logging.error("Noise is NOT added", exc_info=True)

        print(" ")
        print("-----------------------------------------------------")
        print(" Prediction for Test Data")                    
        #test_firingtime = probability_calculation(X_test, timeseries, b)
        y_pred_val = cosine_similar_measure(test_proba, y_test, a, b, c, noise_rep_vectors)

        accuracy = accuracy_score(y_test, y_pred_val)*100
        recall = recall_score(y_test, y_pred_val , average="macro")
        precision = precision_score(y_test, y_pred_val , average="macro")
        f1 = f1_score(y_test, y_pred_val, average="macro")
        ACC[i,0] = accuracy
        PRECISION[i, 0] = precision
        RECALL[i, 0] = recall
        F1SCORE[i,0] = f1 

    print("")
    print("Saving Results")


        # define the name of the directory to be created


        # define the access rights
        #access_rights = 0o755

    try:
        os.makedirs(resultpath)
    except OSError:
        print ("Creation of the result directory %s failed" % resultpath)
    else:
        print ("Successfully created the result directory %s" % resultpath)

    np.savetxt(resultpath +"/"+ data_name +"-"+ method +"-ACC.csv",ACC, delimiter= ',', fmt='%1.3f')
    np.savetxt(resultpath + "/" + data_name + "-"+ method + "-PRECISION.csv", PRECISION, delimiter= ',', fmt='%1.3f')
    np.savetxt(resultpath + "/" + data_name + "-"+ method + "-RECALL.csv", RECALL, delimiter= ',', fmt='%1.3f')
    np.savetxt(resultpath + "/" + data_name + "-"+ method + "-F1SCORE.csv", F1SCORE, delimiter= ',', fmt='%1.3f') 
    np.savetxt(resultpath + "/" + data_name + "-"+ method + "-SNR.txt", [SNR_in_dB], delimiter= ',', fmt='%1.3f') 
#np.savetxt(resultpath + "/" + data_name + "-"+ method + "-representation_vector_last.csv", avg_class_dist_1, fmt='%1.8f') 
#     with file(resultpath + "/" + data_name + "-"+ method +"-tot_representation_vector_last.txt", "w") as outfile:
# 	    for slice_2d in avg_total_class_prob:
# 		np.savetxt(outfile, slice_2d, fmt='%1.8f')

    print("")
    print("Saving Graphs")
    import os

# define the name of the directory to be created
    path = os.getcwd()
    graphpath = path + '/noise-graphs-fixed-snr/'  + data_name + '/'+ method

# define the name of the directory to be created


# define the access rights
#access_rights = 0o755

    try:
	os.makedirs(graphpath)
    except OSError:
        print ("Creation of the result directory %s failed" % graphpath)
    else:
	print ("Successfully created the result directory %s" % graphpath)

    import matplotlib.pyplot as plt
    n = np.arange(1,samples_per_class + 1, 1)
    plt.figure(figsize=(10,10))
    plt.plot(n, ACC,linewidth = 2.0)# initial value is 0.0005
    plt.xlabel('Number of training samples per class', fontsize = 20)
    plt.ylabel(' Accuracy', fontsize = 20)
    plt.xticks(fontsize=12)
#plt.xlim([0.0, 1.0])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)


    plt.savefig(graphpath +"/"+ data_name +"-"+ method +"-ACCURACY.eps", format='eps', dpi=700)

    plt.savefig(graphpath +"/"+ data_name +"-"+ method +"-ACCURACY.jpg", format='jpg', dpi=700)

    plt.figure(figsize=(10,10))
    plt.xlabel('Number of training samples per class', fontsize = 20)
    plt.ylabel(' F1-score', fontsize = 20)
    plt.xticks(fontsize=12)
#plt.xlim([0.0, 1.0])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.plot(n, F1SCORE,linewidth = 2.0)# initial value is 0.0005


    plt.savefig(graphpath +"/"+ data_name +"-"+ method +"-F1SCORE.eps", format='eps', dpi=700)

    plt.savefig(graphpath +"/"+ data_name +"-"+ method +"-F1SCORE.jpg", format='jpg', dpi=700)

#plt.title('MNIST ', fontsize = 20)
#plt.savefig('/home/nithin/HARIKRISHNAN NB/HKNB/2019/LAPTOP_CONTENTS/Chaos/probability_based_classification/results/mnist/accuracy_mnist_p.png', format='png', dpi=500)
    plt.show()
        
    return ACC, PRECISION, RECALL, F1SCORE


# In[ ]:




