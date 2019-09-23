#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging


# In[68]:


def get_data(data_name, var_num):
    import logging
    if data_name == "MNIST":
        folder_path = "data/" + data_name 
        traindata = pd.read_csv(folder_path+"/mnist_train.csv", header = None)
        testdata = pd.read_csv(folder_path+"/mnist_test.csv", header = None)
       
        
        print(" This dataset is having separate train and test data")
        trdata = np.array(traindata)
        tedata = np.array(testdata)
        print("Number of total train data instances", trdata.shape[0])
        X_train = trdata[:,1:]
        y_train = trdata[:, 0]
        print("Number of total test data instances", tedata.shape[0])
        X_test = tedata[:,1:]
        y_test = tedata[:,0]
        
        ## Data_normalization - A Compulsory step
        
        print(" ----------Step -1---------------")
        print("                                 ")
        print(" Data normalization done ")
        X_train_norm = (X_train - np.min(X_train))/np.float(np.max(X_train) - np.min(X_train))
        X_test_norm = (X_test - np.min(X_test))/np.float(np.max(X_test) - np.min(X_test))
        
        

        try:
            assert np.min(X_train_norm) >= 0.0 and np.max(X_train_norm <= 1.0)
        except AssertionError:
            logging.error("Train Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)
            
        try:
            assert np.min(X_test_norm) >= 0.0 and np.max(X_test_norm <= 1.0)
        except AssertionError:
            logging.error("Test Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)
            
        return X_train_norm, y_train, X_test_norm, y_test
        
    elif data_name == "IRIS":
        
        folder_path = "data/" + data_name 
        data = pd.read_csv(folder_path+"/irisdata.csv", header = None)
        #testdata = pd.read_csv(folder_path+"/mnist_test.csv", header = None)
       
        
        print(" This dataset does not have separate test data")
        DATA = np.array(data)
        #tedata = np.array(testdata)
        print("Number of total data instances", DATA.shape[0])
        
        print(" ----------Step -1---------------")
        print("                                 ")
        print(" Data normalization done ")
        
        DATA_n =(DATA[:,0:-1] - np.min(DATA[:,0:-1]))/np.float(np.max(DATA[:,0:-1]) - np.min(DATA[:,0:-1]))
        
        #var_num = 42
        from sklearn.model_selection import train_test_split
        traindata, testdata, y_train, y_test = train_test_split(DATA_n, DATA[:,-1], test_size= 0.80, random_state=var_num)
   
        
        X_train_norm = np.array(traindata)
        X_test_norm = np.array(testdata)
        
        

        try:
            assert np.min(X_train_norm) >= 0 and np.max(X_train_norm <= 1)
        except AssertionError:
            logging.error("Train Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)
            
        try:
            assert np.min(X_test_norm) >= 0 and np.max(X_test_norm <= 1)
        except AssertionError:
            logging.error("Test Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)
            
        return X_train_norm, y_train, X_test_norm, y_test
    
    
    elif data_name == "exoplanet":
        
        folder_path = "data/" + data_name 
        data = pd.read_csv(folder_path+"/dataset-rocky-all-feats.csv")
        #testdata = pd.read_csv(folder_path+"/mnist_test.csv", header = None)
       
        
        print(" This dataset does not have separate test data")
        DATA = np.array(data)
        #tedata = np.array(testdata)
        print("Number of total data instances", DATA.shape[0])
        
        print(" ----------Step -1---------------")
        print("                                 ")
        print(" Data normalization done ")
        
        DATA_n =(DATA[:,1:] - np.min(DATA[:,1:]))/np.float(np.max(DATA[:,1:]) - np.min(DATA[:,1:]))
        
        #var_num = 42
        from sklearn.model_selection import train_test_split
        traindata, testdata, y_train, y_test = train_test_split(DATA_n, DATA[:,0], test_size= 0.70, random_state=var_num)
   
        
        X_train_norm = np.array(traindata)
        X_test_norm = np.array(testdata)
        
        
        try:
            assert np.min(X_train_norm) >= 0 and np.max(X_train_norm <= 1)
        except AssertionError:
            logging.error("Train Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)
            
        try:
            assert np.min(X_test_norm) >= 0 and np.max(X_test_norm <= 1)
        except AssertionError:
            logging.error("Test Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)
            
        return X_train_norm, y_train, X_test_norm, y_test
    
    elif data_name == "exoplanet-without-stemp":
        
        folder_path = "data/" + data_name 
        data = pd.read_csv(folder_path+"/dataset-rocky-no-STemp.csv")
        #testdata = pd.read_csv(folder_path+"/mnist_test.csv", header = None)
       
        
        print(" This dataset does not have separate test data")
        DATA = np.array(data)
        #tedata = np.array(testdata)
        print("Number of total data instances", DATA.shape[0])
        
        print(" ----------Step -1---------------")
        print("                                 ")
        print(" Data normalization done ")
        
        DATA_n =(DATA[:,1:] - np.min(DATA[:,1:]))/np.float(np.max(DATA[:,1:]) - np.min(DATA[:,1:]))
        
        #var_num = 42
        from sklearn.model_selection import train_test_split
        traindata, testdata, y_train, y_test = train_test_split(DATA_n, DATA[:,0], test_size= 0.70, random_state=var_num)
   
        
        X_train_norm = np.array(traindata)
        X_test_norm = np.array(testdata)
        
        

        try:
            assert np.min(X_train_norm) >= 0 and np.max(X_train_norm <= 1)
        except AssertionError:
            logging.error("Train Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)
            
        try:
            assert np.min(X_test_norm) >= 0 and np.max(X_test_norm <= 1)
        except AssertionError:
            logging.error("Test Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)
            
        return X_train_norm, y_train, X_test_norm, y_test
    
    elif data_name == "exoplanet-with-restricted-feat":
        
        folder_path = "data/" + data_name 
        data = pd.read_csv(folder_path+"/dataset-rocky-restricted-feat-set.csv")
        #testdata = pd.read_csv(folder_path+"/mnist_test.csv", header = None)
       
        
        print(" This dataset does not have separate test data")
        DATA = np.array(data)
        #tedata = np.array(testdata)
        print("Number of total data instances", DATA.shape[0])
        
        print(" ----------Step -1---------------")
        print("                                 ")
        print(" Data normalization done ")
        
        DATA_n =(DATA[:,1:] - np.min(DATA[:,1:]))/np.float(np.max(DATA[:,1:]) - np.min(DATA[:,1:]))
        
        #var_num = 42
        from sklearn.model_selection import train_test_split
        traindata, testdata, y_train, y_test = train_test_split(DATA_n, DATA[:,0], test_size= 0.70, random_state=var_num)
   
        
        X_train_norm = np.array(traindata)
        X_test_norm = np.array(testdata)
        
        

        try:
            assert np.min(X_train_norm) >= 0 and np.max(X_train_norm <= 1)
        except AssertionError:
            logging.error("Train Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)
            
        try:
            assert np.min(X_test_norm) >= 0 and np.max(X_test_norm <= 1)
        except AssertionError:
            logging.error("Test Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)
            
        return X_train_norm, y_train, X_test_norm, y_test
    
    
    elif data_name == "KDDCUP":
        
        folder_path = "data/" + data_name 
        data = pd.read_csv(folder_path+"/kdd_10_percent/kddcup.data_10_percent.csv", header = None)
        #testdata = pd.read_csv(folder_path+"/mnist_test.csv", header = None)
       
        
        print(" This dataset does not have separate test data")
        totaldata = np.array(data)
        
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        [m,n] = totaldata.shape
        DATA = np.zeros((m,n))
        for i in range(0, n):
            col = totaldata[:,i].reshape(m,1)
            le.fit(col)
            v1 = le.transform(totaldata[:,i])
            DATA[:,i] = v1 
            
        # the data belonging this class we are taking[0, 5, 9, 11, 15, 17, 18, 20, 21]
        cl_val = [0, 5, 9, 11, 15, 17, 18, 20, 21]

        tempdata = np.zeros((DATA.shape[0], DATA.shape[1] -1))
        templabel = np.zeros((DATA.shape[0], 1))
        k = -1
        count = 0
        j = 0
        for cl in cl_val:
            k = k+1
            count = count + DATA[:,-1].tolist().count(cl)
            for i in range(0, DATA.shape[0]):
                if(DATA[i,-1] == cl):
                    tempdata[j,:] = DATA[i,0 : -1]
                    templabel[j,0] = k
                    j = j + 1
        
        newdata = tempdata[0:count,:]
        newlabel = templabel[0:count,0]
        

        #tedata = np.array(testdata)
        print("Number of total data instances", totaldata.shape[0])
        print("                                 ")
        print(" ----------Step -1---------------")
        print("                                 ")
        print(" Data normalization done ")
        
        DATA_n =(newdata - np.min(newdata))/np.float(np.max(newdata) - np.min(newdata))

  


        from sklearn.model_selection import train_test_split
        traindata, testdata, y_train, y_test = train_test_split(DATA_n, newlabel, test_size= 0.70, random_state=var_num)

        
        #var_num = 42
        
   
        X_train_norm = np.array(traindata)
        X_test_norm = np.array(testdata)
        
        

        try:
            assert np.min(X_train_norm) >= 0 and np.max(X_train_norm <= 1)
        except AssertionError:
            logging.error("Train Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)
            
        try:
            assert np.min(X_test_norm) >= 0 and np.max(X_test_norm <= 1)
        except AssertionError:
            logging.error("Test Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)
            
        return X_train_norm, y_train, X_test_norm, y_test
        


# In[ ]:



