#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from argparse import ArgumentParser
# ### Load dataset

# In[2]:

######
def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--data_name', type=str,
                        dest='data_name', help='Name of dataset,(MNIST, IRIS, exoplanet, exoplanet-without-stemp, KDDCUP, exoplanet-with-restricted-feat, ',
                        metavar='data_name', default="IRIS")
    return parser
	

def main():
	parser = build_parser()
	options = parser.parse_args()
	######
	#print(parser)
	from load_data import get_data
	from parameterfile import (parameter_file, parameter_file_non_linear_coupling)

	var_num = 42
	data_name = options.data_name
	X_train, y_train, X_test, y_test = get_data(data_name, var_num)


	try:
	    assert np.min(X_train) >= 0 and np.max(X_train <= 1)
	except AssertionError:
	    logging.error("Train Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)

	try:
	    assert np.min(X_test) >= 0.0 and np.max(X_test <= 1.0)
	except AssertionError:
	    logging.error("Test Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)




	# In[3]:


	### The following are parameters are different for different datasets


	# In[4]:


	### IRIS 
        q2 = 0.56
        coeff0 = 0.999/2.0
        coeff1 = 0.999/2.0
        coeff2 = 1 - (coeff0 + coeff1)
      
	#--------------------------------------
        a, b, c, q, length, num_classes, samples_per_class, check, details, var, method, epsilon = parameter_file_non_linear_coupling(data_name)

	
	# In[5]:


	from Codes import (skew_tent, iterations, firingtime_calculation, probability_calculation, class_avg_distance, cosine_similar_measure, class_wise_data, test_split_generator,chaos_method, CHAOSNET)
	from layer_2_non_linear_coupling import (chaos_second_layer, CHAOSNETWORKS)
	y_pred_val, avg_class_dist_1, ACC, PRECISION, RECALL, F1SCORE, avg_total_class_prob, test_proba = CHAOSNETWORKS(X_train, y_train, X_test, y_test, num_classes, samples_per_class, check, q, a, b, c, length, var, details, method, epsilon, data_name, q2, coeff0, coeff1, coeff2)
	# In[6]:

#epsilon = 0.01
if __name__ == '__main__':
    main()



# In[ ]:




