# chaos-ml

There are 3 folders in this repository: chaosnet, chaosnet-noise, 2-layer-non-linear-coupling. chaosnet folder contains the code for the proposed TT-SS based classification algorithm. The details for running the code are as follows:


Steps to do: 
Step 1: Open the main file (main.ipynb or main.py)

Step 2: Assign a random state for dividing the data into training and testing (var_num = 42 (or any number))

Step 2: data_name used in the main file can be given from the following list:
"MNIST"
"IRIS"
"exoplanet"
"exoplanet-without-stemp"
"exoplanet-with-restricted-feat"
"KDDCUP"

For example: data_name = "MNIST" (MNIST dataset will be considered for analysis)

The following are the hyperparameters and details for the CHAOSNET architecture for different datasets. These parameters are found by hyperparameter tuning.

"b"  represents internal discrimnation threshold.

"q" represents the initial value of the chaotic map (or initial membrane potential).

"length" represents the number of iterations from the initial value q of the chaotic map.

"num_classes" represents the number of classes in the dataset.

"samples_per_class" represents the maximum number of data instances from each class required for training.(For examples samples_per_class = 3, will consider 3 trials of training with 1, 2 and 3 samples per class.

"check" - check = "Sk-T" will use skew tent map , ="Sk-B" will use skew binary map

"details" - details = "full" returns predicted label, mean representation vector of last trial of traning and full trials of training, accuracy, precision, recall, f1score.

details = "short" returns predicted label and mean representation vector of last trail of training.

"method" - method = "TT-SS" uses TT-SS method, method = "TT" uses firing time - (Ref: N B, Harikrishnan and Nagaraj, Nithin. "A Novel Chaos Theory Inspired Neuronal Architecture." arXiv preprint arXiv:1905.12601 (2019).) 

"var" Assign a random state for dividing the data into training and testing (var_num = 42 (or any number))

"epsilon" - represents the neighbourhood of features.


The following are the parameters used. The parameters are found in parameterfile.py 

### MNIST

--------------------------------------------

a = 0.0

c = 1.0

b = 0.3310 # Found by hyperparametertuning 

q = 0.336

length = 20000

num_classes = 10

samples_per_class = 21

check = "Sk-B"

details = "full"

var = 42

method = "TT-SS" # or TT

epsilon = 0.01

### IRIS 

--------------------------------------

a = 0.0

c = 1.0

q =0.6000 # Found by hyperparametertuning 

b = 0.9867556

length = 20000

num_classes = 3

samples_per_class = 7

check = "Sk-B"

details = "full"

var = 42

method = "TT-SS" # or TT

epsilon = 0.01

### Exoplanet 
------------------------

a = 0.0

c = 1.0

b =0.149 # Found by hyperparametertuning 

q = 0.26242424242424245

length = 20000

num_classes = 3

samples_per_class = 7

check = "Sk-T"

details = "full"

var = 42

method = "TT-SS" # or TT

epsilon = 0.01

### Exoplanet with no surface temperature parameters
---------------------------------------

a = 0.0

c = 1.0

b =0.149 # Found by hyperparametertuning 

q = 0.26242424242424245

length = 20000

num_classes = 3

samples_per_class = 7

check = "Sk-T"

details = "full"

var = 42

method = "TT-SS" # or TT

epsilon = 0.01



### Exoplanet with restricted features
--------------------------------------

a = 0.0

c = 1.0

b = 0.4760 # Found by hyperparametertuning 

q = 0.9500000000000006

length = 20000

num_classes = 3

samples_per_class = 7

check = "Sk-T"

details = "full"

var = 42

method = "TT-SS" # or TT

epsilon = 0.001



### KDDCUP

----------------------------------------

a = 0.0

c = 1.0

b = 0.3350 # Found by hyperparametertuning

q = 0.6000

length = 20000

num_classes = 9

samples_per_class = 7

check = "Sk-T"

details = "full"

var = 98 # b and q works for this particular random state

method = "TT-SS" # or TT

epsilon = 0.01


## To Run the code(for example: python main.py --data_name "exoplanet")


If the user wants to apply the method for a new data. The user have to call the data in load_data.py. Also it is important to find the q and b for the new dataset. This part is a trial and error method as of now.

