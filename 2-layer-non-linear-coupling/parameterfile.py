def parameter_file(data_name):
    
    import logging
    if data_name == "MNIST":
        
        
                ### MNIST
        #--------------------------------------------
        a = 0.0
        c = 1.0
        b = 0.331 # Found by hyperparametertuning 
        q = 0.336
        length = 20000
        num_classes = 10
        samples_per_class = 21
        check = "Sk-B"
        details = "full"
        var = 42
        method = "TT-SS" # or TT
        epsilon = 0.01
        
        return a, b, c, q, length, num_classes, samples_per_class, check, details, var, method, epsilon
    
    elif data_name == "IRIS":

        ### IRIS 
        #--------------------------------------
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
        
        return a, b, c, q, length, num_classes, samples_per_class, check, details, var, method, epsilon
    
    elif data_name == "exoplanet":
    

### Exoplanet 
        #------------------------

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
        
        return a, b, c, q, length, num_classes, samples_per_class, check, details, var, method, epsilon
    
    elif data_name == "exoplanet-without-stemp":
        

### Exoplanet with no surface temperature parameters
#---------------------------------------

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
        
        return a, b, c, q, length, num_classes, samples_per_class, check, details, var, method, epsilon
    
    elif data_name == "exoplanet-with-restricted-feat":

        ### Exoplanet with restricted features
        #--------------------------------------
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

        return a, b, c, q, length, num_classes, samples_per_class, check, details, var, method, epsilon
    
    elif data_name == "KDDCUP":
    


        ### KDDCUP
        #----------------------------------------
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
        
        return a, b, c, q, length, num_classes, samples_per_class, check, details, var, method, epsilon


def parameter_file_non_linear_coupling(data_name):
    
    import logging
    if data_name == "MNIST":
        
        
                ### MNIST
        #--------------------------------------------
        a = 0.0
        c = 1.0
        b = 0.331 # Found by hyperparametertuning 
        q = 0.336
        length = 20000
        num_classes = 10
        samples_per_class = 21
        check = "Sk-B"
        details = "full"
        var = 42
        method = "TT-SS" # or TT
        epsilon = 0.01
        
        return a, b, c, q, length, num_classes, samples_per_class, check, details, var, method, epsilon
    
    elif data_name == "IRIS":

        ### IRIS 
        #--------------------------------------
        a = 0.0
        c = 1.0
        q =0.91 # Found by hyperparametertuning 
        b = 0.9867556
        length = 20000
        num_classes = 3
        samples_per_class = 7
        check = "Sk-B"
        details = "full"
        var = 42
        method = "TT-SS" # or TT
        epsilon = 0.01
        
        return a, b, c, q, length, num_classes, samples_per_class, check, details, var, method, epsilon
    
    elif data_name == "exoplanet":
    

### Exoplanet 
        #------------------------

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
        
        return a, b, c, q, length, num_classes, samples_per_class, check, details, var, method, epsilon
    
    elif data_name == "exoplanet-without-stemp":
        

### Exoplanet with no surface temperature parameters
#---------------------------------------

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
        
        return a, b, c, q, length, num_classes, samples_per_class, check, details, var, method, epsilon
    
    elif data_name == "exoplanet-with-restricted-feat":

        ### Exoplanet with restricted features
        #--------------------------------------
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

        return a, b, c, q, length, num_classes, samples_per_class, check, details, var, method, epsilon
    
    elif data_name == "KDDCUP":
    


        ### KDDCUP
        #----------------------------------------
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
        
        return a, b, c, q, length, num_classes, samples_per_class, check, details, var, method, epsilon
        
        

