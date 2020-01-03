### Generating ambulance travel time function
# This function generates a random travel time from a stroke incident area to the hospital
# Here we assume all areas that we 'sampled' have equal probably of occurring

# INPUTS:
## data: subsetted data of 50 samples (10 points * 5 times) for a single SED.
## test_time: Unix time for incident.
## travel_time_data: a pandas dataframe containing all travel times in format specified in notebooks.
## sed_population_data: a pandas dataframe containing a field, "Tot_P_P" corresponding to the total population of each SED.
## var: the variation in travel times (default +-10%)
## proportion_second_closest: the probability that ambulances choose the second closest hospital, rather than the closest

## none for now
# OUTPUTS:

## 1. Random time 
## 2. Destination 
## 3. Origin Point
## 4. 

import pandas as pd
import numpy as np
import random
import time
import datetime 

def travel_times(data, test_time, var = 0.1,proportion_second_closest = 0.05):
    # Use the dataset provided - the edited pickle file has only relevant columns
    # To view the whole dataset, use 'travel_times.pkl'
    
    success = False
    
    while not success: # keep checking in case of missing data
        try:
            random_choice = random.randint(0,9)
            random_date = random.randint(1,100000)

            ## Plus minus 10%
            rand_variation = np.random.uniform()*2*var + (1-var)

            test_datetime = datetime.datetime.strptime(test_time,'%H:%M') ##Get it in time format

            random_hosp = 1 + np.random.binomial(1,proportion_second_closest)
            time_taken = 0

            if test_datetime >= datetime.datetime(1900,1,1,22,30) or test_datetime < datetime.datetime(1900,1,1,4,30):
                if random_hosp == 1:
                    time_taken = data.iloc[random_choice*5+4,3]*rand_variation
                else:
                    time_taken = data.iloc[random_choice*5+4,10]*rand_variation

            elif test_datetime >= datetime.datetime(1900,1,1,4,30) or test_datetime < datetime.datetime(1900,1,1,11,00):
                if random_hosp == 1:
                    time_taken = data.iloc[random_choice*5,3]*rand_variation
                else:
                    time_taken = data.iloc[random_choice*5,10]*rand_variation

            elif test_datetime >= datetime.datetime(1900,1,1,11,00) or test_datetime < datetime.datetime(1900,1,1,15,00):
                if random_hosp == 1:
                    time_taken = data.iloc[random_choice*5+1,3]*rand_variation
                else:
                    time_taken = data.iloc[random_choice*5+1,10]*rand_variation

            elif test_datetime >= datetime.datetime(1900,1,1,15,00) or test_datetime < datetime.datetime(1900,1,1,19,00):
                if random_hosp == 1:
                    time_taken = data.iloc[random_choice*5+2,3]*rand_variation
                else:
                    time_taken = data.iloc[random_choice*5+2,10]*rand_variation

            else:
                if random_hosp == 1:
                    time_taken = data.iloc[random_choice*5+3,3]*rand_variation
                else:
                    time_taken = data.iloc[random_choice*5+3,10]*rand_variation

            if random_hosp == 1:
                destination = data.closest_destination.iloc[random_choice*5] 
            else:
                destination = data.second_closest_destination.iloc[random_choice*5]  

            time_taken = int(time_taken)
            
            if type(destination) == str:
                success = True
            else:
                continue
                
        except ValueError: # missing data
            continue
        
    location = (data.origin_latitude.iloc[random_choice*5],data.origin_longitude.iloc[random_choice*5])
    sed = data.sed_name.iloc[random_choice*5]
    sed_type = data.type.iloc[random_choice*5]
    
    assert (type(destination) == str)
    
    return time_taken, destination, location, sed, sed_type
