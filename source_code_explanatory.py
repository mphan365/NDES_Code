import random
import simpy
import functools
from functools import partial, wraps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
from scipy.stats import norm

strokes_per_year = 14239
start_date = "01/01/2019"
simulation_runtime = 5
burn_in = 0.5
proportion_to_receive_ECR = [0.03, 0.05, 0.1, 0.15]
proportion_nearest_rural = 0.95
proportion_nearest_peripheral = 0.95
proportion_nearest_central = 0.95
proportion_transfer = 0.01
proportion_transferred_requiring_CT = 0.3

# initial ambulance call time distribution
amb_call_time_q1 = 1
amb_call_time_median = 13
amb_call_time_q3 = 29

# initial ambulance arrival time distribution
amb_arrive_time_q1 = 8
amb_arrive_time_median = 11
amb_arrive_time_q3 = 14

# initial ambulance assessment time distribution
amb_assess_time_q1 = 10
amb_assess_time_median = 14
amb_assess_time_q3 = 20

# generic hospital admission time distribution
hosp_admit_time_q1 = 8
hosp_admit_time_median = 15
hosp_admit_time_q3 = 23.5

# csc hospital admission time distribution
cschosp_admit_time_median = 10
cschosp_admit_time_sd = 2.5

# csc hospital CT scan time distribution
csc_ct_scan_time_median = 15
csc_ct_scan_time_sd = 2.5

# generic CT scan time distribution
ct_scan_time_q1 = 44
ct_scan_time_median =59.5
ct_scan_time_q3 = 83

# csc hospital door to puncture time distribution
csc_door_to_puncture_median = 15
csc_door_to_puncture_sd = 2.5

# intra hospital AV time to arrive for transfer time distribution
transfer_amb_arrive_time_q1 = 7
transfer_amb_arrive_time_median = 10
transfer_amb_arrive_time_q3 = 14

# intra hospital AV time to depart for transfer time distribution
transfer_amb_depart_time_q1 = 14
transfer_amb_depart_time_median = 17
transfer_amb_depart_time_q3 = 21

# admission for transferred codestroke patient to CSC distribution
csc_receive_ecr_transfer_time_q1 = 15
csc_receive_ecr_transfer_time_median = 20
csc_receive_ecr_transfer_time_q3 = 35.75

# time for angio reperfusion distribution
angio_reperfusion_time_q1 = 23.75
angio_reperfusion_time_median = 40
angio_reperfusion_time_q3 = 73.5

# angio cut off time variables
angio_cutoff_time_hours = 6
anigo_optimal_time_hours = 2

# acute bed stay time distribution
acutebed_stay_median_hours = 48
acute_bed_stay_sd_hours = 6

# Calculations
mins_per_one_stroke = (365.25 * 24 * 60) / strokes_per_year
max_time = 365.25 * (simulation_runtime + burn_in) * 24 * 60
burn_in_mins = 365.25 * burn_in * 24 * 60
start_date_unix = time.mktime(
    datetime.datetime.strptime(start_date, "%d/%m/%Y").timetuple()
)
angio_cutoff_time_mins = angio_cutoff_time_hours*60
anigo_optimal_time_mins = anigo_optimal_time_hours*60

acutebed_stay_median_mins = acutebed_stay_median_hours*60
acute_bed_stay_sd_mins = acute_bed_stay_sd_hours *60

# ##Functions
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


def get_stroke_incident_time(mins_per_one_stroke=mins_per_one_stroke):
    time = 0
    while time < max_time:
        time_until_next_stroke = np.random.geometric(1/mins_per_one_stroke)
        if time + time_until_next_stroke > max_time:
            break
        yield (time + time_until_next_stroke)
        time += time_until_next_stroke

def get_patient(
    test_time,
    travel_time_data=travel_time_data,
    sed_id_list=sed_id_list,
    weighted_proportions=weighted_proportions,
):
    # Returns tuple of form (time_taken, destination, location, sed, sed_type)
    # Note that time taken is in SECONDS and converted to the nearest minute in this function
    # Of the location tuple, index 0 is latitude and index 1 is longitude
    sed_id = np.random.choice(sed_id_list, p=weighted_proportions)
    data = travel_time_data[travel_time_data.sed_id == sed_id]
    (time_taken, destination, location, sed, sed_type) = travel_times(data, test_time)
    return (time_taken // 60, destination, location[0], location[1], sed, sed_type)

# geometric distribution extrapolation from q1 q2 q3
def get_geometric(q1,q2,q3):
    n=67
    mean = (q1+q2+q3)/3
    var = (q3-q1)/(2*norm.ppf((0.75*n-0.125)/(n+0.25)))
    x=(2*var)/(np.sqrt(1+4*var)-1)
    return int(mean-x+np.random.geometric(1/x))

def get_time_for_call():
    return(get_geometric(amb_call_time_q1,
                        amb_call_time_median,
                        amb_call_time_q3))

def get_time_for_ambulance_arrival():
    return(get_geometric(amb_arrive_time_q1,
                        amb_arrive_time_median,
                        amb_arrive_time_q3))

def get_time_for_ambulance_assesment():
    return(get_geometric(amb_assess_time_q1,
                        amb_assess_time_median,
                        amb_assess_time_q3))

def get_time_for_admission(location_type):
    if location_type == "central":
        return max(1, int(np.random.normal(cschosp_admit_time_median,
                                            cschosp_admit_time_sd)))
    else:
        return get_geometric(hosp_admit_time_q1,
                            hosp_admit_time_median,
                            hosp_admit_time_q3)

def get_time_for_ct_scan(location_type):
    if location_type == "central":
        return max(1, int(np.random.normal(csc_ct_scan_time_median,
                                            csc_ct_scan_time_sd)))
    else:
        return get_geometric(ct_scan_time_q1,
                            ct_scan_time_median,
                            ct_scan_time_q3)

def get_time_for_csc_door_to_puncture():
    return max(1, int(np.random.normal(csc_door_to_puncture_median,
                                        csc_door_to_puncture_sd)))

def get_time_for_ambulance_arrival_for_transfer():
    return get_geometric(transfer_amb_arrive_time_q1,
                        transfer_amb_arrive_time_median,
                        transfer_amb_arrive_time_q3)

def get_time_for_ambulance_departure_for_transfer():
    return get_geometric(transfer_amb_arrive_time_q1,
                        transfer_amb_depart_time_median,
                        transfer_amb_depart_time_q3)

def get_time_for_CSC_receive_ECR_transfer():
    return get_geometric(csc_receive_ecr_transfer_time_q1,
                        csc_receive_ecr_transfer_time_median,
                        csc_receive_ecr_transfer_time_q3)

def get_transfer_decision(proportion_transfer, is_ECR):
    if is_ECR == 1:
        return 1
    else:
        return np.random.binomial(n=1, p=proportion_transfer)

def get_transfer_CT_decision():
    return np.random.binomial(n=1, p=proportion_transferred_requiring_CT)

def get_ECR_status():
    return np.random.binomial(n=1, p=proportion_to_receive_ECR)

def get_angio_decision(time_for_treatment_wait):
    if time_for_treatment_wait <= angio_cutoff_time_mins:
        return 1
    else:
        return 0

def get_angio_category(time_for_treatment_wait):
    if time_for_treatment_wait < anigo_optimal_time_mins:
        return 1
    elif anigo_optimal_time_mins <= time_for_treatment_wait < angio_cutoff_time_mins:
        return 2
    else:
        return 3

def get_angio_time():
    return get_geometric(angio_reperfusion_time_q1,
                        angio_reperfusion_time_median,
                        angio_reperfusion_time_q3)
