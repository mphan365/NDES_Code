# ## Imports
import random
import simpy
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial, wraps
import time
import datetime
from modules.travel_times import travel_times

# ## Variables

# number of strokes per year
strokes_per_year = 14239

# average time (in minutes) until a stroke happens
mins_per_one_stroke = (365.25 * 24 * 60) / strokes_per_year

## assuming max time is 6 months
max_time = 365.25 * 12 * 24 * 60

# probability of going to nearest hospital
proportion_nearest_rural = 0.95
proportion_nearest_peripheral = 0.95
proportion_nearest_central = 0.95

# probability of transfer to CSC for treatment
proportion_transfer_from_rural = 0
proportion_transfer_from_peripheral = 0

# proportion to receive ECR
proportion_to_receive_ECR = 0.1

start_date = "01/01/2019"  # initial start date of simulation
start_date_unix = time.mktime(
    datetime.datetime.strptime(start_date, "%d/%m/%Y").timetuple()
)


# ## Submission Data

# The data below is Victoria-specific.

# import hospital name and capacity
hospital_data_columns = ["name", "capacity"]

rural_hospital_data = pd.read_csv(
    "data/internal/2hosp/rhosp.csv", names=hospital_data_columns
)
rural_hospital_data["hospital_type"] = "rural"

peripheral_hospital_data = pd.read_csv(
    "data/internal/2hosp/pschosp.csv", names=hospital_data_columns
)
peripheral_hospital_data["hospital_type"] = "peripheral"

central_hospital_data = pd.read_csv(
    "data/internal/2hosp/cschosp.csv", names=hospital_data_columns
)
central_hospital_data["hospital_type"] = "central"


hospital_data = pd.concat(
    [rural_hospital_data, peripheral_hospital_data, central_hospital_data]
).set_index("name")

# import transfer data
transfer_to_monash_data = (
    pd.read_csv("data/internal/2hosp/t2monash.csv").set_index("Hospitals").T
)
transfer_to_rmh_data = pd.read_csv("data/internal/2hosp/t2rmh.csv").set_index("Hospitals").T
central_hospitals = ["Monash Health", "Royal Melbourne Hospital"]
transfer_to_central_data = [
    transfer_to_monash_data,
    transfer_to_rmh_data,
]

# travel time data
travel_time_data = pd.read_pickle("data/preprocessed/travel_times_updatehospitals.pkl")
population_data = pd.read_csv("data/external/2016Census_G01_VIC_SED.csv")
population_data = population_data[:-2]  # exclude two non-location SEDs

total_population = sum(population_data["Tot_P_P"])
weighted_proportions = population_data["Tot_P_P"] / total_population
sed_id_list = population_data["SED_CODE_2016"]


# ## Functions

## Value generation functions during simulation

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


# Pre-hospital Functions
def get_geometric(median, IQR):
    x=(2*IQR)/(np.sqrt(1+4*IQR)-1)
    return int(median-x+np.random.geometric(1/x))

def get_time_for_call(median=13, IQR=28):
    return(get_geometric(median, IQR))

def get_time_for_ambulance_arrival(median=11, IQR=6):
    return(get_geometric(median, IQR))

def get_time_for_ambulance_assesment(median=14, IQR=10):
    return(get_geometric(median, IQR))

# At-Hospital Functions
def get_time_for_admission(median=15, IQR=16):
    return get_geometric(median, IQR)

def get_time_for_ct_scan(median=60, IQR=45):
    return get_geometric(median, IQR)

# def get_time_for_admission(hour):
#     if 8<=hour<=16:
#         return max(1, int(np.random.normal(25, 6.25)))
#     elif 16<hour<=23 or hour==00:
#         return max(1, int(np.random.normal(25, 6.25)))
#     else:
#         return max(1, int(np.random.normal(25, 6.25)))

# Transfer Functions
def get_time_for_ambulance_arrival_for_transfer(median=10, IQR=7):
    return get_geometric(median, IQR)

def get_time_for_ambulance_departure_for_transfer(median=17, IQR=7):
    return get_geometric(median, IQR)

def get_transfer_decision(proportion_transfer, is_ECR):
    if is_ECR == 1:
        return 1
    else:
        return np.random.binomial(n=1, p=proportion_transfer)

def get_hospital_of_transfer(hospital, hour):
    monash_time = transfer_to_monash_data[hospital].iloc[hour]
    rmh_time = transfer_to_rmh_data[hospital].iloc[hour]

    if monash_time <= rmh_time:
        choice = 0
        return central_hospitals[choice], choice, monash_time
    else:
        choice = 1
        return central_hospitals[choice], choice, rmh_time

def get_time_for_CSC_receive_ECR_transfer(median=20, IQR=21):
    return get_geometric(median, IQR)

# # Angio Functions

def get_ECR_status():
    return np.random.binomial(n=1, p=proportion_to_receive_ECR)

def get_angio_time(median=40, IQR=50):
    return get_geometric(median, IQR)

def get_angio_time_V(hour):
    median=40
    IQR=50
    if 8>=int(hour)>=18:
        return get_geometric(median, IQR)
    else:
        return get_geometric(median, IQR)

# Average
def average(x):
    return sum(x) / len(x)


# Time functions
def getDateTime(x):  # where x = env.now
    unixTime = start_date_unix + x * 60
    return datetime.datetime.fromtimestamp(int(unixTime)).strftime("%d-%m-%y %H:%M")

def getTime(x):  # where x = env.now
    unixTime = start_date_unix + x * 60
    return datetime.datetime.fromtimestamp(int(unixTime)).strftime("%H:%M")

def getHour(x):
    unixTime = start_date_unix + x * 60
    return datetime.datetime.fromtimestamp(int(unixTime)).strftime("%H")

def getDate(x):  # where x = env.now
    unixTime = start_date_unix + x * 60
    return datetime.datetime.fromtimestamp(int(unixTime)).strftime("%d-%m-%y")

def getDay(x):  # where x = env.now
    unixTime = start_date_unix + x * 60
    day = datetime.datetime.fromtimestamp(int(unixTime)).strftime("%a")
    return day


# # Simulation

# ## Data Container


columns = [
    "stroke_incident_time_minutes",
    "stroke_incident_time_date",
    "stroke_location_latitude",
    "stroke_location_longitude",
    "stroke_location_sed",
    "location_type",
    "time_for_call",
    "time_for_ambulance_arrival",
    "time_for_ambulance_assessment",
    "time_for_ambulance_transfer",
    "time_for_hospital_bed_wait",
    "hospital_of_arrival",
    "time_for_admission",
    "time_for_ct_scan",
    "is_transferred",
    "time_for_ambulance_arrival_for_transfer",
    "get_time_for_ambulance_departure_for_transfer",
    "time_for_hospital_transfer",
    "time_for_transfer_bed_wait",
    "hospital_of_transfer",
    "time_for_transferred_admission",
    "is_ECR",
    "is_angio",
    "is_limited_angio",
    "time_for_angio_resource_wait",
    "time_for_angio",
    "stroke_end_time_minutes",
    "time_total",
    "category",
]

container = []

# ## Resource Utilisation

class MonitoredResource(simpy.Resource):
    def __init__(self, resource_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = resource_name
        self.data = []

    def record_item(self, event):
        current_item = {}
        current_item["name"] = self.name
        current_item["event"] = event
        current_item["event_time"] = self._env.now
        current_item["capacity_at_event_time"] = self.count
        current_item["queue_at_event_time"] = len(self.queue)
        self.data.append(current_item)

    def request(self, *args, **kwargs):
        self.record_item("request")
        return super().request(*args, **kwargs)

    def release(self, *args, **kwargs):
        self.record_item("release")
        return super().release(*args, **kwargs)

# ## Running the Simulation

# ### Environment


env = simpy.Environment()


# ### Resources

# Dictionary of simpy hospital resources keyed by hospital name
hospital_resources = {
    hospital: (MonitoredResource(hospital, env, capacity=row["capacity"]))
    for hospital, row in hospital_data.iterrows()
}

# Dictionary of simpy angio resources keyed by hospital name (central only)
# default capacity 1?
angio_resources = {
    hospital: (MonitoredResource(hospital, env, capacity=1))
    for hospital, row in hospital_data.iterrows()
    if row["hospital_type"] == "central"
}

angio_resources["The Alfred"] = MonitoredResource("The Alfred", env, capacity=1)


# ### Simulation Body


def patient(env, stroke_incident_time):

    # Stroke occurs.
    patient_info = get_patient(test_time = getTime(stroke_incident_time))
    stroke_incident_timedate = getDateTime(stroke_incident_time)

    stroke_loc_lat = patient_info[2]
    stroke_loc_long = patient_info[3]
    stroke_loc_sed = patient_info[4]
    hospital_of_arrival = patient_info[1]
    location_type = hospital_data.at[hospital_of_arrival, "hospital_type"]
    is_ECR = get_ECR_status()

    # if location_type == "rural":
    #     rate_call = 2.5
    #     rate_ambulance = 30
    # else:
    #     rate_call = 2
    #     rate_ambulance = 20

    # Ambulance is called and transfers patient.
    time_for_call = get_time_for_call()
    time_for_ambulance_arrival = get_time_for_ambulance_arrival()
    time_for_ambulance_assessment = get_time_for_ambulance_assesment()
    time_for_ambulance_transfer = patient_info[0]

    # Patient arrives at hospital.
    time_for_admission = get_time_for_admission()
    time_for_ct_scan = get_time_for_ct_scan()

    # Supply times to the simulation.
    yield env.timeout(stroke_incident_time)
    start_time = env.now
    yield env.timeout(time_for_call)
    yield env.timeout(time_for_ambulance_arrival)
    yield env.timeout(time_for_ambulance_assessment)
    yield env.timeout(time_for_ambulance_transfer)

    # Save data
    data = {
        "is_ECR": is_ECR,
        "stroke_incident_time_minutes": stroke_incident_time,
        "stroke_incident_time_date": stroke_incident_timedate,
        "stroke_location_latitude": stroke_loc_lat,
        "stroke_location_longitude": stroke_loc_long,
        "stroke_location_sed": stroke_loc_sed,
        "location_type": location_type,
        "time_for_call": time_for_call,
        "time_for_ambulance_arrival": time_for_ambulance_arrival,
        "time_for_ambulance_assessment": time_for_ambulance_assessment,
        "time_for_ambulance_transfer": time_for_ambulance_transfer,
        "hospital_of_arrival": hospital_of_arrival,
        "time_for_admission": time_for_admission,
        "time_for_ct_scan": time_for_ct_scan,
    }

    time_before_arrival = env.now

    # Apply next timeouts only after hospital resource becomes available
    with hospital_resources[hospital_of_arrival].request() as request:
        yield request
        data["time_for_hospital_bed_wait"] = env.now - time_before_arrival
        yield env.timeout(time_for_admission)
        yield env.timeout(time_for_ct_scan)

        # If arrived at a central hospital
        if hospital_of_arrival in central_hospitals:
            is_transferred = 0

            data["is_transferred"] = is_transferred

            # # An angio decision is made at the initial, central hospital
            # is_angio = get_angio_decision(0.95)
            # data["is_angio"] = is_angio

            is_angio = is_ECR
            is_limited_angio = 0
            data["is_angio"] = is_angio
            data["is_limited_angio"] = is_limited_angio

            if is_angio:
                time_before_angio = env.now
                with angio_resources[hospital_of_arrival].request() as angio_request:
                    yield angio_request
                    data["time_for_angio_resource_wait"] = env.now - time_before_angio
                    time_for_angio = get_angio_time_V(getHour(env.now))

                    data["time_for_angio"] = time_for_angio

                    yield env.timeout(time_for_angio)
                    data["category"] = 1
            else:
                data["category"] = 2
        # Else, a transfer decision is made if rural or peripheral.
        else:
            if location_type == "rural":
                is_transferred = get_transfer_decision(proportion_transfer_from_rural, is_ECR)
            else:  # is metropolitan, peripheral
                is_transferred = get_transfer_decision(
                    proportion_transfer_from_peripheral, is_ECR
                )

            data["is_transferred"] = is_transferred

            if is_transferred:
                time_before_angio = env.now
                if hospital_of_arrival == "The Alfred" and 8<=int(getHour(env.now))<=16 and is_ECR ==1:
                    is_limited_angio = is_ECR
                    data["is_limited_angio"] = is_limited_angio
                    data["is_angio"] = is_ECR
                    with angio_resources["The Alfred"].request() as angio_request:
                        yield angio_request
                        data["time_for_angio_resource_wait"] = env.now - time_before_angio
                        time_for_angio = get_angio_time_V(getHour(env.now))
                        data["time_for_angio"] = time_for_angio
                        yield env.timeout(time_for_angio)
                        data["category"] = 3
                else:
                    is_limited_angio = 0
                    data["is_limited_angio"] = is_limited_angio
                    hospital_transfer_data = get_hospital_of_transfer(hospital_of_arrival, int(getHour(env.now)))
                    hospital_of_transfer = hospital_transfer_data[0]
                    to_hospital_id = hospital_transfer_data[1]
                    time_for_ambulance_arrival_for_transfer = get_time_for_ambulance_arrival_for_transfer()
                    time_for_ambulance_departure_for_transfer = get_time_for_ambulance_departure_for_transfer()
                    time_for_hospital_transfer = hospital_transfer_data[2]
                    time_for_transferred_admission = get_time_for_CSC_receive_ECR_transfer()
                    # time_for_transferred_ct_scan = (
                    #     time_for_ct_scan
                    # )  # Use the same one? #get_time_for_ct_scan()

                    data["time_for_ambulance_arrival_for_transfer"] = time_for_ambulance_arrival_for_transfer
                    data["get_time_for_ambulance_departure_for_transfer"] = time_for_ambulance_departure_for_transfer
                    data["time_for_hospital_transfer"] = time_for_hospital_transfer
                    data["hospital_of_transfer"] = hospital_of_transfer
                    data["time_for_transferred_admission"] = time_for_transferred_admission
                    # data["time_for_transferred_ct_scan"] = time_for_transferred_ct_scan
                    yield env.timeout(time_for_ambulance_arrival_for_transfer)
                    yield env.timeout(time_for_ambulance_departure_for_transfer)
                    yield env.timeout(time_for_hospital_transfer)

                    time_before_transfer = env.now

                    # Apply only after hospital resource becomes available.
                    with hospital_resources[
                        hospital_of_transfer
                    ].request() as transfer_request:
                        yield transfer_request
                        data["time_for_transfer_bed_wait"] = env.now - time_before_transfer
                        yield env.timeout(time_for_transferred_admission)
                        # yield env.timeout(time_for_transferred_ct_scan)

                        is_angio = is_ECR
                        data["is_angio"] = is_angio

                        if is_angio:
                            time_before_angio = env.now
                            with angio_resources[
                                hospital_of_transfer
                            ].request() as angio_request:
                                yield angio_request
                                data["time_for_angio_resource_wait"] = env.now - time_before_angio
                                time_for_angio = get_angio_time_V(getHour(env.now))

                                data["time_for_angio"] = time_for_angio

                                yield env.timeout(time_for_angio)
                                if location_type == "rural": #rural/PSC ECR
                                    data["category"] = 7
                                else:
                                    data["category"] = 4
                        else:
                            if location_type == "rural": #rural/PSC non ECR
                                data["category"] = 8
                            else:
                                data["category"] = 5
            else:
                is_limited_angio = 0
                data["is_limited_angio"] = is_limited_angio
                if location_type == "rural":
                    data["category"] = 9
                else:
                    data["category"] = 6

    end_time = env.now
    stroke_end_time_minutes = end_time
    time_total = (
        end_time - start_time
    )  # sanity check. Should be calculateable from dataframe.

    data["stroke_end_time_minutes"] = end_time
    data["time_total"] = time_total
    container.append(data)

# Register processes

for incident_time in get_stroke_incident_time():
    env.process(patient(env, incident_time))


# Run the simulation

env.run()


# Main Data Frame

dataframe = pd.DataFrame.from_records(container, columns=columns)

# Resource dataframe
hospital_resource_data = [record
                          for resource in hospital_resources.keys()
                          for record in hospital_resources[resource].data]
angio_resource_data = [record
                       for resource in angio_resources
                       for record in angio_resources[resource].data]

hospital_usage = pd.DataFrame.from_records(hospital_resource_data)
angio_usage = pd.DataFrame.from_records(angio_resource_data)

# ### Saving Data

file_datetime = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")

dataframe.to_csv("data/processed/2hosp/2hosp_10_simulation_{}.csv".format(file_datetime))
hospital_usage.to_csv("data/processed/2hosp/2hosp_10_hospital_usage_{}.csv".format(file_datetime))
angio_usage.to_csv("data/processed/2hosp/2hosp_10_angio_usage_{}.csv".format(file_datetime))


# # Data Analysis
# See the notebooks/submission_exploratory.ipynb for graphics.


min_time = min(dataframe.time_total)
max_time = max(dataframe.time_total)

plt.figure()
plt.suptitle("Total Stroke Management Times by Data Subset -2hosp 10% ECR")

combinations = [
    ("rural", 0, 0),
    ("rural", 1, 0),
    ("peripheral", 0, 0),
    ("peripheral", 1, 1),
    ("peripheral", 1, 0),
    ("central", 0, 0),
    ("central", 1, 0),
]

def ECR_string(is_ECR):
    if is_ECR:
        return "ECR"
    else:
        return "No ECR"

def limited_string(is_limited_angio):
    if is_limited_angio:
        return "Limited"
    else:
        return ""

for i, combo in enumerate(combinations):
    time_total_array = np.array(
                            dataframe[
                                (dataframe.location_type == combo[0])
                                & (dataframe.is_ECR == combo[1])
                                & (dataframe.is_limited_angio == combo[2])
                                & (dataframe.stroke_incident_time_minutes >= 2*365.25*60*24) #burn in time
                            ].time_total
                        )
    plt.subplot(1, 7, i + 1)
    plt.boxplot(time_total_array.reshape(-1,1))
    plt.xlabel("{} {}\n{}\n{}\n{}\n{}".format(combo[0].upper(), ECR_string(combo[1]), limited_string(combo[2]), np.percentile(time_total_array,75), np.percentile(time_total_array, 50), np.percentile(time_total_array, 25)))
    plt.ylim([min_time, max_time])

plt.show()
