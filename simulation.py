# ## Imports
import random
import simpy
import functools
from functools import partial, wraps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
from modules.travel_times import travel_times
from scipy.stats import norm

## Victoria Specific Data

# Hospital Data Imports
hosp_3_data_dict = {
    "rural_hospital_data": "data/internal/hospitalupdate/3hosp/rhosp.csv",
    "peripheral_hospital_data": "data/internal/hospitalupdate/3hosp/pschosp.csv",
    "ecrCapable_hospital_data": "data/internal/hospitalupdate/3hosp/ecrCapablehosp.csv",
    "central_hospital_data": "data/internal/hospitalupdate/3hosp/cschosp.csv",
    "transfer_to_austin_data": "data/internal/hospitalupdate/3hosp/t2austin.csv",
    "transfer_to_monash_data": "data/internal/hospitalupdate/3hosp/t2monash.csv",
    "transfer_to_rmh_data": "data/internal/hospitalupdate/3hosp/t2rmh.csv"
}

hosp_2_data_dict = {
    "rural_hospital_data": "data/internal/hospitalupdate/2hosp/rhosp.csv",
    "peripheral_hospital_data": "data/internal/hospitalupdate/2hosp/pschosp.csv",
    "ecrCapable_hospital_data": "data/internal/hospitalupdate/2hosp/ecrCapablehosp.csv",
    "central_hospital_data": "data/internal/hospitalupdate/2hosp/cschosp.csv",
    "transfer_to_austin_data": "data/internal/hospitalupdate/2hosp/t2austin.csv",
    "transfer_to_monash_data": "data/internal/hospitalupdate/2hosp/t2monash.csv",
    "transfer_to_rmh_data": "data/internal/hospitalupdate/2hosp/t2rmh.csv"
}

hosp_data_dict_list = [hosp_2_data_dict, hosp_3_data_dict]
proportions_list = [0.03, 0.05, 0.1, 0.15]

def vicSim(
hosp_data_dict,
proportion_to_receive_ECR
):
# ## Variables
    hosp_data_dict = hosp_data_dict
    strokes_per_year = 14239
    start_date = "01/01/2019"
    simulation_runtime = 5
    burn_in = 0.5
    proportion_to_receive_ECR = proportion_to_receive_ECR
    proportion_nearest_rural = 0.95
    proportion_nearest_peripheral = 0.95
    proportion_nearest_central = 0.95
    proportion_transfer = 0.01

    amb_call_time_q1 = 1
    amb_call_time_median = 13
    amb_call_time_q3 = 29

    amb_arrive_time_q1 = 8
    amb_arrive_time_median = 11
    amb_arrive_time_q3 = 14

    amb_assess_time_q1 = 10
    amb_assess_time_median = 14
    amb_assess_time_q3 = 20

    hosp_admit_time_q1 = 8
    hosp_admit_time_median = 15
    hosp_admit_time_q3 = 23.5

    cschosp_admit_time_median = 10
    cschosp_admit_time_sd = 2.5

    csc_ct_scan_time_median = 15
    csc_ct_scan_time_sd = 2.5

    ct_scan_time_q1 = 44
    ct_scan_time_median =59.5
    ct_scan_time_q3 = 83

    csc_door_to_puncture_median = 15
    csc_door_to_puncture_sd = 2.5

    transfer_amb_arrive_time_q1 = 7
    transfer_amb_arrive_time_median = 10
    transfer_amb_arrive_time_q3 = 14

    transfer_amb_depart_time_q1 = 14
    transfer_amb_depart_time_median = 17
    transfer_amb_depart_time_q3 = 21

    csc_receive_ecr_transfer_time_q1 = 15
    csc_receive_ecr_transfer_time_median = 20
    csc_receive_ecr_transfer_time_q3 = 35.75

    angio_reperfusion_time_q1 = 23.75
    angio_reperfusion_time_median = 40
    angio_reperfusion_time_q3 = 73.5

    angio_cutoff_time_hours = 6
    anigo_optimal_time_hours = 2
    acutebed_stay_median_hours = 48
    acute_bed_stay_sd_hours = 6

    proportion_transferred_requiring_CT = 0.3

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

    nonacute_bed_stay_ECR_median_mins = nonacute_bed_stay_ECR_median_days*24*60
    nonacute_bed_stay_ECR_IQR_mins = nonacute_bed_stay_ECR_IQR_days*24*60
    nonacute_bed_stay_nECR_median_mins = nonacute_bed_stay_nECR_median_days*24*60
    nonacute_bed_stay_nECR_IQR_mins = nonacute_bed_stay_nECR_IQR_days*24*60

    # Import Victoria Data
    hospital_data_columns = [
        "name",
        "codestrokeCapacity",
        "acutebedCapacity",
        "nonacutebedCapacity"
    ]

    rural_hospital_data = pd.read_csv(
        hosp_data_dict["rural_hospital_data"], names=hospital_data_columns
    )
    peripheral_hospital_data = pd.read_csv(
        hosp_data_dict["peripheral_hospital_data"], names=hospital_data_columns
    )
    ecrCapable_hospital_data = pd.read_csv(
        hosp_data_dict["ecrCapable_hospital_data"], names=hospital_data_columns
    )
    central_hospital_data = pd.read_csv(
        hosp_data_dict["central_hospital_data"], names=hospital_data_columns
    )

    rural_hospital_data["hospital_type"] = "rural"
    peripheral_hospital_data["hospital_type"] = "peripheral"
    ecrCapable_hospital_data["hospital_type"] = "ecrCapable"
    central_hospital_data["hospital_type"] = "central"

    hospital_data = pd.concat(
        [rural_hospital_data, peripheral_hospital_data, ecrCapable_hospital_data, central_hospital_data]
    ).set_index("name")

    # Transfer Data Imports
    transfer_to_austin_data = (
        pd.read_csv(hosp_data_dict["transfer_to_austin_data"]).set_index("Hospitals").T
    )
    transfer_to_monash_data = (
        pd.read_csv(hosp_data_dict["transfer_to_monash_data"]).set_index("Hospitals").T
    )
    transfer_to_rmh_data = (
        pd.read_csv(hosp_data_dict["transfer_to_rmh_data"]).set_index("Hospitals").T
    )

    transfer_to_central_data = [
        transfer_to_austin_data,
        transfer_to_monash_data,
        transfer_to_rmh_data,
    ]

    # Travel Time Data
    travel_time_data = pd.read_pickle("data/preprocessed/travel_times_updatehospitals.pkl")
    population_data = pd.read_csv("data/external/2016Census_G01_VIC_SED.csv")
    population_data = population_data[:-2]  # exclude two non-location SEDs
    total_population = sum(population_data["Tot_P_P"])
    weighted_proportions = population_data["Tot_P_P"] / total_population
    sed_id_list = population_data["SED_CODE_2016"]

    # ##Functions
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

    def get_acute_bed_stay():
        return np.random.normal(acutebed_stay_median_mins, acute_bed_stay_sd_mins)

    def getDateTime(now):
        unixTime = start_date_unix + now * 60
        return datetime.datetime.fromtimestamp(int(unixTime)).strftime("%d-%m-%y %H:%M")

    def getTime(now):
        unixTime = start_date_unix + now * 60
        return datetime.datetime.fromtimestamp(int(unixTime)).strftime("%H:%M")

    def getHour(now):
        unixTime = start_date_unix + now * 60
        return datetime.datetime.fromtimestamp(int(unixTime)).strftime("%H")

 # Victoria Specific Simulation Functions
    def get_hospital_of_transfer(hospital, hour):
        if hosp_data_dict == hosp_3_data_dict:
            central_hospitals = ["Austin Health", "Monash Health", "Royal Melbourne Hospital"]
            austin_time = transfer_to_austin_data[hospital].iloc[hour]
            monash_time = transfer_to_monash_data[hospital].iloc[hour]
            rmh_time = transfer_to_rmh_data[hospital].iloc[hour]

            if austin_time < monash_time and austin_time < rmh_time:
                choice = 0
                return central_hospitals[choice], choice, austin_time
            elif monash_time < austin_time and monash_time < rmh_time:
                choice = 1
                return central_hospitals[choice], choice, monash_time
            else:
                choice = 2
                return central_hospitals[choice], choice, rmh_time
        else:
            central_hospitals = ["Monash Health", "Royal Melbourne Hospital"]
            monash_time = transfer_to_monash_data[hospital].iloc[hour]
            rmh_time = transfer_to_rmh_data[hospital].iloc[hour]

            if monash_time <= rmh_time:
                choice = 0
                return central_hospitals[choice], choice, monash_time
            else:
                choice = 1
                return central_hospitals[choice], choice, rmh_time

    # # Simulation

    # ## Data Container
    columns = [
        "stroke_incident_time_minutes",
        "stroke_incident_time_date",
        "stroke_location_latitude",
        "stroke_location_longitude",
        "stroke_location_sed",
        "location_type",
        "is_ECR",
        "time_for_call",
        "time_for_ambulance_arrival",
        "time_for_ambulance_assessment",
        "time_for_ambulance_transfer",
        "time_for_hospital_bed_wait",
        "hospital_of_arrival",
        "time_for_admission",
        "time_for_ct_scan",
        "door_to_puncture",
        "is_transferred",
        "time_for_ambulance_arrival_for_transfer",
        "time_for_ambulance_departure_for_transfer",
        "time_for_hospital_transfer",
        "time_for_transferred_admission",
        "time_for_transfer_bed_wait",
        "hospital_of_transfer",
        "time_for_transferred_admission",
        "time_for_treatment_wait",
        "angio_decision",
        "angio_category",
        "angio_time",
        "acute_bed_stay_time",
        "time_total",
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

    # Dictionary of simpy hospital codestroke resources keyed by hospital name
    hospital_codestroke_resources = {
        hospital: (MonitoredResource(hospital, env, capacity=row["codestrokeCapacity"]))
        for hospital, row in hospital_data.iterrows()
    }
    # Dictionary of simpy hospital acute bed resources keyed by hospital name
    hospital_acutebed_resources = {
        hospital: (MonitoredResource(hospital, env, capacity=row["acutebedCapacity"]))
        for hospital, row in hospital_data.iterrows()
        if row["hospital_type"] == "central" or row["hospital_type"] == "ecrCapable"
    }

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


        # Ambulance is called, arrives, assesses, and transfers patient.
        time_for_call = get_time_for_call()
        time_for_ambulance_arrival = get_time_for_ambulance_arrival()
        time_for_ambulance_assessment = get_time_for_ambulance_assesment()
        time_for_ambulance_transfer = patient_info[0]

        # Patient arrives at hospital.
        time_for_admission = get_time_for_admission(location_type)
        time_for_ct_scan = get_time_for_ct_scan(location_type)

        # Supply times to the simulation.
        yield env.timeout(stroke_incident_time)
        start_time = env.now
        yield env.timeout(time_for_call)
        yield env.timeout(time_for_ambulance_arrival)
        yield env.timeout(time_for_ambulance_assessment)
        yield env.timeout(time_for_ambulance_transfer)
        time_before_arrival = env.now

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

        # Apply next timeouts only after hospital resource becomes available
        with hospital_codestroke_resources[hospital_of_arrival].request() as request:
            yield request
            data["time_for_hospital_bed_wait"] = env.now - time_before_arrival
            yield env.timeout(time_for_admission)
            yield env.timeout(time_for_ct_scan)

            # If arrived at a central hospital
            if location_type == "central":
                is_transferred = 0
                hospital_for_angio = hospital_of_arrival
                if is_ECR:
                    start_wait_for_angio = time_before_arrival
                    hospital_for_angio = hospital_of_arrival
                    door_to_puncture = get_time_for_csc_door_to_puncture()
                    yield env.timeout(door_to_puncture)
                    data["door_to_puncture"] = door_to_puncture
            elif location_type == "ecrCapable":
                if is_ECR and 8<=int(getHour(env.now))<= 17:
                    start_wait_for_angio = time_before_arrival
                    hospital_for_angio = hospital_of_arrival
                    is_transferred = 0
                    door_to_puncture = get_time_for_csc_door_to_puncture()
                    yield env.timeout(door_to_puncture)
                    data["door_to_puncture"] = door_to_puncture
                elif is_ECR:
                    is_transferred = 1

                else:
                    is_transferred=0
            else:
                if is_ECR:
                    is_transferred = 1


                else:
                    is_transferred=get_transfer_decision(proportion_transfer, is_ECR)

        data["is_transferred"] = is_transferred
        if is_transferred:
            hospital_transfer_data = get_hospital_of_transfer(hospital_of_arrival, int(getHour(env.now)))
            hospital_of_transfer = hospital_transfer_data[0]
            to_hospital_id = hospital_transfer_data[1]
            time_for_ambulance_arrival_for_transfer = get_time_for_ambulance_arrival_for_transfer()
            time_for_ambulance_departure_for_transfer = get_time_for_ambulance_departure_for_transfer()
            time_for_hospital_transfer = hospital_transfer_data[2]
            time_for_transferred_admission = get_time_for_CSC_receive_ECR_transfer()
            yield env.timeout(time_for_ambulance_arrival_for_transfer)
            yield env.timeout(time_for_ambulance_departure_for_transfer)
            yield env.timeout(time_for_hospital_transfer)
            data["hospital_of_transfer"] = hospital_of_transfer
            data["time_for_ambulance_arrival_for_transfer"] = time_for_ambulance_arrival_for_transfer
            data["time_for_ambulance_departure_for_transfer"] = time_for_ambulance_departure_for_transfer
            data["time_for_hospital_transfer"] = time_for_hospital_transfer
            start_wait_for_angio = env.now
            hospital_for_angio = hospital_of_transfer
            with hospital_codestroke_resources[hospital_of_transfer].request() as request:
                yield request
                post_transfer_ct = get_transfer_CT_decision()
                if post_transfer_ct:
                    transferred_ct_time = get_time_for_ct_scan("central")
                    transferred_ct_to_puncture_time = get_time_for_admission("central")
                    env.timeout(transferred_ct_time)
                    env.timeout(transferred_ct_to_puncture_time)
                else:
                    yield env.timeout(time_for_transferred_admission)
                    data["time_for_transferred_admission"] = time_for_transferred_admission

        if is_ECR:
            with hospital_acutebed_resources[hospital_for_angio].request() as request:
                yield request
                time_for_treatment_wait = env.now - start_wait_for_angio
                angio_decision = get_angio_decision(time_for_treatment_wait)
                angio_category = get_angio_category(time_for_treatment_wait)
                data["time_for_treatment_wait"] = time_for_treatment_wait
                data["angio_decision"] = angio_decision
                data["angio_category"] = angio_category
                if angio_decision:
                    angio_time = get_angio_time()
                    acute_bed_stay_time = get_acute_bed_stay()
                    yield env.timeout(angio_time)
                    yield env.timeout(acute_bed_stay_time)
                    data["angio_time"] = angio_time

        end_time = env.now
        if is_ECR:
            if angio_category ==1 or angio_category ==2:
                end_time = env.now - acute_bed_stay_time - angio_time
        stroke_end_time_minutes = end_time
        time_total = end_time - start_time # sanity check. Should be calculateable from dataframe.
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
    dataframe = dataframe[dataframe["stroke_incident_time_minutes"] >= burn_in_mins]

    # Resource dataframe
    codestroke_resource_data = [record
                      for resource in hospital_codestroke_resources.keys()
                      for record in hospital_codestroke_resources[resource].data]
    acutebed_resource_data = [record
                        for resource in hospital_acutebed_resources.keys()
                        for record in hospital_acutebed_resources[resource].data]

    codestroke_usage = pd.DataFrame.from_records(codestroke_resource_data)
    acutebed_usage = pd.DataFrame.from_records(acutebed_resource_data)


    # # Data Analysis
    data_anal_array_dict = {
        "central_angio_array": np.array(dataframe[
                                        (dataframe.location_type == "central") &
                                        (dataframe.angio_decision ==1)
        ].time_total),
        "central_non_angio_array": np.array(dataframe[
                                        (dataframe.location_type == "central") &
                                        (dataframe.is_ECR ==0)
        ].time_total),
        "ecrCapable_angio_array": np.array(dataframe[
                                        (dataframe.location_type == "ecrCapable") &
                                        (dataframe.angio_decision ==1)
        ].time_total),
        "ecrCapable_non_angio_array": np.array(dataframe[
                                        (dataframe.location_type == "ecrCapable") &
                                        (dataframe.is_ECR ==0)
        ].time_total),
        "peripheral_angio_array": np.array(dataframe[
                                        (dataframe.location_type == "peripheral") &
                                        (dataframe.angio_decision ==1)
        ].time_total),
        "peripheral_non_angio_array": np.array(dataframe[
                                        (dataframe.location_type == "peripheral") &
                                        (dataframe.is_ECR ==0)
        ].time_total),
        "rural_angio_array": np.array(dataframe[
                                        (dataframe.location_type == "rural") &
                                        (dataframe.angio_decision ==1)
        ].time_total),
        "rural_non_angio_array": np.array(dataframe[
                                        (dataframe.location_type == "rural") &
                                        (dataframe.is_ECR ==0)
        ].time_total),
        "optimal_angio_array": np.array(dataframe[
                                        (dataframe.angio_category == 1)
        ].time_total),
        "delayed_angio_array": np.array(dataframe[
                                        (dataframe.angio_category == 2)
        ].time_total),
        "unable_angio_array": np.array(dataframe[
                                        (dataframe.angio_category == 3)
        ].time_total),
        "no_ECR_array": np.array(dataframe[
                                        (dataframe.is_ECR ==1)
        ].time_total),
        "received_angio_array": np.array(dataframe[
                                        (dataframe.angio_decision==1)
        ].time_total),
        "rural_transferred_array": np.array(dataframe[
                                        (dataframe.is_transferred ==1) &
                                        (dataframe.location_type == "rural")
        ].time_total),
        "peripheral_transferred_array": np.array(dataframe[
                                        (dataframe.is_transferred ==1) &
                                        (dataframe.location_type== "peripheral")
        ].time_total),
        "central_array": np.array(dataframe[
                                        (dataframe.location_type == "central")
        ].time_total),
    }


    data_anal_dict = {
        "median_IQR_central_angio": [],
        "median_IQR_central_non_angio": [],
        "median_IQR_ecrCapable_angio": [],
        "median_IQR_ecrCapable_non_angio": [],
        "median_IQR_peripheral_angio": [],
        "median_IQR_peripheral_non_angio": [],
        "median_IQR_rural_angio": [],
        "median_IQR_rural_non_angio": [],
        "number_optimal_angio": [],

        "number_delayed_angio": [],

        "number_unable_angio": [],

        "median_IQR_angio_optimal_and_delayed": [],
        "median_IQR_angio_optimal_only": [],
        "number_received_angio": [],
        "number_no_angio": [],
        "number_transferred_from_rural": [],
        "number_transferred_from_peripheral": [],
        "number_originating_central": [],
        "median_IQR_rural_transport": [],
        "median_IQR_peripheral_transport": [],

    }

    percentiles = [50,25,75]

    for i in range(len(percentiles)):
        q = percentiles[i]
        data_anal_dict["median_IQR_central_angio"].append(
            np.percentile(data_anal_array_dict["central_angio_array"], percentiles[i]))
        data_anal_dict["median_IQR_central_non_angio"].append(
            np.percentile(data_anal_array_dict["central_non_angio_array"], percentiles[i]))
        data_anal_dict["median_IQR_ecrCapable_angio"].append(
            np.percentile(data_anal_array_dict["ecrCapable_angio_array"], percentiles[i]))
        data_anal_dict["median_IQR_ecrCapable_non_angio"].append(
            np.percentile(data_anal_array_dict["ecrCapable_non_angio_array"], percentiles[i]))
        data_anal_dict["median_IQR_peripheral_angio"].append(
            np.percentile(data_anal_array_dict["peripheral_angio_array"], percentiles[i]))
        data_anal_dict["median_IQR_peripheral_non_angio"].append(
            np.percentile(data_anal_array_dict["peripheral_non_angio_array"], percentiles[i]))
        data_anal_dict["median_IQR_rural_angio"].append(
            np.percentile(data_anal_array_dict["rural_angio_array"], percentiles[i]))
        data_anal_dict["median_IQR_rural_non_angio"].append(
            np.percentile((data_anal_array_dict["rural_non_angio_array"]), percentiles[i]))
        data_anal_dict["median_IQR_angio_optimal_and_delayed"].append(
            np.percentile(data_anal_array_dict["received_angio_array"], percentiles[i]))
        data_anal_dict["median_IQR_angio_optimal_only"].append(
            np.percentile((data_anal_array_dict["optimal_angio_array"]), percentiles[i]))
        data_anal_dict["median_IQR_rural_transport"].append(
            np.percentile((data_anal_array_dict["rural_transferred_array"]), percentiles[i]))
        data_anal_dict["median_IQR_peripheral_transport"].append(
            np.percentile((data_anal_array_dict["peripheral_transferred_array"]), percentiles[i]))

    t_angio = len(data_anal_array_dict["optimal_angio_array"])+len(data_anal_array_dict["delayed_angio_array"])
    t_no_angio = len(data_anal_array_dict["no_ECR_array"])+len(data_anal_array_dict["unable_angio_array"])
    n_optimal_angio = len(data_anal_array_dict["optimal_angio_array"])
    n_delayed_angio = len(data_anal_array_dict["delayed_angio_array"])
    n_unable_angio = len(data_anal_array_dict["unable_angio_array"])
    p_optimal_angio = n_optimal_angio/t_angio
    p_delayed_angio = n_delayed_angio/t_angio
    p_unable_angio = n_unable_angio/t_angio

    data_anal_dict["number_delayed_angio"].append(n_delayed_angio)
    data_anal_dict["number_delayed_angio"].append(p_delayed_angio)

    data_anal_dict["number_optimal_angio"].append(n_optimal_angio)
    data_anal_dict["number_optimal_angio"].append(p_optimal_angio)

    data_anal_dict["number_unable_angio"].append(n_unable_angio)
    data_anal_dict["number_unable_angio"].append(p_unable_angio)

    data_anal_dict["number_received_angio"].append(t_angio)
    data_anal_dict["number_no_angio"].append(t_no_angio)
    data_anal_dict["number_transferred_from_rural"].append(len(data_anal_array_dict["rural_transferred_array"]))
    data_anal_dict["number_transferred_from_peripheral"].append(len(data_anal_array_dict["peripheral_transferred_array"]))
    data_anal_dict["number_originating_central"].append(len(data_anal_array_dict["central_array"]))

    df_data_anal=pd.DataFrame.from_dict(data_anal_dict,orient='index').transpose()


    # ### Saving Data

    file_datetime = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    proportion = str(int(proportion_to_receive_ECR * 100))
    if hosp_data_dict == hosp_3_data_dict:
        n_hosp = "3hosp"
    elif hosp_data_dict == hosp_2_data_dict:
        n_hosp = "2hosp"
    else:
        n_hosp = "n_hosp"

    dataframe.to_csv("data/processed/{}_{}_{}_vicsim_simdata.csv".format(file_datetime, proportion, n_hosp))
    df_data_anal.to_csv("data/processed/{}_{}_{}_vicsim_dataanal.csv".format(file_datetime, proportion, n_hosp))
    codestroke_usage.to_csv("data/processed/{}_{}_{}_vicsim_codestroke_usage.csv".format(file_datetime, proportion, n_hosp))
    acutebed_usage.to_csv("data/processed/{}_{}_{}_vicsim_acutebed_usage.csv".format(file_datetime, proportion, n_hosp))


    min_time = min(dataframe.time_total)
    max_time = max(dataframe.time_total)


for x in range(len(proportions_list)):
    for y in range(len(hosp_data_dict_list)):
        vicSim(hosp_data_dict_list[y], proportions_list[x])
        print("completed {} {}".format(x, y))
