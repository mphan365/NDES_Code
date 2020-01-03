import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Extract the Simulations 
two_hosp = pd.read_csv("data/processed/2hosp/2019_03_08_2241_simulation.csv")
three_hosp = pd.read_csv("data/processed/3hosp/2019_03_08_2242_simulation.csv")


## ANGIO-FREE TIME
two_hosp.fillna(0,inplace=True)
two_hosp['angio_free_time'] = two_hosp.time_total - two_hosp.time_for_angio_resource_wait - two_hosp.time_for_angio

three_hosp.fillna(0,inplace=True)
three_hosp['angio_free_time'] = three_hosp.time_total - three_hosp.time_for_angio_resource_wait - three_hosp.time_for_angio


### 'BOXPLOT OF PATIENTS REQUIRING ANGIO'
two_hosp[two_hosp.is_transferred == True].boxplot(['angio_free_time'],by=['location_type'])
plt.title('Total Stroke Management without Angio Treatment by transfer (2 hospitals)')
plt.xlabel('2 hospitals')
plt.ylabel('time taken (minutes)')

three_hosp[three_hosp.is_transferred == True].boxplot(['angio_free_time'],by=['location_type'])
plt.title('Total Stroke Management without Angio Treatment by transfer (3 hospitals)')
plt.xlabel('3 hospitals')
plt.ylabel('time taken (minutes)')

two_hosp[two_hosp.is_transferred == False].boxplot(['angio_free_time'],by=['location_type'])
plt.title('Total Stroke Management without Angio Treatment (no transfer, 2 hospitals)')
plt.xlabel('2 hospitals')
plt.ylabel('time taken (minutes)')

three_hosp[three_hosp.is_transferred == False].boxplot(['angio_free_time'],by=['location_type'])
plt.title('Total Stroke Management Time without Angio Treatment (no transfer, 3 hospitals)')
plt.xlabel('3 hospitals')
plt.ylabel('time taken (minutes)')

### PLOTTING BASED ON LOCATION TYPE (RURAL AND PERIPHERAL SEPARATED BY LOCATION)
two_hosp[two_hosp.is_angio == True].boxplot(['time_total'],by=['location_type'])
plt.title('Total Stroke Management by location (2 hospitals)')
plt.xlabel('2 hospitals')
plt.ylabel('time taken (minutes)')

three_hosp[three_hosp.is_angio == True].boxplot(['time_total'],by=['location_type'])
plt.title('Total Stroke Management by location (3 hospitals)')
plt.xlabel('3 hospitals')
plt.ylabel('time taken (minutes)')

# ### SEPARATE BY 2 vs 3 hosp
# min_time = 0
# max_time = max(max(two_hosp.time_total),max(three_hosp.time_total))+100
#
# two_hosp.dropna(subset=['is_angio'],inplace=True)
# three_hosp.dropna(subset=['is_angio'],inplace=True)
# 
# plt.figure()
# plt.subplot(1,2,1)
# two_hosp[two_hosp.location_type == 'central'].boxplot(['time_total'])
# plt.xlabel('2 hospitals')
# plt.ylabel('time taken (minutes)')
# plt.ylim(min_time,max_time)
# plt.subplot(1,2,2)
# three_hosp[three_hosp[three.location_type == 'central'].boxplot(['time_total'])
# plt.xlabel('3 hospitals')
# plt.ylabel('time taken (minutes)')
# plt.ylim(min_time,max_time)
# 
# 
# plt.figure()
# plt.subplot(1,2,1)
# two_hosp[two_hosp.location_type == 'rural'].boxplot(['time_total'])
# plt.xlabel('2 hospitals')
# plt.ylabel('time taken (minutes)')
# plt.ylim(min_time,max_time)
# plt.subplot(1,2,2)
# three_hosp[three_hosp.location_type == 'rural'].boxplot(['time_total'])
# plt.xlabel('3 hospitals')
# plt.ylabel('time taken (minutes)')
# plt.ylim(min_time,max_time)
#
# 
# plt.figure()
# plt.subplot(1,2,1)
# two_hosp[two_hosp.location_type == 'peripheral'].boxplot(['time_total'])
# plt.xlabel('2 hospitals')
# plt.ylabel('time taken (minutes)')
# plt.ylim(min_time,max_time)
# plt.subplot(1,2,2)
# three_hosp[three_hosp.location_type == 'peripheral'].boxplot(['time_total'])
# plt.xlabel('3 hospitals')
# plt.ylabel('time taken (minutes)')
# plt.ylim(min_time,max_time)




plt.show()