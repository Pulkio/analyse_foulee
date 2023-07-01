import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math


############################ STEP 1 : import your datas  ##############################################
# Read data from csv file
#accel_data = pd.read_csv("../data/test_4_kyks.csv", header=0, skiprows=11)

accel_data = pd.read_csv("../datas_V2/breq1_full.csv", header=0, skiprows=1)
accel_data = accel_data.iloc[:, :-1]


#######################################################################################################


############################ STEP 2 : clean up data ##############################################
#Using dv instead of Acc : "Acc" measures acceleration and "dv" measures the changes in velocity.

# Rename columns
accel_data = accel_data.rename(columns={"dv[1]": "dv1", "dv[2]": "dv2", "dv[3]": "dv3"}) 

# Calculate time in minutes
accel_data["time"] = accel_data["PacketCounter"] / 60

# Calculate gravitational acceleration along z-axis
g = accel_data["dv3"].mean()

# Correct acceleration data along z-axis by subtracting gravitational acceleration
accel_data["accel_z_corrected"] = accel_data["dv3"] - g

#######################################################################################################


############################ STEP 3 : Stride detection and  refocuses data from the first and last detected stride ##############################################

# 60 hz frequency
fs = 60

# Calculate the minimum distance between peaks
min_distance =math.ceil(fs * 0.25) #Allows 4 steps in 1 sec knowing that 4 * 60 = 240 ppm >>>> 180 ppm average

#data_filtered corresponds to the filtered data of accel_data. This dataframe is used in the following code
data_filtered = accel_data

# search peaks in acceleration data (zUp) 
valleys, _ = find_peaks(-data_filtered["accel_z_corrected"], distance=min_distance)

# Look for the time the first stride is detected
globalStartTime = data_filtered.iloc[valleys[0]]["time"]

# Look for the time the last stride is detected
globalEndTime = data_filtered.iloc[valleys[-1]]["time"]

#  refocuses data from the first and last detected stride
data_filtered = accel_data[(accel_data["time"] > globalStartTime) & (accel_data["time"] < globalEndTime)]

#Start the time from 0
data_filtered["time"] = data_filtered["time"] - data_filtered["time"].iloc[0]

# Reset index to 0
data_filtered = data_filtered.reset_index(drop=True)

#######################################################################################################


############################ STEP 4 : Looks for peaks with refocused datas and plot the result ##############################################

valleys, _ = find_peaks(-data_filtered["accel_z_corrected"], distance=min_distance)


# Plot the valleys on top of the filtered data
plt.plot(data_filtered["time"], data_filtered["accel_z_corrected"], color="red")
plt.plot(data_filtered["time"].iloc[valleys], data_filtered["accel_z_corrected"].iloc[valleys], "x", color="green")
plt.xlabel("Time")
plt.ylabel("Acceleration in Z-axis")
plt.show()

#######################################################################################################


############################ STEP 5 : Create a stride column that increases its number by 1 with each stride change ##############################################

# Initialize stride column with zeros
data_filtered["stride"] = 0

# Set the first stride as 1 (beginning of first stride)
data_filtered.at[valleys[0], "stride"] = 1

# Assign stride number to each row based on the index of the previous stride
for i in range(1, len(valleys)):
    data_filtered["stride"].iloc[valleys[i-1]:valleys[i]] = i
    
#######################################################################################################




############################ STEP 6 : Multiply the number of strides by 2 by dividing the time of each stride by 2, for more relevant results in the analysis ##############################################

#Remove lines to zero at the end of stride
def update_zeros(dataframe, col_name):
    last_num = None
    for index, row in dataframe.iterrows():
        if last_num is None:
            last_num = row[col_name]
        elif row[col_name] == 0 and last_num > 0:
            dataframe.at[index, col_name] = last_num + 1
        else:
            last_num = row[col_name]
    return dataframe


# Use the "update_zeros" function to update the "stride" column in the "data_filtered" dataframe
data_filtered = update_zeros(data_filtered, "stride")


#looks for index of stride change
stride = data_filtered['stride'].to_numpy()
stride_change = np.diff(stride) != 0
stride_change = np.append(stride_change, True) # Add a value for the last stride


# Multiply each value by 2

# lambda function which multiplies each number by 2
mult_by_2 = lambda x: x * 2

# apply the lambda function to the "colonne_nums" column of the dataframe
data_filtered["stride"] = data_filtered["stride"].apply(mult_by_2)


# Divide each stride into 2 strides 

# Count the iterations of each value in the "stride" column
count_stride_iter = pd.DataFrame(data_filtered['stride'].value_counts())

# Rename columns for clarity
count_stride_iter = count_stride_iter.rename(columns={'stride': 'iterations'})
 
count_stride_iter = count_stride_iter.sort_index()


# Divide the number of iterations by 2
count_stride_iter['iterations'] = count_stride_iter['iterations'] / 2

count_stride_iter.reset_index(inplace=True)


# double the number of lines
count_stride_iter = pd.concat([count_stride_iter, count_stride_iter])
# reset the index
count_stride_iter = count_stride_iter.reset_index(drop=True)
# get the index of the last line
last_row_index = count_stride_iter.iloc[-1].name

# Allows rounding to the upper and lower integer after the division by 2 to keep the same number of values
zero_count = 0
for i, row in count_stride_iter.iterrows():
    if row['index'] == 0:
        zero_count += 1
    if zero_count >= 2:
        count_stride_iter.at[i, 'iterations'] = math.ceil(row['iterations'])
        count_stride_iter.at[i, 'index'] = row['index'] + 1
    else:
        count_stride_iter.at[i, 'iterations'] = math.floor(row['iterations'])

count_stride_iter = count_stride_iter.sort_values('index')

count_stride_iter.reset_index(inplace=True)
count_stride_iter = count_stride_iter.drop(count_stride_iter.columns[0], axis=1)

repeated_index = count_stride_iter['index'].repeat(count_stride_iter['iterations'])
data_filtered["stride"] = repeated_index.values


#######################################################################################################


############################ STEP 7 : Getting the results we want ##############################################

# Count the number of strides
num_strides = data_filtered["stride"].max()
num_strides = round(num_strides / 2)

print("\n\n\n")
print("RESULTS : ")
print("Number of strides :", num_strides)

# Calculate the total duration of the run in seconds
total_duration = data_filtered["time"].iloc[-1]

# Calculate the number of strides per second
strides_per_second = num_strides / total_duration

print("Step rate per minute :", round(strides_per_second * 60,2), " ppm")


# looks for the duration of each stride
def stride_time(data):
    stride_times = []
    current_stride = 0
    start_time = data['time'].iloc[0]
    for i, row in data.iterrows():
        if row['stride'] != current_stride:
            end_time = row['time']
            stride_times.append({'stride_num': current_stride,
                                 'time_elapsed': end_time - start_time})
            current_stride = row['stride']
            start_time = end_time
    return pd.DataFrame(stride_times)

stride_times = stride_time(data_filtered)

# Creates a table that will contain the average speed of each stride
mean_speed_strides = []
# Creates a table that will contain the average high of each stride
mean_high_stride = []
# the number of strides
max_stride = int(np.max(data_filtered["stride"])) # Valeur maximale de la colonne "stride"

oldTime = 0

# Calculate the average speed and the average height of each stride, thanks to the cumulative integration of dv for each axis
for i in range(max_stride):
    # Extracting data from the first stride
    stride_0_data = data_filtered[data_filtered["stride"] == i]
    
    if not stride_0_data.empty:
        # Selects only the most relevant data
        excludeEndOfStrideDatas = 1.12 # I've got the better results by doing this
        contact_time = (stride_times["time_elapsed"].iloc[i] / excludeEndOfStrideDatas) + oldTime
        contact_time_high = (stride_times["time_elapsed"].iloc[i] / 2) + oldTime
        
        #speed processing by resetting the accumulated integration to zero at the beginning of each stride to avoid errors 
  
        oldTime += stride_times["time_elapsed"].iloc[i]
        contact_data = stride_0_data.loc[stride_0_data['time'] < contact_time]
        contact_duration = contact_time - contact_data.iloc[0]["time"]
        
        # Integration of acceleration to calculate instantaneous speed
        dt = contact_duration / len(contact_data)
        v = np.cumsum(contact_data[["dv1", "dv2", "accel_z_corrected"]], axis=0) * dt
        
    
        
        # Calculation of the norm of the 3 axes to obtain the instantaneous speed for a stride
        v_mean = np.mean(np.linalg.norm(v, axis=1))
        
        
        #Add the result to the mean_speed_stride table
        mean_speed_strides.append(v_mean)
        
        #Do the same as for speed, but for stride height, so use only the Z axis ######################################################## 
        contact_data_high = stride_0_data.loc[stride_0_data['time'] < contact_time_high]

        contact_duration_high = contact_time_high - contact_data_high.iloc[0]["time"]
        
        # Integration of acceleration to calculate instantaneous speed
        dt = contact_duration_high / len(contact_data_high)
        
        # Get Z Accelerations data
        accel_z_high = contact_data_high["accel_z_corrected"]

        # Digital integration for vertical speed
        velocity_z_high = np.cumsum(accel_z_high) * dt

        position_z_high = np.cumsum(velocity_z_high) * dt
        
        #Add the result to the mean_high_stride table
        mean_high_stride.append(max(position_z_high))
    else:
        print("stride_data est vide pour la valeur de i spécifiée.")

# Calculate the average running speed, from the average speeds of each stride
moyenne_speed_strides = sum(mean_speed_strides) / len(mean_speed_strides)

# Calculate the average running high, from the average high of each stride
moyenne_high_strides = sum(mean_high_stride) / len(mean_high_stride)

# Multiply the value obtained by 100 to have the speed in m.s
moyenne_speed_strides = moyenne_speed_strides * 100

# Displays speed in km/h
print("The average speed is of :", round(moyenne_speed_strides * 3.6, 2 ), " km/h")

# Calculating distance travelled from mean speed and time
distance_parcourue = moyenne_speed_strides * total_duration

#Displays the distance travelled in m
print("The distance travelled is of : ", round(distance_parcourue,2), "m")

# Calculation of average stride length from distance travelled and number of strides
longueur_moy_foulee = distance_parcourue / num_strides

#Displays average stride length in m 
print("The average stride length is of : ", round(longueur_moy_foulee,2), "m")

#Displays average stride high in m 
print("The average stride high is of : " , round(moyenne_high_strides*1000,2), " m")

print("\n\n\n")
