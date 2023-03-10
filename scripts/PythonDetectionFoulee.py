import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math
# Read data from csv file
accel_data = pd.read_csv("../data/test1.csv", header=0, skiprows=11)


############################################ a commenter si test1
#accel_data = pd.read_csv("../data/test6.csv", header=0, skiprows=1)
#accel_data = accel_data.iloc[:, :-1]
#################################################################

# Rename columns
accel_data = accel_data.rename(columns={"dv[1]": "dv1", "dv[2]": "dv2", "dv[3]": "dv3"}) 

# Calculate time in minutes
accel_data["time"] = accel_data["PacketCounter"] / 60

# Calculate gravitational acceleration along z-axis
g = accel_data["dv3"].mean()

# Correct acceleration data along z-axis by subtracting gravitational acceleration
accel_data["accel_z_corrected"] = accel_data["dv3"] - g

# Plot acceleration data
plt.plot(accel_data["time"], accel_data["accel_z_corrected"], color="blue")
plt.xlabel("Time")
plt.ylabel("Acceleration in Z-axis")
plt.show()

# Filter data for time > 7 minutes
data_filtered = accel_data[accel_data["time"] > 7]
# Substract the time of the first observation from time column
data_filtered["time"] = data_filtered["time"] - data_filtered["time"].iloc[0]



# Plot filtered data
#plt.plot(data_filtered["time"], data_filtered["accel_z_corrected"], color="red")
plt.plot(data_filtered["time"], data_filtered["accel_z_corrected"], color="red")
plt.xlabel("Time")
plt.ylabel("Acceleration in Z-axis")
plt.show()


data_filtered = data_filtered.reset_index(drop=True)

# Calculate the sampling frequency
fs = 60

# Calculate the minimum distance between peaks
min_distance =math.ceil(fs * 0.1)

# Find all the valleys in the filtered data that are below -1.2

#valleys, _ = find_peaks(-data_filtered["accel_z_corrected"], height=1.2, distance=min_distance)

################################ a commenter si test 1
#valleys, _ = find_peaks(-data_filtered["accel_z_corrected"], height=0.5, distance=min_distance)
############################################################################
valleys, _ = find_peaks(-data_filtered["accel_z_corrected"], height=0.5, distance=min_distance)

# Plot the valleys on top of the filtered data
plt.plot(data_filtered["time"], data_filtered["accel_z_corrected"], color="red")
plt.plot(data_filtered["time"].iloc[valleys], data_filtered["accel_z_corrected"].iloc[valleys], "x", color="green")
plt.xlabel("Time")
plt.ylabel("Acceleration in Z-axis")
plt.show()



# Initialize stride column with zeros
data_filtered["stride"] = 0

# Set the first stride as 1 (beginning of first stride)
data_filtered.at[valleys[0], "stride"] = 1

# Assign stride number to each row based on the index of the previous stride
for i in range(1, len(valleys)):
    data_filtered["stride"].iloc[valleys[i-1]:valleys[i]] = i
    


# Count the number of strides
num_strides = data_filtered["stride"].max()

print("Number of strides:", num_strides)


# Calculate the total duration of the walk in seconds
total_duration = data_filtered["time"].iloc[-1]

# Calculate the number of strides per second
strides_per_second = num_strides * 2 / total_duration

print("Number of strides per second:", strides_per_second)



stride = data_filtered['stride'].to_numpy()
stride_change = np.diff(stride) != 0
stride_change = np.append(stride_change, True) # Add a value for the last stride