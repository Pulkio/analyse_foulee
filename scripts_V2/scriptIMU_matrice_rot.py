# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:05:32 2023

@author: guill
"""
#pip install numpy-quaternion


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
#from scipy import signal
import math
import quaternion



#♣accel_data = pd.read_csv("../data/test_5_kyks.csv", header=0, skiprows=11)

############################################ a commenter si test1
accel_data = pd.read_csv("../data/test7_tita.csv", header=0, skiprows=1)
accel_data = accel_data.iloc[:, :-1]
#################################################################

# Définir les angles d'Euler et les accélérations locales pour chaque ligne
#euler_x = accel_data['Euler_X']
# euler_y = accel_data['Euler_Y']
# euler_z = accel_data['Euler_Z']
# dv1 = accel_data['dv[1]']
# dv2 = accel_data['dv[2]']
# dv3 = accel_data['dv[3]']

# # Initialiser la matrice de rotation globale
# R_totale = np.zeros((3,3))

# # Calculer la matrice de rotation pour chaque ligne
# for i in range(len(accel_data)):
    
#     # Calculer la matrice de rotation pour la ligne i
#     Rx = np.array([[1, 0, 0],
#                    [0, np.cos(euler_x[i]), -np.sin(euler_x[i])],
#                    [0, np.sin(euler_x[i]), np.cos(euler_x[i])]])
#     Ry = np.array([[np.cos(euler_y[i]), 0, np.sin(euler_y[i])],
#                    [0, 1, 0],
#                    [-np.sin(euler_y[i]), 0, np.cos(euler_y[i])]])
#     Rz = np.array([[np.cos(euler_z[i]), -np.sin(euler_z[i]), 0],
#                    [np.sin(euler_z[i]), np.cos(euler_z[i]), 0],
#                    [0, 0, 1]])
    
#     # Calculer la matrice de rotation totale pour la ligne i
#     R_totale = Rz @ Ry @ Rx
    
#     # Calculer les accélérations globales pour la ligne i
#     a_imu = np.array([dv1[i], dv2[i], dv3[i]])
#     a_global = np.dot(R_totale, a_imu)
    
#     # Ajouter les nouvelles colonnes d'accélération au dataframe pour la ligne i
#     accel_data.loc[i, 'dv[1]'] = a_global[0]
#     accel_data.loc[i, 'dv[2]'] = a_global[1]
#     accel_data.loc[i, 'dv[3]'] = a_global[2]


# Déterminer si accel_data contient des angles d'Euler ou des quaternions
if 'Quat_W' in accel_data.columns:
    quat_mode = True
    q0 = accel_data['Quat_W']
    q1 = accel_data['Quat_X']
    q2 = accel_data['Quat_Y']
    q3 = accel_data['Quat_Z']

elif 'dq_W' in accel_data.columns:
        quat_mode = True
        q0 = accel_data['dq_W']
        q1 = accel_data['dq_X']
        q2 = accel_data['dq_Y']
        q3 = accel_data['dq_Z']
else:
    quat_mode = False
    euler_x = accel_data['Euler_X']
    euler_y = accel_data['Euler_Y']
    euler_z = accel_data['Euler_Z']

# Définir les accélérations locales pour chaque ligne
dv1 = accel_data['dv[1]']
dv2 = accel_data['dv[2]']
dv3 = accel_data['dv[3]']

# Initialiser la matrice de rotation globale
R_totale = np.zeros((3,3))

# Calculer la matrice de rotation pour chaque ligne
for i in range(len(accel_data)):
    
    # Calculer la matrice de rotation pour la ligne i
    if quat_mode:
        q = np.array([q0[i], q1[i], q2[i], q3[i]])
        R = quaternion.as_rotation_matrix(quaternion.from_float_array(q))
    else:
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(euler_x[i]), -np.sin(euler_x[i])],
                       [0, np.sin(euler_x[i]), np.cos(euler_x[i])]])
        Ry = np.array([[np.cos(euler_y[i]), 0, np.sin(euler_y[i])],
                       [0, 1, 0],
                       [-np.sin(euler_y[i]), 0, np.cos(euler_y[i])]])
        Rz = np.array([[np.cos(euler_z[i]), -np.sin(euler_z[i]), 0],
                       [np.sin(euler_z[i]), np.cos(euler_z[i]), 0],
                       [0, 0, 1]])
        R = Rz @ Ry @ Rx
    
    # Calculer les accélérations globales pour la ligne i
    a_imu = np.array([dv1[i], dv2[i], dv3[i]])
    a_global = np.dot(R, a_imu)
    
    # Ajouter les nouvelles colonnes d'accélération au dataframe pour la ligne i
    accel_data.loc[i, 'dv[1]'] = a_global[0]
    accel_data.loc[i, 'dv[2]'] = a_global[1]
    accel_data.loc[i, 'dv[3]'] = a_global[2]



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



valleys, _ = find_peaks(-data_filtered["accel_z_corrected"], height=1, distance=min_distance)
if(len(valleys) < 300):
    valleys, _ = find_peaks(-data_filtered["accel_z_corrected"], height=0.5, distance=min_distance)
if(len(valleys) < 5):
    #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    valleys, _ = find_peaks(-data_filtered["accel_z_corrected"], height=0.3, distance=min_distance)
if(len(valleys) < 5):
    valleys, _ = find_peaks(-data_filtered["accel_z_corrected"], height=0.1, distance=min_distance)
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
    






############ FIN TRAITEMENT DES FOULEES POUR AVOIR LA COLONNE STRIDE PERTINENTE #################################################




# Count the number of strides
num_strides = data_filtered["stride"].max()

print("Number of strides:", num_strides)


# Calculate the total duration of the walk in seconds
total_duration = data_filtered["time"].iloc[-1]

# Calculate the number of strides per second
strides_per_second = num_strides / total_duration

print("Number of strides per second:", strides_per_second)



##############################################################################################################################


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
#print(stride_times)

mean_speed_strides = []
max_stride = np.max(data_filtered["stride"]) # Valeur maximale de la colonne "stride"

oldTime = 0
for i in range(max_stride):
    #print(i)
    # Extraction des données de la première foulée
    stride_0_data = data_filtered[data_filtered["stride"] == i]
    
    if not stride_0_data.empty:
        # Sélectionner les données de la première moitié du contact au sol
        contact_time = (stride_times["time_elapsed"].iloc[i] /2.7) + oldTime
        oldTime += stride_times["time_elapsed"].iloc[i]
        contact_data = stride_0_data.loc[stride_0_data['time'] < contact_time]

        contact_duration = contact_time - contact_data.iloc[0]["time"]
        # Intégration de l'accélération pour calculer la vitesse instantanée
        dt = contact_duration / len(contact_data)
        v = np.cumsum(contact_data[["dv1", "dv2"]], axis=0) * dt

        # Calcul de la vitesse moyenne sur le contact au sol
        v_mean = np.mean(np.linalg.norm(v, axis=1))
        mean_speed_strides.append(v_mean)
    # Faire quelque chose avec stride_0_data
    else:
        print("stride_data est vide pour la valeur de i spécifiée.")

   

moyenne_speed_strides = sum(mean_speed_strides) / len(mean_speed_strides)
moyenne_speed_strides = moyenne_speed_strides * 100
print("La moyenne est :", moyenne_speed_strides)

