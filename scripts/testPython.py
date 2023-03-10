import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
#from scipy import signal
import math
#from scipy.integrate import simps
# Read data from csv file
accel_data = pd.read_csv("../data/test_4_kyks.csv", header=0, skiprows=11)


############################################ a commenter si test1
#accel_data = pd.read_csv("../data/test6_guill.csv", header=0, skiprows=1)
accel_data = accel_data.iloc[:, :-1]
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
data_filtered = accel_data[accel_data["time"] > 2]
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
if(len(valleys) < 10):
    valleys, _ = find_peaks(-data_filtered["accel_z_corrected"], height=0.5, distance=min_distance)
elif(len(valleys) < 10):
    valleys, _ = find_peaks(data_filtered["accel_z_corrected"], height=0.1, distance=min_distance)


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
    



# ############ TRAITEMENT DES FOULEES POUR AVOIR LA COLONNE STRIDE PERTINENTE #################################################

# Enlever les lignes à zéro à la fin de stride
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


# Utiliser la fonction "update_zeros" pour mettre à jour la colonne "stride" dans le dataframe "data_filtered"
data_filtered = update_zeros(data_filtered, "stride")



stride = data_filtered['stride'].to_numpy()
stride_change = np.diff(stride) != 0
stride_change = np.append(stride_change, True) # Add a value for the last stride


# Multiplier chaque valeurs par 2

# fonction lambda qui multiplie chaque nombre par 2
mult_by_2 = lambda x: x * 2

# appliquer la fonction lambda à la colonne "colonne_nums" du dataframe
data_filtered["stride"] = data_filtered["stride"].apply(mult_by_2)


# Diviser chaque foulée en 2 foulées 

# Compter les itérations de chaque valeur dans la colonne "stride"
count_stride_iter = pd.DataFrame(data_filtered['stride'].value_counts())

# Renommer les colonnes pour plus de clarté
count_stride_iter = count_stride_iter.rename(columns={'stride': 'iterations'})
 
count_stride_iter = count_stride_iter.sort_index()


# Diviser par 2 le nombre d'itérations
count_stride_iter['iterations'] = count_stride_iter['iterations'] / 2

count_stride_iter.reset_index(inplace=True)


# doubler le nombre de lignes
count_stride_iter = pd.concat([count_stride_iter, count_stride_iter])
# réinitialiser l'index
count_stride_iter = count_stride_iter.reset_index(drop=True)
# obtenir l'indice de la dernière ligne
last_row_index = count_stride_iter.iloc[-1].name

# Permet d'arrondir à l'entier supérieur et inférieur après la division par 2 pour garder le même nombre de valeurs
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


############ FIN TRAITEMENT DES FOULEES POUR AVOIR LA COLONNE STRIDE PERTINENTE #################################################




# Count the number of strides
num_strides = data_filtered["stride"].max()

print("Number of strides:", num_strides / 2)


# Calculate the total duration of the walk in seconds
total_duration = data_filtered["time"].iloc[-1]

# Calculate the number of strides per second
strides_per_second = num_strides / total_duration

print("Number of strides per second:", strides_per_second /2)



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
        contact_time = (stride_times["time_elapsed"].iloc[i] / 2.7) + oldTime
        oldTime += stride_times["time_elapsed"].iloc[i]
        contact_data = stride_0_data.loc[stride_0_data['time'] < contact_time]

        contact_duration = contact_time - contact_data.iloc[0]["time"]
        # Intégration de l'accélération pour calculer la vitesse instantanée
        dt = contact_duration / len(contact_data)
        v = np.cumsum(contact_data[["dv1", "dv2", "accel_z_corrected"]], axis=0) * dt

        # Calcul de la vitesse moyenne sur le contact au sol
        v_mean = np.mean(np.linalg.norm(v, axis=1))
        mean_speed_strides.append(v_mean)
    # Faire quelque chose avec stride_0_data
    else:
        print("stride_data est vide pour la valeur de i spécifiée.")

   

moyenne_speed_strides = sum(mean_speed_strides) / len(mean_speed_strides)
moyenne_speed_strides = moyenne_speed_strides * 100
print("La moyenne est :", moyenne_speed_strides)





                                 
