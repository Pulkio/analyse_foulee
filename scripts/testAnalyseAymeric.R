library(dplyr)
library(ggplot2)
library(pracma)
library(data.table)
library(lubridate)
library(signal)
library(wavethresh)



accel_data <- fread("../data/test6.csv", header=TRUE, sep=",", dec=".")
colnames(accel_data)[colnames(accel_data) == "dv[1]"] <- "dv1"
colnames(accel_data)[colnames(accel_data) == "dv[2]"] <- "dv2"
colnames(accel_data)[colnames(accel_data) == "dv[3]"] <- "dv3"

# Calculer le temps en minutes
accel_data$time <- accel_data$PacketCounter / 60

# a ignorer
#accel_data <- read.csv("../data/test1.csv", header = TRUE, skip =11)


#colnames(accel_data)[colnames(accel_data) == "dv.1."] <- "dv1"
#colnames(accel_data)[colnames(accel_data) == "dv.2."] <- "dv2"
#colnames(accel_data)[colnames(accel_data) == "dv.3."] <- "dv3"


#accel_data <- select(accel_data, dv1, dv2, dv3, time, dq_W, dq_X, dq_Y, dq_Z)
#accel_data <- select(accel_data, dv1, dv2, dv3, time, dq_W, dq_X, dq_Y, dq_Z)

# fin de a ignorer

# Calculer la composante d'accélération gravitationnelle le long de l'axe z car correspond à l'axe le plus impacté
#par la gravité
g <- mean(accel_data$dv3)

# Enlever la composante d'accélération gravitationnelle des données d'accélération le long de l'axe z
accel_data <- mutate(accel_data, accel_z_corrected = accel_data$dv3 - g)

# Calculer le temps entre les échantillons en secondes
delta_t <- 1/60 

# Calculer la vitesse instantanée le long des axes x, y et z en intégrant deux fois l'accélération
accel_data <- mutate(accel_data, 
                     vel_x = cumsum(accel_data$dv1 * delta_t),
                     vel_y = cumsum(accel_data$dv2 * delta_t),
                     vel_z = cumsum(accel_data$accel_z_corrected * delta_t))

# Calculer le déplacement cumulatif le long des axes x, y et z en intégrant la vitesse
accel_data <- mutate(accel_data, 
                     disp_x = cumsum(accel_data$vel_x * delta_t),
                     disp_y = cumsum(accel_data$vel_y * delta_t),
                     disp_z = cumsum(accel_data$vel_z * delta_t))

# Calculer la magnitude du vecteur de déplacement total
accel_data <- mutate(accel_data, 
                     disp_mag = sqrt(disp_x^2 + disp_y^2 + disp_z^2))

# Calculer la vitesse instantanée
accel_data <- mutate(accel_data, 
                     vel_mag = c(0, diff(accel_data$disp_mag)) / delta_t)

# Tracer le graphique de la vitesse instantanée en fonction du temps
ggplot(accel_data, aes(x = time, y = vel_mag)) + 
  geom_line() + 
  labs(x = "Time (s)", y = "Instantaneous Velocity (m/s)", 
       title = "Instantaneous Velocity vs. Time")

# Calculer la distance totale parcourue
total_distance <- accel_data$disp_mag[nrow(accel_data)]

# Afficher la distance totale parcourue
print(paste("Total distance traveled:", round(total_distance, 2), "meters"))




















