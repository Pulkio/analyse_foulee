library(ggplot2)
library(pracma)
library(RWeka)


accel_data <- read.csv("../data/test1.csv", header = TRUE, skip =11)


colnames(accel_data)[colnames(accel_data) == "dv.1."] <- "dv1"
colnames(accel_data)[colnames(accel_data) == "dv.2."] <- "dv2"
colnames(accel_data)[colnames(accel_data) == "dv.3."] <- "dv3"

accel_data$time <- accel_data$PacketCounter / 60

# Calculer la composante d'accélération gravitationnelle le long de l'axe z car correspond à l'axe le plus impacté
#par la gravité
g <- mean(accel_data$dv3)

# Enlever la composante d'accélération gravitationnelle des données d'accélération le long de l'axe z
accel_data <- mutate(accel_data, accel_z_corrected = accel_data$dv3 - g)



ggplot(accel_data, aes(x = time, y = accel_z_corrected)) +
  geom_line(color = "blue") +
  xlab("Accélération en z") +
  ylab("Temps")


# Filtrer les données en ne gardant que celles où time > 7
data_filtered <- subset(accel_data, time > 7)

ggplot(data_filtered, aes(x = time, y = accel_z_corrected)) +
  geom_line(color = "red") +
  xlab("Accélération en z") +
  ylab("Temps")



# Extraire la colonne accel_z_corrected
accel_z <- data_filtered$accel_z_corrected

# Normaliser la colonne accel_z_corrected
accel_z_norm <- scale(accel_z)

# Calculer la transformée de Fourier de la colonne accel_z_corrected
fft_z <- fft(accel_z_norm)
fft_z_abs <- abs(fft_z)
fft_z_abs[1] <- 0  # supprimer la composante DC

# Extraire les fréquences dominantes
freqs <- seq(0, length(fft_z_abs)-1)
peak_freqs <- freqs[fft_z_abs > quantile(fft_z_abs, 0.95)]

# Appliquer l'algorithme de clustering spectral aux fréquences dominantes
library(RWeka)
period_clusters <- spectralClustering(fft_z_abs, k = 3)

# Trouver la période dominante pour chaque cluster
periods <- numeric(length(period_clusters))
for (i in 1:length(period_clusters)) {
  cluster_indices <- which(period_clusters == i)
  peak_cluster_freqs <- peak_freqs[peak_freqs %in% cluster_indices]
  periods[i] <- 1 / mean(peak_cluster_freqs)
}

# Afficher les résultats
cat("Périodes trouvées : ", round(periods, 2))