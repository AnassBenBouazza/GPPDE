# Hydrological constants

theta_r = 0.075
theta_s = 0.287
k_s = 1.5

# Van Genuchten model parameters

alpha = 0.05
n = 2
m = 1 - 1 / n

param = [theta_r, theta_s, k_s, alpha, n, m]

# Richards equation resolution parameters

h_t0 = -65
h_z0 = -65
h_z60 = -20.7

dz = 1
dt = 1