import numpy as np

# Load the CSV file
data = np.loadtxt('transit.csv', skiprows=1, delimiter=',', dtype=int)
print('run')

# Save the data to a .npy file
np.save('transit.npy', data)

print('len np',len(data))