### Randomly sample files from a directory

import numpy as np
import os

# List all files in dir
files = os.listdir("C:/Users/.../Myfiles")

# Select 0.5 of the files randomly 
random_files = np.random.choice(files, int(len(files)*.5))

# Get the remaining files
other_files = [x for x in files if x not in random_files]

# Do something with the files
for x in random_files:
    print(x)
