
import os
import numpy as np
import matplotlib.pyplot as plt

path = './data'
actors = np.load(os.path.join(path, 'actors.npy'))
unique = np.unique(actors)

print(f"Total entries: {len(actors)}")
print(f"Unique Actors found: {len(unique)}")
print(f"Actor IDs: {unique}")

# Load the saved data
x = np.load(f'{path}/features.npy')
y = np.load(f'{path}/labels.npy')

print(f"Data Shape: {x.shape}") 
# Should be (12960, 128, 130, 1) or similar

# Visual Check: Plot the first spectrogram
plt.imshow(x[0].reshape(128, -1), aspect='auto', origin='lower')
plt.title(f"Label: {y[0]}")
plt.colorbar()
plt.show()
