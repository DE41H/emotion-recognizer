
import os
import numpy as np

path = './data'
actors = np.load(os.path.join(path, 'actors.npy'))
unique = np.unique(actors)

print(f"Total entries: {len(actors)}")
print(f"Unique Actors found: {len(unique)}")
print(f"Actor IDs: {unique}")
