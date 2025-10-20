import matplotlib
matplotlib.use('MacOSX')

import matplotlib.pyplot as plt
import h5py
import numpy as np
from gtda.homology import CubicalPersistence
from gtda.diagrams import PersistenceImage

def read_patch(h5_path, index, modality="MS"):
    with h5py.File(h5_path, 'r') as f:
        data = f['sen2'][:] if modality == "MS" else f['sen1'][:]
        patch = data[index]  # Shape: [32, 32, channels]
        return patch.transpose(2, 0, 1)  # to CxHxW

def preprocess_band(X, band=2):
    """
    Takes a band from MS patch and scales it for TDA.
    Band index in MS is from 0 to 9. Band=2 is B4 (red).
    """
    img = X[band] / (2.8 / 255)
    return np.clip(img, 0, 255).astype(np.float64)

if __name__ == "__main__":
    h5_path = "../data/lcz42/training.h5"
    index = 37888  # Example patch
    band = 2       # B4 (red)

    X = read_patch(h5_path, index, modality="MS")
    img = preprocess_band(X, band)

    cp = CubicalPersistence(homology_dimensions=(0, 1))
    diagrams = cp.fit_transform(img[np.newaxis, :, :])

    # Vectorize with PersistenceImage
    pimg = PersistenceImage(
        sigma=0.5,                 # Gaussian smoothing
        n_bins=50,                 # 50x50 image per homology dim
        weight_function=None,      # No weighting (uniform)
    )
    features = pimg.fit_transform(diagrams) # shape: (1, n_homology, n_bins, n_bins)

    # Plot (homology dim 0 and 1)
    for i, dim in enumerate(pimg.homology_dimensions_):
        plt.figure()
        plt.imshow(features[0, i], cmap='hot')
        plt.title(f"Persistence Image — $H_{dim}$")
        plt.colorbar()
        plt.axis('off')
        plt.show()

    #print("Diagram shape:", diagrams.shape)
 
    #fig = CubicalPersistence.plot(diagrams, sample=0)
    #fig.show()

    #plt.imshow(img, cmap='gray') 
    #plt.title(f"Patch {index} — Band {band}")
    #plt.axis('off')
    #plt.show()

    