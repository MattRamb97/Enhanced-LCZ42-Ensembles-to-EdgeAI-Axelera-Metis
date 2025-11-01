import h5py

path = "tda_MS_features_bsrnet2x.h5"

with h5py.File(path, "r") as f:
    print(f"\n[INFO] HDF5 structure of {path}\n")
    
    def explore(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"📂 Dataset: /{name}")
            print(f"    shape: {obj.shape}")
            print(f"    dtype: {obj.dtype}")
            if obj.compression:
                print(f"    compression: {obj.compression}")
            if obj.chunks:
                print(f"    chunks: {obj.chunks}")
            if obj.attrs:
                print(f"    attrs: {dict(obj.attrs)}")
        elif isinstance(obj, h5py.Group):
            print(f"📁 Group: /{name}")
            if obj.attrs:
                print(f"    attrs: {dict(obj.attrs)}")

    f.visititems(explore)