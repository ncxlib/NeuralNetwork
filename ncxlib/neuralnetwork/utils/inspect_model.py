import h5py

# use to check models attriubtes after saving it
def inspect_saved_model(filepath):
    h5_suffix = ".h5"
    final_path = filepath + h5_suffix
    with h5py.File(final_path, 'r') as f:
        print("Model Attributes:")
        for attr in f.attrs:
            print(f"{attr}: {f.attrs[attr]}")

        print("\nLayers and Neurons:")
        for key in f.keys():
            dataset = f[key]
            if dataset.shape == ():  
                print(f"{key}: {dataset[()]}")
            else:
                print(f"{key}: {dataset[:]}")