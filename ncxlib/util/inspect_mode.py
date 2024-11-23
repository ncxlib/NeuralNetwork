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
        
        num_layers = f.attrs.get("num_layers", 0)
        for i in range(num_layers):
            activation = f.attrs.get(f"layer_{i}_activation", "None")
            print(f"layer_{i}_activation: {activation}")
            
            weights_key = f"layer_{i}_weights"
            bias_key = f"layer_{i}_bias"
            
            if weights_key in f:
                print(f"{weights_key}: {f[weights_key][:]}")
            if bias_key in f:
                print(f"{bias_key}: {f[bias_key][:]}")