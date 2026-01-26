import numpy as np
import json
import os

# Target folder
folder = "web_demo"

def force_clean_json(filename):
    npy_path = os.path.join(folder, filename + ".npy")
    json_path = os.path.join(folder, filename + ".json")
    
    if os.path.exists(npy_path):
        # Load data and ensure it is a flat 1D list
        data = np.load(npy_path).flatten().tolist()
        
        # Overwrite the JSON with ONLY that list
        with open(json_path, 'w') as f:
            json.dump(data, f)
        print(f"✅ Re-generated clean JSON: {json_path}")
    else:
        print(f"❌ Could not find {npy_path}")

force_clean_json("norm_mean")
force_clean_json("norm_std")