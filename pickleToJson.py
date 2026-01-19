import json
import pickle

# Load your existing label encoder
with open('web_demo/label_encoder.pickle', 'rb') as f:
    le = pickle.load(f)

# Export to JSON
with open('web_demo/labels.json', 'w') as f:
    json.dump(le.classes_.tolist(), f)