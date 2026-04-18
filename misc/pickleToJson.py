import json
import pickle

# Load your existing label encoder
with open('web_demo/ALPHANUM/label_encoder_ALPHANUM.pickle', 'rb') as f:
    le = pickle.load(f)

# Export to JSON
with open('web_demo/ALPHANUM/labels_ALPHANUM.json', 'w') as f:
    json.dump(le.classes_.tolist(), f)