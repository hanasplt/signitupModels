from tensorflow.keras.models import load_model
from tensorflowjs.converters import save_keras_model
import os

model_path = r"C:/Users/Janna/Documents/GitHub/signitupModels/trained_models/gesture_lstm_model.h5"
out_dir    = "tfjs_model"          # will be created

model = load_model(model_path)
save_keras_model(model, out_dir)
print("✅ TF-JS model saved to", os.path.abspath(out_dir))