from faster_whisper import download_model
import os

# Directory where you want to save models
models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)

print(f"🔽 Downloading 'base' model into {models_dir}...")
download_model("base", models_dir)
print("✅ Model downloaded successfully!")
