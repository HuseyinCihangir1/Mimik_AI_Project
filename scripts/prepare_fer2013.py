import os
import pandas as pd
import numpy as np
from PIL import Image

# PATHS
RAW_PATH = "data/raw/fer2013.csv"
OUTPUT_DIR = "data/processed"

train_dir = os.path.join(OUTPUT_DIR, "train")
val_dir = os.path.join(OUTPUT_DIR, "valid")
test_dir = os.path.join(OUTPUT_DIR, "test")

for d in [train_dir, val_dir, test_dir]:
    os.makedirs(d, exist_ok=True)

# LOAD CSV
df = pd.read_csv(RAW_PATH)

emotion_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

# LOOP
for i, row in df.iterrows():
    pixels = np.array(row["pixels"].split(), dtype="uint8")
    img = pixels.reshape(48, 48)

    label = emotion_map[row["emotion"]]
    usage = row["Usage"]

    if usage == "Training":
        save_path = os.path.join(train_dir, label)
    elif usage == "PublicTest":
        save_path = os.path.join(val_dir, label)
    else:
        save_path = os.path.join(test_dir, label)

    os.makedirs(save_path, exist_ok=True)

    img = Image.fromarray(img)
    img.save(os.path.join(save_path, f"{i}.jpg"))

print("✅ Dataset hazırlandı!")