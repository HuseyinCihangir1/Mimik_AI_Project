import os
from tensorflow.keras.optimizers import Adam
from src.data_loader import get_data_generators
from src.model import base_model
from src.train import train_model
from src.eval import plot_history, evaluate_model
from src.utils import save_class_indices

# --- CONFIG(Ayarlamalar) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE_DIR, "data", "processed")
EVAL = "../evaluation/stage1_baseline"
MODEL_PATH = "../models/stage1_baseline.keras"
os.makedirs(EVAL, exist_ok=True)

# 1. DATA: Train, Valid ve Test setlerini yükle 
train_gen, valid_gen, test_gen = get_data_generators(DATA)

# 2. MODEL: ResNet-50 Baseline (Transfer Learning) 
model = base_model(model_type="resnet50")
model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

# 3. TRAIN: Eğitimi başlat ve en iyiyi kaydet 
history = train_model(model, train_gen, valid_gen, epochs=10, model_path=MODEL_PATH)

# 4. EVAL: Sonuçları (Grafik, Rapor, Matris) kaydet 
save_class_indices(train_gen, EVAL)
plot_history(history, EVAL)
evaluate_model(model, test_gen, EVAL)

print(f"✅ İşlem tamam! Sonuçlar burada: {EVAL}")