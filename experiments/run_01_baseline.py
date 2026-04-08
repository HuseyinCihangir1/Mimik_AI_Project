import os
import sys
from tensorflow.keras.optimizers import Adam

# Kök dizini tanıtmak için (ModuleNotFoundError önleyici)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import get_data_generators
from src.model import base_model
from src.train import train_model
from src.eval import plot_history, evaluate_model
from src.utils import save_class_indices

# --- CONFIG (Yollar Sabitlendi) ---
# Mevcut dosyanın (experiments/run_01...) iki üstü ana kök dizindir.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation", "stage1_baseline")

# Klasörleri oluştur (Bu satır çok kritik!)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "stage1_baseline.keras")

# 1. DATA
train_gen, valid_gen, test_gen = get_data_generators(DATA)

# 2. MODEL
model = base_model(model_type="resnet50")
# Baseline için 1e-4 iyi bir değer, stabil öğrenme sağlar.
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# 3. TRAIN
print(f"🚀 Eğitim Başlıyor... Model buraya kaydedilecek: {MODEL_PATH}")
history = train_model(model, train_gen, valid_gen, epochs=10, model_path=MODEL_PATH)

# 4. EVAL & SAVE
save_class_indices(train_gen, EVAL_DIR)
plot_history(history, EVAL_DIR)

# Son Epoch değerlerini yazdır
final_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print("\n" + "="*30)
print(f"🎯 EĞİTİM TAMAMLANDI")
print(f"📈 Son Eğitim Doğruluğu: %{final_acc*100:.2f}")
print(f"📉 Son Doğrulama Doğruluğu: %{final_val_acc*100:.2f}")
print("="*30)

# Değerlendirmeyi çalıştır (Metrikleri hem basar hem kaydeder)
evaluate_model(model, test_gen, EVAL_DIR)
print(f"✅ Her şey kaydedildi! Klasörleri kontrol et: \n1. {MODELS_DIR}\n2. {EVAL_DIR}")