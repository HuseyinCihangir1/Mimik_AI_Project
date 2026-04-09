import os
import sys
import tensorflow as tf

# --- GPU YÖNETİMİ ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU Aktif!")
    except RuntimeError as e:
        print(e)

# 1. Proje Yollarını Ayarla
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(BASE_DIR)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from src.data_loader import get_data_generators
from src.train import train_model
from src.eval import plot_history, evaluate_model

# --- CONFIG ---
DATA = os.path.join(BASE_DIR, "data", "processed")
EVAL = os.path.join(BASE_DIR, "evaluation", "stage3_finetuning")
# ÖNCEKİ MODELİ YÜKLÜYORUZ (Stage 2'den gelen ağırlıklar temelimiz olacak)
PREV_MODEL_PATH = os.path.join(BASE_DIR, "models", "stage2_aug.keras")
NEW_MODEL_PATH = os.path.join(BASE_DIR, "models", "stage3_finetuned.keras")
os.makedirs(EVAL, exist_ok=True)

print("🚀 Stage 3: Full Fine-Tuning Deneyi Başlıyor...")

# Veri yükleme (Augmentation'ı bu aşamada koruyoruz)
aug_params = {
    "rotation_range": 10,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "horizontal_flip": True,
    "fill_mode": "nearest"
}
train_gen, valid_gen, test_gen = get_data_generators(DATA, augmentation_params=aug_params)

# --- MODEL STRATEJİSİ ---
with tf.device('/GPU:0'):
    # Stage 2'de eğittiğimiz modeli yüklüyoruz
    if os.path.exists(PREV_MODEL_PATH):
        model = load_model(PREV_MODEL_PATH)
        print("📥 Stage 2 modeli yüklendi, üzerine inşa ediliyor.")
    else:
        print("⚠️ Stage 2 modeli bulunamadı! Sıfırdan başlanıyor (Önerilmez).")
        from src.model import base_model
        model = base_model(model_type="resnet50")

    # KRİTİK ADIM: Katmanları daha fazla açıyoruz. 
    # ResNet-50'nin son 50 katmanını eğitime açalım (Blok 4 ve 5)
    for layer in model.layers[-50:]:
        layer.trainable = True

    optimizer = Adam(learning_rate=1e-5) #L.R.'i dusuk tutuyoruz 
    
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # EĞİTİM (Daha sabırlı bir eğitim için Epoch'u 20'e cıkarrdık)
    history = train_model(model, train_gen, valid_gen, epochs=20, model_path=NEW_MODEL_PATH)

# Kaydetme
model.save(NEW_MODEL_PATH, include_optimizer=False)

# Metrikler
plot_history(history, EVAL)
evaluate_model(model, test_gen, EVAL)

print(f"\n✅ Stage 3 tamamlandı!  {EVAL}")