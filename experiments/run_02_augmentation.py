import os
import sys
import tensorflow as tf

# --- GPU ZORLAMASI VE BELLEK YÖNETİMİ ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU Aktif Edildi!")
    except RuntimeError as e:
        print(f"❌ GPU Hatası: {e}")

# 1. Yolları Ayarla
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(BASE_DIR)

from tensorflow.keras.optimizers import Adam
from src.data_loader import get_data_generators
from src.model import base_model
from src.train import train_model
from src.eval import plot_history, evaluate_model

# --- CONFIG ---
DATA = os.path.join(BASE_DIR, "data", "processed")
EVAL = os.path.join(BASE_DIR, "evaluation", "stage2_augmentation")
MODEL_PATH = os.path.join(BASE_DIR, "models", "stage2_aug.keras")
os.makedirs(EVAL, exist_ok=True)

# Hafifletilmiş Augmentation (Başarıyı korumak için oranları düşürdük)
aug_params = {
    "rotation_range": 10,      # 15'ten 10'a düşürdük
    "width_shift_range": 0.1,  
    "height_shift_range": 0.1, 
    "horizontal_flip": True,   
    "fill_mode": "nearest"     # Boşlukları doldurma modu
}

print("🚀 Stage 2: Optimize Edilmiş Veri Artırma Deneyi Başlıyor...")

# Veri yükleme
train_gen, valid_gen, test_gen = get_data_generators(DATA, augmentation_params=aug_params)

# --- MODEL STRATEJİSİ ---
with tf.device('/GPU:0'): # İşlemleri GPU'ya zorla
    model = base_model(model_type="resnet50")
    
    # KRİTİK DOKUNUŞ: ResNet'in son birkaç katmanını eğitime açıyoruz (Fine-tuning başlangıcı)
    # Bu sayede model augmentation ile gelen değişiklikleri öğrenebilir.
    for layer in model.layers[-20:]: 
        layer.trainable = True

    # Daha düşük bir Learning Rate (0.0001) ile hassas öğrenme sağlıyoruz
    optimizer = Adam(learning_rate=1e-4) 
    
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # EĞİTİM (20 Epoch)
    history = train_model(model, train_gen, valid_gen, epochs=20, model_path=MODEL_PATH)

# Kaydetme (Optimize edilmiş hali)
model.save(MODEL_PATH, include_optimizer=False)

# Metrikler
plot_history(history, EVAL)
evaluate_model(model, test_gen, EVAL)

print(f"\n✅ Stage 2 tamamlandı! %60 barajını zorluyoruz. Sonuçlar: {EVAL}")