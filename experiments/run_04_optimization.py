import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from src.model import base_model
from src.data_loader import get_data_generators
from src.eval import evaluate_model
import os

# --- AYARLAR ---
DATA_DIR = "data/processed"
SAVE_PATH = "models/stage4_opt.keras"
EVAL_PATH = "evaluation/stage4_opt"
os.makedirs(EVAL_PATH, exist_ok=True)

# 1. VERİ YÜKLEME (Stage 2'deki Augmentation'ı burada da kullanıyoruz)
aug_params = {
    "rotation_range": 20,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "shear_range": 0.2,
    "zoom_range": 0.2,
    "horizontal_flip": True,
    "fill_mode": "nearest"
}
train_gen, valid_gen, test_gen = get_data_generators(DATA_DIR, augmentation_params=aug_params)

# 2. MODELİ HAZIRLA (Fine-tuning yapılmış haliyle başla)
model = base_model(model_type="resnet50")
# ResNet'in son bloklarını eğitime açıyoruz (Stage 3 mantığı)
model.trainable = True
for layer in model.layers[:-30]: # Son 30 katman açık kalsın
    layer.trainable = False

# 3. OPTİMİZASYON ARAÇLARI (Stage 4'ün kalbi)
# Doğrulama kaybı (val_loss) iyileşmediğinde öğrenme hızını %20'sine düşürür
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=3, 
    min_lr=1e-6, 
    verbose=1
)

# Model artık gelişmiyorsa eğitimi vaktinde durdurur
early_stopper = EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    restore_best_weights=True
)

# 4. DERLEME VE EĞİTİM
model.compile(
    optimizer=Adam(learning_rate=1e-4), # Düşük hızla başla
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=30, # Callback'ler sayesinde 30 yazmak güvenli
    callbacks=[lr_reducer, early_stopper]
)

# 5. KAYDET VE DEĞERLENDİR
model.save(SAVE_PATH)
evaluate_model(model, test_gen, EVAL_PATH)