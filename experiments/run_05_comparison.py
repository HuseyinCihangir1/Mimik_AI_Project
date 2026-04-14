import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import load_model
from src.data_loader import get_data_generators
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --- AYARLAR ---
DATA_DIR = "data/processed"
FINAL_MODEL_PATH = "models/stage4_opt.keras" # En iyi modelimiz
COMPARISON_PATH = "evaluation/stage5_comparison"
os.makedirs(COMPARISON_PATH, exist_ok=True)

# 1. TEST VERİSİNİ YÜKLE
_, _, test_gen = get_data_generators(DATA_DIR)

# 2. EN İYİ MODELİ YÜKLE
model = load_model(FINAL_MODEL_PATH)

# 3. KARŞILAŞTIRMALI ANALİZ (Duygu Bazlı)
test_gen.reset()
predictions = model.predict(test_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes
labels = list(test_gen.class_indices.keys())

# --- GÖRSELLEŞTİRME 1: Başarı Özeti ---
report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
accuracies = [report[label]['f1-score'] for label in labels]

plt.figure(figsize=(10, 6))
sns.barplot(x=labels, y=accuracies, palette="viridis")
plt.title("Duygu Bazlı F1-Skor Başarısı (Stage 5)")
plt.ylabel("Skor (0.0 - 1.0)")
plt.savefig(f"{COMPARISON_PATH}/f1_scores.png")
print("✅ Duygu bazlı başarı grafiği kaydedildi.")

# --- GÖRSELLEŞTİRME 2: Örnek Tahminler (Showcase) ---
# Test setinden 9 örnek fotoğraf alıp tahminlerini basalım
plt.figure(figsize=(12, 12))
x_batch, y_batch = next(test_gen)
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_batch[i]) # ResNet preprocess_input yüzünden renkler garip görünebilir
    
    true_label = labels[np.argmax(y_batch[i])]
    pred_label = labels[np.argmax(model.predict(x_batch[i:i+1]))]
    
    color = "green" if true_label == pred_label else "red"
    plt.title(f"G: {true_label}\nT: {pred_label}", color=color)
    plt.axis("off")

plt.tight_layout()
plt.savefig(f"{COMPARISON_PATH}/sample_predictions.png")
print("✅ Örnek tahmin görseli kaydedildi.")

# --- STAGE GELİŞİM TABLOSU (Manuel Veri Girişi) ---
stages = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
results = [61, 63, 70, 75] # Senin aldığın sonuçları buraya yaz

plt.figure(figsize=(8, 5))
plt.plot(stages, results, marker='o', linestyle='-', color='b', linewidth=2)
plt.title("Proje Gelişim Süreci (Doğruluk Artışı)")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.savefig(f"{COMPARISON_PATH}/development_curve.png")
print("✅ Gelişim eğrisi kaydedildi.")