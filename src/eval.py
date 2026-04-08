import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

def plot_history(history, save_path):
    """Accuracy ve Loss grafiklerini çizer ve kaydeder."""
    plt.figure(figsize=(12, 4))
    
    # Accuracy Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim')
    plt.plot(history.history['val_accuracy'], label='Doğrulama')
    plt.title('Model Doğruluğu (Accuracy)')
    plt.legend()
    
    # Loss Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim')
    plt.plot(history.history['val_loss'], label='Doğrulama')
    plt.title('Model Kaybı (Loss)')
    plt.legend()
    
    plt.savefig(os.path.join(save_path, "learning_curves.png"))
    plt.close()

def evaluate_model(model, test_gen, save_path):
    """Confusion Matrix ve Detaylı Rapor oluşturur."""
    test_gen.reset()
    predictions = model.predict(test_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    # 1. Raporu Yazdır ve Kaydet
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print("\n--- SINIFLANDIRMA RAPORU ---")
    print(report)
    
    with open(os.path.join(save_path, "performance_report.txt"), "w") as f:
        f.write(report)

    # 2. Karmaşıklık Matrisi (Confusion Matrix)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Duygu Analizi Karmaşıklık Matrisi')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.close()