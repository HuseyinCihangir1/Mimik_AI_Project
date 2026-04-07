import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(model, train_gen, valid_gen, epochs=10, model_path="../models/best_model.keras"):
    """Modeli eğitir ve en iyi ağırlıkları kaydeder."""
    
    # Callback'ler: En iyi modeli kaydet ve gelişme yoksa durdur
    checkpoint = ModelCheckpoint(
        model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=epochs,
        callbacks=[checkpoint, early_stop]
    )
    
    return history