def apply_fine_tuning(model, unfreeze_layers=20, learning_rate=1e-5):
    """Modelin son katmanlarını eğitime açar."""
    # Tüm katmanları önce aç
    model.trainable = True
    
    # Baştaki katmanları tekrar dondur (sadece son kısımlar eğitilsin)
    # ResNet-50 yaklaşık 175 katmandır.
    for layer in model.layers[:-unfreeze_layers]:
        layer.trainable = False
        
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model