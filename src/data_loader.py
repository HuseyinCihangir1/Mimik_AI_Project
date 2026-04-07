import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators( # train/test/valid için; Epoch'lar icin batch'ler(veri parçaları) üretir. 
        data_dir, 
        img_size=(224, 224), 
        batch_size=32,
        augmentation_params=None):
    
    # 1. Train Generator(Eğitim)
    if augmentation_params:
        # Stage 2'de buraya sözlük olarak parametre göndereceğiz
        train_datagen = ImageDataGenerator(**augmentation_params)
    else:
        train_datagen = ImageDataGenerator(rescale=1./255) 

    # 2. Validation(Doğrulama)
    # Sadece normalizasyon uygulanır 
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"), 
        target_size=img_size, 
        batch_size=batch_size, 
        class_mode="categorical"
    )
    
    valid_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "valid"), 
        target_size=img_size, 
        batch_size=batch_size, 
        class_mode="categorical"
    )

    # Test = Model hiç görmediği verilerle test edilir.
    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "test"), 
        target_size=img_size, 
        batch_size=batch_size, 
        class_mode="categorical",
        shuffle=False  # Metriklerin doğru eşleşmesi için testte karıştırma yapılmaz
    )

    return train_gen, valid_gen, test_gen