import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

def get_data_generators(# train/test/valid için; Epoch'lar icin batch'ler(veri parçaları) üretir. 
        data_dir, 
        img_size=(224, 224), 
        batch_size=32,
        augmentation_params=None):
    
    # 1. Train Generator
    if augmentation_params:
        # Stage 2 için hazır
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            **augmentation_params
        )
    else:
        # Baseline: Sadece ResNet'in beklediği normalizasyon
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) 

    # 2. Validation & Test
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"), 
        target_size=img_size, batch_size=batch_size, class_mode="categorical"
    )
    
    valid_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "valid"), 
        target_size=img_size, batch_size=batch_size, class_mode="categorical"
    )

    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "test"), 
        target_size=img_size, batch_size=batch_size, class_mode="categorical",
        shuffle=False
    )

    return train_gen, valid_gen, test_gen