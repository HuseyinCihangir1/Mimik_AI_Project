from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def base_model(
        model_type="resnet50", 
        num_classes=7, 
        input_shape=(224, 224, 3)): 
    
    if model_type == "resnet50":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
            
    elif model_type == "efficientnetb0":
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    
    # Baseline için dondurma işlemi
    base_model.trainable = False 
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)
    
    return Model(inputs=base_model.input, outputs=output)