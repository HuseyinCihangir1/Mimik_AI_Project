import tensorflow as tf

# 1. Modeli yükle
model = tf.keras.models.load_model("../models/stage4_opt.keras")
# 2. TFLite dönüştürücüyü hazırla
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Modeli optimize et (Boyutu küçültür: ~100MB -> ~25MB)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Dönüştür ve Kaydet
tflite_model = converter.convert()
with open("models/model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model başarıyla Tflite formatına çevrildi: models/model.tflite")