import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Duygu Analiz Asistanı v1.0")
        self.root.geometry("500x650")
        
        # Model
        try:
            self.model = tf.keras.models.load_model("models/stage4_opt.keras")
            self.labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        except Exception as e:
            messagebox.showerror("Hata", f"Model yüklenemedi: {e}")

        # Gui Kismi
        self.label_title = tk.Label(root, text="Yüz Fotoğrafı Seçin", font=("Arial", 16, "bold"))
        self.label_title.pack(pady=20)

        self.canvas = tk.Canvas(root, width=300, height=300, bg="gray")
        self.canvas.pack()

        self.btn_select = tk.Button(root, text="Görsel Seç", command=self.upload_image, font=("Arial", 12))
        self.btn_select.pack(pady=20)

        self.result_text = tk.Label(root, text="", font=("Arial", 14, "italic"), fg="blue")
        self.result_text.pack(pady=10)

    #Cihazdan gorsel yukleme
    def upload_image(self): 
        file_path = filedialog.askopenfilename()
        if file_path:
            # Görseli Göster
            img = Image.open(file_path).convert('RGB')
            img_display = img.resize((300, 300))
            self.photo = ImageTk.PhotoImage(img_display)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            # Tahmin Kismi
            self.predict_emotion(img)

    def predict_emotion(self, img):
        img_reshaped = img.resize((224, 224))
        img_array = np.array(img_reshaped)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = self.model.predict(img_array)
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        result_str = f"Tahmin: {self.labels[class_idx]}\nDoğruluk: %{confidence:.2f}"
        self.result_text.config(text=result_str)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()