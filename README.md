# Classification-of-Tomato-Ripeness
# Klasifikasi Kematangan Tomat

Proyek ini mengimplementasikan sistem klasifikasi otomatis untuk menentukan tingkat kematangan tomat menggunakan Deep Learning. Sistem ini dapat mengklasifikasikan tomat ke dalam 4 kategori: Damaged (Rusak), Old (Tua), Ripe (Matang), dan Unripe (Mentah).

## Dataset

Dataset terdiri dari 6,487 gambar tomat yang terbagi dalam 4 kelas:
- **Old**: 2,214 gambar
- **Ripe**: 2,195 gambar
- **Unripe**: 1,419 gambar
- **Damaged**: 659 gambar

### Pembagian Dataset
- **Training**: 64% (4,150 gambar)
- **Validation**: 16% (1,038 gambar)
- **Testing**: 20% (1,298 gambar)

## Model yang Diimplementasikan

### 1. CNN (Convolutional Neural Network)
Model CNN sederhana dengan arsitektur custom:
- 3 layer Conv2D (32, 64, 128 filters)
- MaxPooling2D setelah setiap Conv2D
- Fully connected layer dengan 128 neurons
- Dropout 0.5 untuk regularisasi

**Hasil:**
- Test Accuracy: **76%**
- Performa terbaik pada kelas Unripe (F1-score: 0.98)
- Performa terendah pada kelas Damaged (F1-score: 0.09)

### 2. ResNet50 (Transfer Learning)
Model pre-trained ResNet50 dengan fine-tuning:
- Base model: ResNet50 (ImageNet weights)
- GlobalAveragePooling2D
- Dense layer 256 neurons
- Dropout 0.5
- Early stopping dengan patience=5

**Hasil:**
- Test Accuracy: **97%**
- Performa sangat baik di semua kelas
- F1-score tertinggi: Unripe (0.99)
- F1-score terendah: Damaged (0.91)

### 3. EfficientNetB0 (Transfer Learning)
Model pre-trained EfficientNetB0 dengan arsitektur efisien:
- Base model: EfficientNetB0 (ImageNet weights)
- GlobalAveragePooling2D
- Dense layer 256 neurons
- Dropout 0.5
- Early stopping dengan patience=5

**Hasil:**
- Test Accuracy: **98%** (Model Terbaik)
- Performa excellent di semua kelas
- F1-score Unripe & Ripe: 1.00 dan 0.99
- Balanced performance across classes

## Perbandingan Model

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| CNN | 76% | 0.79 | 0.76 | 0.73 |
| ResNet50 | 97% | 0.97 | 0.97 | 0.97 |
| **EfficientNetB0** | **98%** | **0.98** | **0.98** | **0.98** |

## Requirements

```python
numpy
pandas
pillow
matplotlib
seaborn
scikit-learn
tensorflow>=2.0
opencv-python
```

## Instalasi

```bash
pip install numpy pandas pillow matplotlib seaborn scikit-learn tensorflow opencv-python
```

## Cara Penggunaan

### 1. Persiapan Dataset
```python
import zipfile

zip_path = "dataset-tomat.zip"
extract_path = "/content/tomato"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
```

### 2. Training Model (EfficientNetB0)
```python
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load base model
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# Build model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=20,
    callbacks=[early_stop]
)
```

### 3. Prediksi
```python
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model('model_efficientnet.h5')

# Load dan preprocess image
img_path = 'path/to/tomato.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Prediksi
predictions = model.predict(img_array)
class_idx = np.argmax(predictions[0])

classes = ['Damaged', 'Old', 'Ripe', 'Unripe']
print(f"Prediksi: {classes[class_idx]}")
print(f"Confidence: {predictions[0][class_idx]*100:.2f}%")
```

## Struktur Direktori

```
project/
│
├── dataset-tomat.zip
├── models/
│   ├── cnn_model.py
│   ├── resnet_model.py
│   └── efficientnet_model.py
│
├── notebooks/
│   ├── 01_CNN.ipynb
│   ├── 02_ResNet.ipynb
│   └── 03_EfficientNet.ipynb
│
├── saved_models/
│   ├── model_cnn.h5
│   ├── model_resnet.h5
│   └── model_efficientnet.h5
│
└── README.md
```

## Visualisasi Hasil

Model menghasilkan beberapa visualisasi untuk analisis:
1. **Confusion Matrix**: Menampilkan performa klasifikasi per kelas
2. **Training History**: Grafik accuracy dan loss selama training
3. **Sample Predictions**: Contoh gambar dengan prediksi model

## Catatan Penting

- Model terbaik adalah **EfficientNetB0** dengan accuracy 98%
- Kelas "Damaged" paling sulit diklasifikasikan karena jumlah data yang sedikit
- Transfer learning memberikan hasil jauh lebih baik dibanding CNN from scratch
- Early stopping membantu mencegah overfitting

## Pengembangan Selanjutnya

- [ ] Implementasi data augmentation untuk meningkatkan performa kelas Damaged
- [ ] Fine-tuning layer dalam base model
- [ ] Ensemble model untuk meningkatkan akurasi
- [ ] Deployment model ke web/mobile application
- [ ] Real-time classification menggunakan webcam

## Lisensi

MIT License

## Kontak

Untuk pertanyaan atau kolaborasi, silakan hubungi melalui repository ini.

---

**Dibuat dengan ❤️ menggunakan TensorFlow dan Keras**
