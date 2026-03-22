# Fashion MNIST ResNet-18 Model

## 📋 Mô Tả Dự Án

Dự án này xây dựng một mô hình **ResNet-18** để phân loại hình ảnh từ tập dữ liệu **Fashion MNIST**. Mô hình được huấn luyện với các kỹ thuật tiên tiến bao gồm:

- **Data Augmentation**: Lật ngang, xoay, zoom hình ảnh
- **Batch Normalization**: Chuẩn hóa từng batch trong quá trình huấn luyện
- **L2 Regularization**: Giảm overfitting
- **Learning Rate Scheduling**: Giảm tốc độ học theo quá trình huấn luyện
- **Early Stopping**: Dừng huấn luyện khi validation loss không cải thiện

## 🎯 Tập Dữ Liệu

**Fashion MNIST** là tập dữ liệu chứa 70,000 hình ảnh 28×28 pixel của các item thời trang:

| Chỉ số | Loại Item   |
| ------ | ----------- |
| 0      | T-shirt/top |
| 1      | Trouser     |
| 2      | Pullover    |
| 3      | Dress       |
| 4      | Coat        |
| 5      | Sandal      |
| 6      | Shirt       |
| 7      | Sneaker     |
| 8      | Bag         |
| 9      | Ankle boot  |

**Chia tập dữ liệu:**

- Training: 50,000 mẫu
- Validation: 10,000 mẫu (1/6 của training)
- Test: 10,000 mẫu

## 🏗️ Kiến Trúc Mô Hình

### ResNet-18 Chi Tiết:

```
Input (28×28×1)
  ↓
Data Augmentation Layer
  ↓
Conv2D(64, 3×3) + BatchNorm + ReLU
  ↓
Stage 1: 2× Residual Block (filters=64, 28×28)
  ↓
Stage 2: 2× Residual Block (filters=128, stride=2, 14×14)
  ↓
Stage 3: 2× Residual Block (filters=256, stride=2, 7×7)
  ↓
Stage 4: 2× Residual Block (filters=512, stride=2, 4×4)
  ↓
Global Average Pooling
  ↓
Dense(10, softmax)
  ↓
Output (10 classes)
```

### Residual Block:

Mỗi block chứa:

- Conv2D(filters, 3×3) + BatchNorm + ReLU
- Conv2D(filters, 3×3) + BatchNorm
- Shortcut connection (với Conv2D 1×1 nếu cần)
- ReLU activation

## ⚙️ Các Bước Tiền Xử Lý Dữ Liệu

1. **Đọc dữ liệu** từ định dạng binary IDX của MNIST
2. **Chia tập huấn luyện** thành training và validation
3. **Tính toán thống kê**: Mean = {mean:.4f}, Std = {std:.4f} trên tập training
4. **Chuẩn hóa dữ liệu** sử dụng công thức: `(x - mean) / (std + 1e-7)`

## 🔧 Cấu Hình Huấn Luyện

```python
Optimizer: Adam (learning_rate=0.001)
Loss Function: Sparse Categorical Crossentropy
Batch Size: 128
Epochs: 50 (với Early Stopping)
Validation Split: 1/6
```

### Callbacks:

| Callback              | Chức Năng                                                                    |
| --------------------- | ---------------------------------------------------------------------------- |
| **ModelCheckpoint**   | Lưu mô hình tốt nhất dựa trên validation accuracy                            |
| **ReduceLROnPlateau** | Giảm learning rate nếu validation loss ko cải thiện (factor=0.2, patience=3) |
| **EarlyStopping**     | Dừng huấn luyện nếu validation loss ko cải thiện trong 7 epochs              |

## 📊 Kết Quả Huấn Luyện

- **Độ chính xác trên tập Validation**: Được theo dõi qua từng epoch
- **Loss trên tập Test**: Được tính toán sau khi huấn luyện xong
- **Độ chính xác trên tập Test**: Phần trăm mẫu được dự đoán chính xác

Biểu đồ accuracy được hiển thị so sánh:

- Training accuracy qua các epochs
- Validation accuracy qua các epochs

## 📁 Cấu Trúc Dự Án

```
fashion_mnist_ResNet/
│
├── README.md                           # File này
├── FashionMNIST.ipynb                  # Notebook chứa code huấn luyện
│
├── data/                               # Thư mục chứa dữ liệu
│   ├── train-images-idx3-ubyte.gz      # Hình ảnh huấn luyện (nén)
│   ├── train-labels-idx1-ubyte.gz      # Nhãn huấn luyện (nén)
│   ├── t10k-images-idx3-ubyte.gz       # Hình ảnh test (nén)
│   └── t10k-labels-idx1-ubyte.gz       # Nhãn test (nén)
│
└── models/                             # Thư mục lưu mô hình đã huấn luyện
  └── exports/
    ├── best_fashion_resnet.keras   # Mô hình ResNet-18 tốt nhất
    └── norm_params.npy             # Tham số chuẩn hóa (mean, std)
```

## 📦 Yêu Cầu

```
numpy
tensorflow >= 2.0
scikit-learn
matplotlib
```

## 🚀 Cách Sử Dụng

### 1. Chuẩn Bị Dữ Liệu

Download fashion MNIST data và đặt vào thư mục `data/`

### 2. Chạy Notebook

```bash
jupyter notebook FashionMNIST.ipynb
```

### 3. Dự Đoán với Mô Hình Đã Huấn Luyện

```python
import numpy as np
import tensorflow as tf

# Load mô hình và tham số chuẩn hóa
model = tf.keras.models.load_model('best_fashion_resnet.keras')
norm_params = np.load('norm_params.npy')
mean, std = norm_params[0], norm_params[1]

# Chuẩn hóa hình ảnh
img = np.random.rand(28, 28, 1)  # Thay bằng hình ảnh thực tế
img_normalized = (img - mean) / (std + 1e-7)

# Dự đoán
prediction = model.predict(np.expand_dims(img_normalized, axis=0))
class_idx = np.argmax(prediction)

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(f"Dự đoán: {classes[class_idx]}")
```

## 💾 Các File Đầu Ra

- **best_fashion_resnet.keras**: Mô hình ResNet-18 được huấn luyện tốt nhất
- **norm_params.npy**: Tham số chuẩn hóa (mean, std) sử dụng trong inference

## 📈 Visualization

Notebook tạo ra biểu đồ so sánh:

- Accuracy qua các epochs (training vs validation)

## 🛠️ Các Kỹ Thuật Sử Dụng

- **Residual Networks**: Cho phép huấn luyện mô hình sâu hơn
- **Batch Normalization**: Tăng ổn định huấn luyện
- **Data Augmentation**: Tăng khả năng tổng quát hóa
- **L2 Regularization**: Giảm overfitting
- **Adaptive Learning Rate**: Điều chỉnh tốc độ học tự động

## 📝 Ghi Chú

- Dữ liệu được chuẩn hóa (standardization) thay vì normalize
- Mô hình sử dụng L2 regularization với hệ số 1e-4
- Early stopping được áp dụng với patience=7 epochs

## 📄 License

Tự do sử dụng cho mục đích học tập và nghiên cứu
