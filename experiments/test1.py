import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# 1. 載入資料
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. 準備兩種資料版本
# 未正規化 (保持 0-255 uint8)
x_train_raw = x_train
x_test_raw = x_test

# 正規化 (轉換為 0-1 float32)
x_train_norm = x_train.astype('float32') / 255.0
x_test_norm = x_test.astype('float32') / 255.0

# 3. 定義固定模型架構
def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 4. 訓練未正規化資料模型
print("訓練未正規化資料...")
model_raw = create_model()
history_raw = model_raw.fit(x_train_raw, y_train, 
                          epochs=10,
                          validation_data=(x_test_raw, y_test))

# 5. 訓練正規化資料模型
print("\n訓練正規化資料...")
model_norm = create_model()
history_norm = model_norm.fit(x_train_norm, y_train, 
                            epochs=10,
                            validation_data=(x_test_norm, y_test))

# 6. 可視化比較
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_raw.history['val_accuracy'], label='Raw Data')
plt.plot(history_norm.history['val_accuracy'], label='Normalized Data')
plt.title('Validation Accuracy Comparison')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_raw.history['val_loss'], label='Raw Data')
plt.plot(history_norm.history['val_loss'], label='Normalized Data')
plt.title('Validation Loss Comparison')
plt.legend()
plt.savefig('normalization_comparison.png')  # 保存圖片
plt.show()

# 7. 輸出最終測試準確率
raw_test_acc = model_raw.evaluate(x_test_raw, y_test, verbose=0)[1]
norm_test_acc = model_norm.evaluate(x_test_norm, y_test, verbose=0)[1]

print(f"\n最終測試準確率:")
print(f"未正規化資料: {raw_test_acc:.4f}")
print(f"正規化資料: {norm_test_acc:.4f}")