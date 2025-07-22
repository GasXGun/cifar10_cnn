# 測試二
import os
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# === 強制使用 GPU 設定 ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用第一張 GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU 已啟用:", physical_devices)
else:
    print("未檢測到 GPU，將使用 CPU")

# === 載入資料與預處理 ===
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 正規化（基於測試一結論）

# === 固定模型架構 ===
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

# === 實驗：比較 Shuffle 與否 ===
results = {}
for shuffle_flag in [True, False]:
    print(f"\n=== 訓練模式: shuffle={shuffle_flag} ===")
    model = create_model()
    history = model.fit(x_train, y_train,
                       epochs=10,
                       batch_size=64,
                       shuffle=shuffle_flag,  # 關鍵參數
                       validation_data=(x_test, y_test))
    results[f"shuffle_{shuffle_flag}"] = history.history

# === 可視化比較 ===
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

# 繪製準確率曲線
plt.subplot(1, 2, 1)
plt.plot(results['shuffle_True']['val_accuracy'], label='Shuffle=True')
plt.plot(results['shuffle_False']['val_accuracy'], label='Shuffle=False')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 繪製損失曲線
plt.subplot(1, 2, 2)
plt.plot(results['shuffle_True']['val_loss'], label='Shuffle=True')
plt.plot(results['shuffle_False']['val_loss'], label='Shuffle=False')
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('shuffle_comparison.png')  # 保存圖片
plt.show()