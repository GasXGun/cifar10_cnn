import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# 載入資料並正規化
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 定義模型生成函式 (固定3層CNN，調整filters)
def build_model(filters):
    model = Sequential([
        Conv2D(filters, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        MaxPooling2D((2,2)),
        Conv2D(filters*2, (3,3), activation='relu', padding='same'),  # 第二層filters加倍
        MaxPooling2D((2,2)),
        Conv2D(filters*4, (3,3), activation='relu', padding='same'),  # 第三層filters再加倍
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 測試不同filters
results = {}
for filters in [16, 32, 64]:
    print(f"\n=== 訓練 filters={filters} 的3層CNN ===")
    model = build_model(filters)
    history = model.fit(x_train, y_train,
                       epochs=10,
                       batch_size=64,
                       validation_data=(x_test, y_test))
    results[f"filters={filters}"] = history.history

# 繪製比較圖表
plt.figure(figsize=(12, 4))

# 驗證準確率
plt.subplot(1, 2, 1)
for label, history in results.items():
    plt.plot(history['val_accuracy'], label=label)
plt.title('Validation Accuracy by Filters')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 驗證損失
plt.subplot(1, 2, 2)
for label, history in results.items():
    plt.plot(history['val_loss'], label=label)
plt.title('Validation Loss by Filters')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('filters_comparison.png')
plt.show()

# 輸出最終結果
print("\n=== 最終驗證性能 ===")
for label, history in results.items():
    final_acc = history['val_accuracy'][-1]
    final_loss = history['val_loss'][-1]
    print(f"{label}: 準確率={final_acc:.4f}, 損失={final_loss:.4f}")