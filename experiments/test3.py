import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# 強制使用 GPU（可選）
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ========== 載入資料集 ==========
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0  # 正規化
x_test = x_test.astype('float32') / 255.0

# ========== 定義模型架構 ==========
def build_model(num_conv_layers):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))  # 第一次池化：32x32 → 16x16
    
    # 動態增加卷積層
    for i in range(num_conv_layers - 1):
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        if i < 1:  # 只有前兩層加池化，第三層後不加
            model.add(MaxPooling2D((2, 2)))  # 16x16 → 8x8 → 4x4
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# ========== 訓練與比較 ==========
results = {}
for layers in [2, 3, 4]:
    print(f"\n=== 訓練 {layers} 層卷積網路 ===")
    model = build_model(layers)
    history = model.fit(x_train, y_train,  # 確保使用已定義的 x_train, y_train
                       epochs=10,
                       batch_size=64,
                       validation_data=(x_test, y_test))
    results[f"{layers}層CNN"] = history.history

# ========== 繪製結果 ==========
plt.figure(figsize=(12, 4))

# 驗證準確率
plt.subplot(1, 2, 1)
for label, history in results.items():
    plt.plot(history['val_accuracy'], label=label)
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 驗證損失
plt.subplot(1, 2, 2)
for label, history in results.items():
    plt.plot(history['val_loss'], label=label)
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('cnn_layers_comparison.png')
plt.show()

# ========== 輸出最終結果 ==========
for label, history in results.items():
    final_acc = history['val_accuracy'][-1]
    final_loss = history['val_loss'][-1]
    print(f"{label} - 最終驗證準確率: {final_acc:.4f}, 最終驗證損失: {final_loss:.4f}")