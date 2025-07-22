import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# 資料載入與預處理
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def build_model(kernel_size):
    model = Sequential([
        Conv2D(32, kernel_size, activation='relu', padding='same', input_shape=(32,32,3)),
        MaxPooling2D((2,2)),
        Conv2D(64, kernel_size, activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Conv2D(64, kernel_size, activation='relu', padding='same'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 測試不同kernel_size
results = {}
for size in [(3,3), (5,5), (7,7)]:
    print(f"\n=== 訓練 kernel_size={size} ===")
    model = build_model(size)
    history = model.fit(x_train, y_train,
                       epochs=10,
                       batch_size=64,
                       validation_data=(x_test, y_test))
    results[f"kernel{size[0]}x{size[1]}"] = history.history

# 可視化比較
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
for label, history in results.items():
    plt.plot(history['val_accuracy'], label=label)
plt.title('Validation Accuracy')
plt.legend()

plt.subplot(1,2,2)
for label, history in results.items():
    plt.plot(history['val_loss'], label=label)
plt.title('Validation Loss') 
plt.legend()
plt.savefig('kernel_comparison.png')