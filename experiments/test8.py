import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# 資料載入與預處理
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def build_model(hidden_units):
    model = Sequential([
        # 固定卷積部分
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        Flatten(),
        
        # 操縱變因：隱藏層神經元數量
        Dense(hidden_units, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 測試不同神經元數量
results = {}
for units in [512, 128, 64]:
    print(f"\n=== 訓練 hidden_units={units} ===")
    model = build_model(units)
    history = model.fit(x_train, y_train,
                       epochs=10,
                       batch_size=64,
                       validation_data=(x_test, y_test))
    results[f"{units} neurons"] = history.history

# 可視化比較
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
for label, history in results.items():
    plt.plot(history['val_accuracy'], label=label, linewidth=2)
plt.title('Validation Accuracy by Hidden Layer Size', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
for label, history in results.items():
    plt.plot(history['val_loss'], label=label, linewidth=2, linestyle='--')
plt.title('Validation Loss by Hidden Layer Size', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('hidden_units_comparison.png', dpi=300)
plt.show()