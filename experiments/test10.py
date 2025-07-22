import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# 資料載入與預處理
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def build_model(dropout_rate):
    model = Sequential([
        # 卷積部分
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        MaxPooling2D((2,2)),
        Dropout(dropout_rate),  # 第一層Dropout
        
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(dropout_rate),  # 第二層Dropout
        
        Conv2D(64, (3,3), activation='relu', padding='same'),
        Flatten(),
        
        # 全連接部分
        Dense(64, activation='relu'),
        Dropout(dropout_rate),  # 第三層Dropout (可選)
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 測試不同Dropout比例
results = {}
for rate in [0.2, 0.4, 0.6]:
    print(f"\n=== 訓練 Dropout率={int(rate*100)}% ===")
    model = build_model(rate)
    history = model.fit(x_train, y_train,
                       epochs=10,
                       batch_size=64,
                       validation_data=(x_test, y_test))
    results[f"Dropout {int(rate*100)}%"] = history.history

# 可視化比較
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
for label, history in results.items():
    plt.plot(history['val_accuracy'], label=label, linewidth=2)
plt.title('Validation Accuracy by Dropout Rate', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()

plt.subplot(1, 2, 2)
for label, history in results.items():
    plt.plot(history['val_loss'], label=label, linestyle='--', linewidth=2)
plt.title('Validation Loss by Dropout Rate', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()

plt.tight_layout()
plt.savefig('dropout_rate_comparison.png', dpi=300)
plt.show()