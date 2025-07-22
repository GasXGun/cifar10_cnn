import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# 資料載入與預處理
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def build_model(optimizer):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        MaxPooling2D((2,2)),
        Dropout(0.4),
        
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.4),
        
        Conv2D(64, (3,3), activation='relu', padding='same'),
        Flatten(),
        
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=optimizer,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 定義優化器 (相同初始學習率)
optimizers = {
    "SGD": tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
    "RMSprop": tf.keras.optimizers.RMSprop(learning_rate=0.001),
    "Adam": tf.keras.optimizers.Adam(learning_rate=0.001)
}

# 測試不同優化器
results = {}
for name, opt in optimizers.items():
    print(f"\n=== 訓練 優化器={name} ===")
    model = build_model(opt)
    history = model.fit(x_train, y_train,
                       epochs=10,
                       batch_size=64,
                       validation_data=(x_test, y_test))
    results[name] = history.history

# 可視化比較
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
for label, history in results.items():
    plt.plot(history['val_accuracy'], label=label, linewidth=2)
plt.title('Validation Accuracy by Optimizer', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()

plt.subplot(1, 2, 2)
for label, history in results.items():
    plt.plot(history['val_loss'], label=label, linestyle='--', linewidth=2)
plt.title('Validation Loss by Optimizer', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()

plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=300)
plt.show()