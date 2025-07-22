import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 資料載入與預處理
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def build_model(hidden_layers):
    model = Sequential([
        # 固定卷積部分
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        Flatten(),
        
        # 操縱邊因：隱藏層配置
        *[Dense(64 if i==0 else 32, activation='relu') 
          for i in range(hidden_layers)],
        
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 測試不同隱藏層數
results = {}
for layers in [0, 1, 2]:
    print(f"\n=== 訓練 {layers}層隱藏層 ===")
    model = build_model(layers)
    history = model.fit(x_train, y_train,
                       epochs=10,
                       batch_size=64,
                       validation_data=(x_test, y_test))
    results[f"{layers}隱藏層"] = history.history

# 可視化比較
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

# 驗證準確率比較
plt.subplot(1, 2, 1)
for label, history in results.items():
    plt.plot(history['val_accuracy'], label=label, linewidth=2)
plt.title('Validation Accuracy by Hidden Layers', fontsize=14, pad=20)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(range(10), range(1, 11))
plt.grid(alpha=0.3)
plt.legend(title='Hidden Layers', title_fontsize=12)

# 驗證損失比較
plt.subplot(1, 2, 2)
for label, history in results.items():
    plt.plot(history['val_loss'], label=label, linewidth=2, linestyle='--')
plt.title('Validation Loss by Hidden Layers', fontsize=14, pad=20)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.xticks(range(10), range(1, 11))
plt.grid(alpha=0.3)
plt.legend(title='Hidden Layers', title_fontsize=12)

plt.tight_layout()
plt.savefig('hidden_layers_comparison.png', dpi=300, bbox_inches='tight')
plt.show()