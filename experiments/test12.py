import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np

# 資料載入與預處理
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def build_model():
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
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 測試不同Epochs
epochs_list = [10, 20, 50]
results = {}

for epochs in epochs_list:
    print(f"\n=== 訓練 Epochs={epochs} ===")
    model = build_model()
    
    # 添加早停回調 (僅監控不實際停止)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=int(epochs*0.3), 
        verbose=1,
        restore_best_weights=True)
    
    history = model.fit(x_train, y_train,
                       epochs=epochs,
                       batch_size=64,
                       validation_data=(x_test, y_test),
                       callbacks=[early_stopping])
    results[f"{epochs} epochs"] = history.history

# 可視化比較 (動態調整坐標軸範圍)
plt.figure(figsize=(15, 5))
max_epochs = max(epochs_list)

# 準確率曲線
plt.subplot(1, 2, 1)
for label, history in results.items():
    plt.plot(history['val_accuracy'], 
             label=f"{label} (max={max(history['val_accuracy']):.2f})",
             linewidth=2)
plt.title('Validation Accuracy by Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(np.arange(0, max_epochs+1, max_epochs//5))
plt.grid(alpha=0.3)
plt.legend()

# 損失曲線
plt.subplot(1, 2, 2)
for label, history in results.items():
    plt.plot(history['val_loss'], 
             label=f"{label} (min={min(history['val_loss']):.2f})",
             linestyle='--', linewidth=2)
plt.title('Validation Loss by Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.xticks(np.arange(0, max_epochs+1, max_epochs//5))
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('epochs_comparison.png', dpi=300)
plt.show()