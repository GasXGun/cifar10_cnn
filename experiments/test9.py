import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# 載入資料
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def build_model(dropout_layers):
    model = Sequential()
    
    # 固定卷積部分
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)))
    model.add(MaxPooling2D((2,2)))
    if dropout_layers >= 1:
        model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    if dropout_layers >= 2:
        model.add(Dropout(0.3))
    
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(Flatten())
    if dropout_layers >= 3:
        model.add(Dropout(0.4))
    
    model.add(Dense(64, activation='relu'))
    if dropout_layers >= 4:
        model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 測試不同Dropout層數
results = {}
for layers in [0, 2, 3]:
    print(f"\n=== 訓練 {layers}層Dropout ===")
    model = build_model(layers)
    history = model.fit(x_train, y_train,
                       epochs=10,
                       batch_size=64,
                       validation_data=(x_test, y_test))
    results[f"{layers}層Dropout"] = history.history

# 可視化比較
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
for label, history in results.items():
    plt.plot(history['val_accuracy'], label=label, linewidth=2)
plt.title('Validation Accuracy by Dropout Layers')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
for label, history in results.items():
    plt.plot(history['val_loss'], label=label, linestyle='--')
plt.title('Validation Loss by Dropout Layers')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('dropout_comparison.png')
plt.show()