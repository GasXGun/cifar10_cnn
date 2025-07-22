import os
import cv2
import numpy as np
from tensorflow.keras.datasets import cifar10

# 1. 載入資料集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. 合併訓練集和測試集（共60,000張）
x_all = np.concatenate((x_train, x_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

# 3. 創建保存目錄
output_dir = "cifar10_images"
os.makedirs(output_dir, exist_ok=True)

# 4. 定義類別子目錄
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
for class_name in class_names:
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

# 5. 保存圖檔（格式：類別.序號.jpg）
for idx in range(len(x_all)):
    class_id = y_all[idx][0]
    image = x_all[idx]
    
    # 轉換顏色通道 (RGB -> BGR for OpenCV)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 生成檔名 (e.g. "0/0.123.jpg")
    filename = f"{class_id}/{class_id}.{idx % 5000}.jpg"  # 每類5000張
    
    # 保存圖檔
    cv2.imwrite(os.path.join(output_dir, filename), image_bgr)
    
    # 進度顯示
    if idx % 1000 == 0:
        print(f"已處理 {idx+1}/60000 張圖檔")

print("所有圖檔保存完成！")