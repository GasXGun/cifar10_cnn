作業2
Cifar-10資料集，以CNN卷積式神經網路進行訓練

文件結構(recommended by DeepSeek):
```
cifar10_cnn/
├── data/                # 存放數據集
├── models/              # 保存訓練好的模型
├── utils/               # 工具函數
│   └── preprocess.py    # 數據預處理函數
├── experiments/         # 不同實驗
│   ├── test2_normalization.py
│   ├── test3_shuffle.py
│   └── ...
├── config.py            # 共用配置
└── main.py              # 主入口文件
```

測試一：固定模型架構及參數下，資料是否正規化之比較。
![alt text](experiments/normalization_comparison.png)
最終測試準確率:
未正規化資料: 0.6385
正規化資料: 0.6949
在CNN訓練中，輸入數據正規化是必要步驟，尤其當像素值範圍較大時(如 0-255)。本測試顯示正規化可同時提升準確率、加速收斂，並降低訓練不穩定性。

測試二：固定模型架構及參數下，資料是否進行shuffle之比較。
![alt text](experiments/shuffle_comparison.png)
測試結果表明，在訓練CNN模型時啟用資料Shuffle能顯著提升模型效能，最終驗證準確率提高約6%，同時使訓練過程更穩定、收斂更快。

測試三：固定參數下，卷積層層數多寡之比較。至少三種網路架構，例如：兩層、三層、四層。
控制變因：
使用相同的優化器(Adam)、學習率、批次大小(batch_size=64)、正規化(像素值縮放到 0-1)
操縱變因：
2層 CNN：Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense
3層 CNN：額外增加一組 Conv2D → MaxPooling
4層 CNN：再增加一組 Conv2D → MaxPooling
![alt text](experiments/cnn_layers_comparison.png)
| 模型架構   | 最終驗證準確率 | 最終驗證損失 | 訓練準確率 | 關鍵觀察                     |
|------------|----------------|--------------|------------|------------------------------|
| **2層CNN** | 70.71%         | 0.8879       | 80.30%     | 訓練集過擬合明顯（+9.59%）   |
| **3層CNN** | 73.17%         | 0.9160       | 87.07%     | 最佳泛化能力，但後期波動     |
| **4層CNN** | 73.78%         | 0.8593       | 87.38%     | 最高準確率，但過擬合風險最大 |
- **收斂速度**  
  - 所有模型在Epoch 3-4後驗證準確率趨穩，說明**CIFAR-10在10個Epoch內可達初步收斂**。
  - 4層CNN初期損失下降最快，但後期過擬合。

- **過擬合跡象**  
  - 2層/4層模型的訓練準確率比驗證準確率高出 **10%以上**，3層差距最小（13.9%）。