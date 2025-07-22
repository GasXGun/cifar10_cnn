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