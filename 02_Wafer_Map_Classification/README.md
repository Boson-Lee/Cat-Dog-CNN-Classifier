# 02 - Wafer Map Defect Classification: 晶圓瑕疵圖譜分類

使用卷積神經網路 (CNN) 對台積電真實晶圓瑕疵圖譜 (WM-811K) 進行分類，並探討其物理成因。

## 專案結構

```
02_Wafer_Map_Classification/
├── data/               # WM-811K 原始資料 (不上傳 GitHub)
├── notebooks/          # Jupyter Notebook 實驗紀錄
├── src/                # 可重複使用的 Python 模組
├── figures/            # 視覺化圖表輸出
├── download_data.py    # Kaggle 資料下載腳本
└── README.md
```

## 🚀 線上預覽 (Google Colab)

目前我們正在進行探索性資料分析 (EDA)：
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Boson-Lee/Cat-Dog-CNN-Classifier/blob/main/02_Wafer_Map_Classification/notebooks/01_Wafer_Map_EDA.ipynb)

*(未來訓練模型的 Notebook 完成後，也會在這裡新增專屬的 Colab 按鈕！)*

## 資料集

| 項目 | 說明 |
|---|---|
| 來源 | [Kaggle: WM-811K Wafer Map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) |
| 樣本數 | 811,457 片晶圓 (約 17 萬片有人工標記) |
| 分類標籤 | 9 類 (Center, Donut, Edge-Ring, Edge-Local, Local, Random, Scratch, Near-full, None) |
| 挑戰 | 影像尺寸不一、嚴重類別不平衡 |

## 快速開始

1. **安裝 Kaggle API 套件**
   ```bash
   pip install kaggle pandas numpy matplotlib opencv-python
   ```
2. **設定 Kaggle API**
   請確保你已從 Kaggle 帳號下載 `kaggle.json` 並放入正確的路徑 (`C:\Users\User\.kaggle\kaggle.json`)。
3. **下載資料**
   執行下載腳本：
   ```bash
   python download_data.py
   ```
