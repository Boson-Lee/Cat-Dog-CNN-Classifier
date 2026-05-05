"""
下載 WM-811K Wafer Map 資料集
來源: Kaggle (qingyi/wm811k-wafer-map)

注意：執行前請確認已設定好 Kaggle API (kaggle.json)。
"""
import os
import sys

# 修復 Windows 編碼問題
sys.stdout.reconfigure(encoding='utf-8')

def main():
    try:
        import kaggle
    except OSError as e:
        print("[錯誤] 找不到 Kaggle API Token (kaggle.json)。")
        print("請到 https://www.kaggle.com/ 登入後，點擊右上角頭像 -> Settings -> Create New Token。")
        print("然後將下載的 kaggle.json 放到以下路徑：C:\\Users\\User\\.kaggle\\kaggle.json")
        return
    except ImportError:
        print("[錯誤] 尚未安裝 kaggle 套件。請執行: pip install kaggle")
        return

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(save_dir, exist_ok=True)

    dataset_identifier = "qingyi/wm811k-wafer-map"

    print(f"[INFO] 正在從 Kaggle 下載資料集: {dataset_identifier} ...")
    print("[INFO] 檔案較大 (約 100+ MB)，請耐心等候。")
    
    # 呼叫 Kaggle API 下載並解壓縮
    kaggle.api.dataset_download_files(dataset_identifier, path=save_dir, unzip=True)

    print(f"\n[OK] 下載與解壓縮完成！檔案已存入: {save_dir}")
    
    # 列出下載的檔案
    for fname in os.listdir(save_dir):
        fpath = os.path.join(save_dir, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"   - {fname} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
