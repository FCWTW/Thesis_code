import os
import math
import multiprocessing
import mmcv
import torch
from mmseg.apis import inference_model, init_model, show_result_pyplot

# ================= 設定區域 =================
# 您的設定檔路徑
# CONFIG_FILE = '/home/wayne/Documents/Progress/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101b-d8_4xb2-80k_cityscapes-512x1024.py'
# CHECKPOINT_FILE = 'deeplabv3plus_r101b-d8_512x1024_80k_cityscapes_20201226_190843-9c3c93a4.pth'
CONFIG_FILE = '/home/wayne/Documents/Progress/mmsegmentation/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py'
CHECKPOINT_FILE = 'mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024_20221202_141901-28ad20f1.pth'

# 資料集路徑
BASE_ROOT = '/home/wayne/Documents/BDDA'

# 平行處理設定
# 如果您有多張 GPU，可以在這裡列出，如 [0, 1, 2, 3]
# 如果只有一張 GPU，建議先設為 [0]，視顯存大小決定能否設多個 worker 共用同一張卡
GPU_IDS = [0] 

# 同時執行的 Process 數量 (建議不要超過 GPU 數量太多，除非顯存很大)
# DeepLabV3+ R101 模型很大，單張 GPU 建議先設 1 或 2 測試，設太高會 OOM
NUM_PROCESSES = 2 
# ===========================================

def worker_process(file_list, gpu_id, process_idx):
    """
    每個 Process 的工作函數
    """
    print(f'[Process {process_idx}] 啟動於 GPU {gpu_id}, 需處理 {len(file_list)} 張圖片')
    
    # 重點：在 Process 內部初始化模型，避免跨 Process 傳遞 CUDA Tensor 的問題
    device = f'cuda:{gpu_id}'
    try:
        model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=device)
    except Exception as e:
        print(f'[Process {process_idx}] 模型載入失敗: {e}')
        return

    for input_path, output_path in file_list:
        try:
            # 確保輸出資料夾存在 (雖然主程式有做，但多一層保險)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 推論
            img = mmcv.imread(input_path)
            result = inference_model(model, img)
            
            # 存檔 (show_result_pyplot 包含繪圖與存檔)
            show_result_pyplot(model, img, result, show=False, out_file=output_path, opacity=1)
            
        except Exception as e:
            print(f'[Process {process_idx}] 處理圖片錯誤 {input_path}: {e}')
            
    print(f'[Process {process_idx}] 完成！')

def main():
    # 1. 收集所有需要處理的檔案路徑
    all_tasks = []
    print("正在掃描檔案...")
    
    for set_dir in ['training', 'test', 'validation']:
        input_base_path = os.path.join(BASE_ROOT, set_dir, 'camera_frames')
        output_base_path = os.path.join(BASE_ROOT, set_dir, 'segmentation')
        
        if not os.path.exists(input_base_path):
            continue

        for digit_folder in os.listdir(input_base_path):
            input_digit_path = os.path.join(input_base_path, digit_folder)
            output_digit_path = os.path.join(output_base_path, digit_folder)
            
            if not os.path.isdir(input_digit_path):
                continue
                
            # 預先建立資料夾，減少 Process 間的 IO 競爭
            os.makedirs(output_digit_path, exist_ok=True)
            
            for filename in os.listdir(input_digit_path):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    input_image_path = os.path.join(input_digit_path, filename)
                    output_image_path = os.path.join(output_digit_path, filename)
                    all_tasks.append((input_image_path, output_image_path))

    total_files = len(all_tasks)
    print(f"總共發現 {total_files} 張圖片。準備使用 {NUM_PROCESSES} 個 Process 處理。")

    if total_files == 0:
        print("沒有找到圖片，結束程式。")
        return

    # 2. 將任務清單切分成數份 (Chunking)
    chunk_size = math.ceil(total_files / NUM_PROCESSES)
    chunks = [all_tasks[i:i + chunk_size] for i in range(0, total_files, chunk_size)]

    # 3. 啟動多進程
    processes = []
    # 設定啟動方法為 spawn，這在 PyTorch/CUDA 環境下比較穩定
    multiprocessing.set_start_method('spawn', force=True)

    for i in range(NUM_PROCESSES):
        # 輪流分配 GPU (Round Robin)
        gpu_id = GPU_IDS[i % len(GPU_IDS)]
        
        # 建立 Process
        # 每個 Process 處理 chunks[i] 這一部分的檔案
        p = multiprocessing.Process(
            target=worker_process, 
            args=(chunks[i], gpu_id, i)
        )
        p.start()
        processes.append(p)

    # 4. 等待所有 Process 結束
    for p in processes:
        p.join()

    print('全部處理完成 (Finish)!!!')

if __name__ == '__main__':
    main()