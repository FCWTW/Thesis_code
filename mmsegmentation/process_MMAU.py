import os
import math
import multiprocessing
import mmcv
import torch
from mmseg.apis import inference_model, init_model, show_result_pyplot
from tqdm import tqdm  # 加入 tqdm

# CONFIG_FILE = '/home/wayne/Documents/Progress/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101b-d8_4xb2-80k_cityscapes-512x1024.py'
# CHECKPOINT_FILE = 'deeplabv3plus_r101b-d8_512x1024_80k_cityscapes_20201226_190843-9c3c93a4.pth'
CONFIG_FILE = '/home/wayne/Documents/Progress/mmsegmentation/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py'
CHECKPOINT_FILE = 'mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024_20221202_141901-28ad20f1.pth'
BASE_ROOT = '/home/wayne/Documents/MMAU'

GPU_IDS = [0]
NUM_PROCESSES = 2

def worker_process(file_list, gpu_id, process_idx):
    # 這裡的 print 可以保留，但 tqdm 出現後它會被推到上方
    device = f'cuda:{gpu_id}'
    try:
        model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=device)
    except Exception as e:
        print(f'\n[Process {process_idx}] 模型載入失敗: {e}')
        return

    # 使用 tqdm 包裝 file_list
    # position=process_idx 確保每個 process 的進度條固定在不同的行數
    for input_path, output_path in tqdm(file_list, desc=f'Worker {process_idx} (GPU {gpu_id})', position=process_idx, leave=True):
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 推論
            img = mmcv.imread(input_path)
            result = inference_model(model, img)
            
            # 存檔
            show_result_pyplot(model, img, result, show=False, out_file=output_path, opacity=1)
            
        except Exception as e:
            # 使用 tqdm.write 避免錯誤訊息打亂進度條的排版
            tqdm.write(f'[Process {process_idx}] 處理圖片錯誤 {input_path}: {e}')

def main():
    # 1. 收集所有需要處理的檔案路徑
    all_tasks = []
    print("正在掃描檔案...")
    
    for set_dir in ['DADA-DATA', 'CAP-DATA']:
        set_base_path = os.path.join(BASE_ROOT, set_dir)
        if not os.path.exists(set_base_path):
            print(f'Can not find {set_base_path}')
            continue
        
        for category_folder in os.listdir(set_base_path):
            category_path = os.path.join(set_base_path, category_folder)
            if not os.path.exists(category_path):
                print(f'Can not find {category_path}')
                continue
            
            for video_folder in os.listdir(category_path):
                video_path = os.path.join(category_path, video_folder)
                if not os.path.exists(video_path):
                    print(f'Can not find {video_path}')
                    continue

                input_dir = os.path.join(video_path, 'images')
                output_dir = os.path.join(video_path, 'segmentation')
                if not os.path.exists(input_dir):
                    print(f'--- Can not find {input_dir}/images ---')
                    continue
                
                os.makedirs(output_dir, exist_ok=True)
                for filename in os.listdir(input_dir):
                    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                        input_image_path = os.path.join(input_dir, filename)
                        output_image_path = os.path.join(output_dir, filename)
                        all_tasks.append((input_image_path, output_image_path))

    total_files = len(all_tasks)
    print(f"總共發現 {total_files} 張圖片。準備使用 {NUM_PROCESSES} 個 Process 處理。\n")

    if total_files == 0:
        print("沒有找到圖片，結束程式。")
        return

    # 2. 將任務清單切分成數份 (Chunking)
    chunk_size = math.ceil(total_files / NUM_PROCESSES)
    chunks = [all_tasks[i:i + chunk_size] for i in range(0, total_files, chunk_size) if all_tasks[i:i + chunk_size]]

    # 3. 啟動多進程
    processes = []
    multiprocessing.set_start_method('spawn', force=True)

    # 修改這裡：使用 len(chunks) 而不是 NUM_PROCESSES
    # 避免當任務總數極少時，chunks 數量少於 NUM_PROCESSES 導致 IndexError
    for i in range(len(chunks)):
        gpu_id = GPU_IDS[i % len(GPU_IDS)]
        
        p = multiprocessing.Process(
            target=worker_process, 
            args=(chunks[i], gpu_id, i)
        )
        p.start()
        processes.append(p)

    # 4. 等待所有 Process 結束
    for p in processes:
        p.join()

    # 確保最後的完成訊息印在所有進度條的下方
    print('\n' * len(chunks) + '全部處理完成 (Finish)!!!')

if __name__ == '__main__':
    main()