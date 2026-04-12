import argparse
import yaml
import os
from os.path import join
import torch
from torchvision import transforms
from tqdm import tqdm
from train import get_model_class
from utils import img_save
from dataloader import MMAUDataset

blur_func = transforms.GaussianBlur(11, 2)

class Test():
    def __init__(self, dataset_roots):
        self.configs = None
        self.model_params = None
        self.device = None
        self.model = None
        self.dataset_root = dataset_roots

    def load_saved(self, config_dir):
        print('-> Loading configs... ', end='')
        config_path = f'{config_dir}/config.yaml'
        with open(config_path, 'r') as fid:
            self.configs = yaml.safe_load(fid)
        
        self.train_params = self.configs['train_params']
        self.model_params = self.configs['model_params']

        print('-> Loading model... ', end='', flush=True)
        self.model = get_model_class(self.configs['model_class'])(**self.model_params)
        
        best_model_weights = os.path.join(config_dir, self.configs['best_weights'])
        self.model.load_state_dict(torch.load(best_model_weights))
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        torch.backends.cudnn.benchmark = False
        print('Successfully loaded model')

    def test_saved(self, config_dir):
        self.load_saved(config_dir)
        # 實例化您的新資料集
        test_dataset = MMAUDataset(self.model_params['clip_size'], dataset_path=self.dataset_root)
        test_dataset.setup()
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=self.train_params['batch_size'],
                                                  shuffle=False,
                                                  num_workers=self.train_params['no_workers'])
        self.test(test_loader)

    def test(self, test_loader):
        self.model.eval()
        
        for idx, sample in enumerate(tqdm(test_loader, desc='Inferencing')):
            img_clips = sample[0]
            seg_img = sample[1]
            original_h, original_w = sample[2]
            vid_ids, frame_ids = sample[3], sample[4]

            # 取出該 batch 第一張圖片的高與寬，並轉為 int，供 Resize 使用
            h = int(original_h[0].item() if torch.is_tensor(original_h) else original_h)
            w = int(original_w[0].item() if torch.is_tensor(original_w) else original_w)

            # 轉移到 GPU 並調整維度以符合模型輸入
            img_clips = img_clips.to(self.device).permute((0,2,1,3,4))
            seg_img = seg_img.to(self.device).permute((0,2,1,3,4))

            with torch.no_grad():
                pred_sal = self.model(img_clips, seg_img)

            # 將預測結果放大回原圖尺寸 (使用剛剛取出的 h, w 整數)
            pred_sal = transforms.Resize((h, w))(pred_sal)
            pred_sal = blur_func(pred_sal)
            
            # 存圖 (轉回 CPU，保留 batch 維度)
            pred_sal = pred_sal.cpu() 
            self.save_batch(vid_ids, frame_ids, pred_sal)

    def save_batch(self, vid_ids, frame_ids, pred_sal):
        current_batch_size = len(vid_ids)
        
        for i in range(current_batch_size):
            vid_id_str = str(vid_ids[i]) 
            frame_id = int(frame_ids[i].item() if torch.is_tensor(frame_ids[i]) else frame_ids[i])
            
            # 將 "DADA-DATA_1_001" 拆解回 ['DADA-DATA', '1', '001']
            parts = vid_id_str.split('_')
            if len(parts) == 3:
                subset, cat, vid = parts
            else:
                print(f"Warning: Unexpected vid_id format {vid_id_str}")
                subset, cat, vid = "UNKNOWN", "UNKNOWN", vid_id_str
            
            # 建立目標儲存路徑 (ex: /home/.../MMAU/DADA-DATA/1/001/gazemap)
            save_dir = join(self.dataset_root, subset, cat, vid, 'gazemap')
            os.makedirs(save_dir, exist_ok=True)
            
            # 決定檔名的數字格式
            # frame_id 來自 index，通常檔名是從 1 開始，所以 +1
            file_num = frame_id + 1
            
            if subset == 'CAP-DATA':
                # CAP-DATA 是 6 位數 (000016.png)
                save_file = join(save_dir, f'{file_num:06d}.png')
            elif subset == 'DADA-DATA':
                # DADA-DATA 是 4 位數 (0016.png)
                save_file = join(save_dir, f'{file_num:04d}.png')
            else:
                save_file = join(save_dir, f'{file_num:05d}.png')
            
            if not os.path.exists(save_file):
                single_pred = pred_sal[i].squeeze()
                img_save(single_pred, save_file, normalize=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', required=True, type=str, help='Path to config folder')
    parser.add_argument('--dataset_dir', required=True, type=str, help='Path to MMAU dataset')
    args = parser.parse_args()

    test = Test(args.dataset_dir)
    test.test_saved(args.config_dir)