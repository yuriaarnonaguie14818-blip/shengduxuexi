import numpy as np
import cv2
import os
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from tqdm import tqdm
import argparse
import sys
import torch
import random
import numpy as np

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import load_cfg_from_cfg_file

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def get_arguments():
    parser = argparse.ArgumentParser(description='Evaluate segmentation results')
    # 添加与训练和推理一致的参数
    parser.add_argument("--config-file", type=str, help="Path to config file")
    parser.add_argument('--seed', type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory")
    
    # 保留原始参数作为备选
    parser.add_argument('--gt_path', type=str, help='Ground truth mask path (overrides config)')
    parser.add_argument('--seg_path', type=str, help='Segmentation result path (overrides config)')
    parser.add_argument('--save_path', type=str, help='Save path for metrics CSV')
    
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, 
                       help="modify config options using the command-line")
    
    args = parser.parse_args()
    
    # 如果有config文件，加载配置
    if args.config_file:
        cfg = load_cfg_from_cfg_file(args.config_file)
        cfg.update({k: v for k, v in vars(args).items() if v is not None})
        return cfg, args
    else:
        # 如果没有config文件，使用命令行参数
        return None, args

def main():
    # 获取参数
    cfg, args = get_arguments()
    
    # 如果有配置，使用配置构建路径
    if cfg:
        # 设置随机种子
        if cfg.seed >= 0:
            print(f"Setting fixed seed: {cfg.seed}")
            set_random_seed(cfg.seed)
        
        # 根据训练配置生成结果名称（与训练和推理一致）
        rank = getattr(cfg.SAM, 'RANK', 4)
        num_shots = getattr(cfg.DATASET, 'NUM_SHOTS', 10)
        n_ctx_text = getattr(cfg.PROMPT_LEARNER, 'N_CTX_TEXT', 2)
        
        results_name = (
            f"LORA{rank}_"
            f"SHOTS{num_shots}_"
            f"NCTX{n_ctx_text}_"
            f"CSC{cfg.PROMPT_LEARNER.CSC}_"
            f"CTP{cfg.PROMPT_LEARNER.CLASS_TOKEN_POSITION}"
        )
        
        print(f"Results name: {results_name}")
        
        # 获取类别名称（假设第一个是背景，后面是前景）
        classnames = getattr(cfg.PROMPT_LEARNER, 'CLASSNAMES', ['background', 'tumor'])
        
        # 构建路径
        if len(classnames) > 1:
            # 使用第一个前景类别进行评估
            foreground_class = classnames[1]
            seg_path = os.path.join(
                cfg.output_dir,
                cfg.DATASET.NAME,
                "seg_results",
                f"seed{cfg.seed}",
                foreground_class,
                results_name
            )
            gt_path = os.path.join(
                cfg.output_dir,
                cfg.DATASET.NAME,
                "gt_masks",
                f"seed{cfg.seed}",
                foreground_class,
                results_name
            )
            
            # 自动生成保存路径
            save_path = os.path.join(
                cfg.output_dir,
                cfg.DATASET.NAME,
                "trained_models",
                f"seed{cfg.seed}",
                f"metrics_{results_name}.csv"
            )
            
            print(f"Segmentation path: {seg_path}")
            print(f"Ground truth path: {gt_path}")
            print(f"Save metrics to: {save_path}")
        else:
            print("Error: No foreground classes found in CLASSNAMES")
            return
    else:
        # 使用命令行参数
        seg_path = args.seg_path
        gt_path = args.gt_path
        save_path = args.save_path
        
        # 检查必需参数
        if not seg_path or not gt_path:
            print("Error: When not using config file, --seg_path and --gt_path are required")
            return
        
        if not save_path:
            save_path = 'test.csv'
    
    # 检查路径是否存在
    if not os.path.exists(seg_path):
        print(f"Error: Segmentation path does not exist: {seg_path}")
        return
    
    if not os.path.exists(gt_path):
        print(f"Error: Ground truth path does not exist: {gt_path}")
        return
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Get list of filenames
    filenames = os.listdir(seg_path)
    filenames = [x for x in filenames if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.tif')]
    filenames = [x for x in filenames if os.path.exists(os.path.join(seg_path, x))]
    filenames.sort()
    
    if not filenames:
        print(f"Error: No valid image files found in {seg_path}")
        return
    
    print(f"Found {len(filenames)} images to evaluate")
    
    # Initialize metrics dictionary
    seg_metrics = OrderedDict(
        Name = list(),
        DSC = list(),
        NSD = list(),
    )
    
    # Compute metrics for each file
    with tqdm(filenames) as pbar:
        for idx, name in enumerate(pbar):
            seg_metrics['Name'].append(name)
            gt_mask = cv2.imread(os.path.join(gt_path, name), cv2.IMREAD_GRAYSCALE)
            seg_mask = cv2.imread(os.path.join(seg_path, name), cv2.IMREAD_GRAYSCALE)
            
            if gt_mask is None:
                print(f"Warning: Could not read ground truth mask {name}")
                continue
                
            if seg_mask is None:
                print(f"Warning: Could not read segmentation mask {name}")
                continue
            
            # 如果分割掩码和真实掩码尺寸不一致，调整分割掩码尺寸
            if gt_mask.shape != seg_mask.shape:
                seg_mask = cv2.resize(seg_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # 二值化处理
            gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)[1]
            seg_mask = cv2.threshold(seg_mask, 127, 255, cv2.THRESH_BINARY)[1]
            gt_data = np.uint8(gt_mask)
            seg_data = np.uint8(seg_mask)

            gt_labels = np.unique(gt_data)[1:]
            seg_labels = np.unique(seg_data)[1:]
            labels = np.union1d(gt_labels, seg_labels)

            if len(labels) == 0:
                print(f"Warning: No labels found in ground truth mask {name}")
                # 如果没有标签，默认DSC和NSD为0
                DSC = 0
                NSD = 0
            else:
                DSC_arr = []
                NSD_arr = []
                for i in labels:
                    if np.sum(gt_data==i)==0 and np.sum(seg_data==i)==0:
                        DSC_i = 1
                        NSD_i = 1
                    elif np.sum(gt_data==i)==0 and np.sum(seg_data==i)>0:
                        DSC_i = 0
                        NSD_i = 0
                    else:
                        i_gt, i_seg = gt_data == i, seg_data == i
                        DSC_i = compute_dice_coefficient(i_gt, i_seg)
                        surface_distances = compute_surface_distances(i_gt[..., None], i_seg[..., None], [1, 1, 1])
                        NSD_i = compute_surface_dice_at_tolerance(surface_distances, 3)

                    DSC_arr.append(DSC_i)
                    NSD_arr.append(NSD_i)

                DSC = np.mean(DSC_arr)
                NSD = np.mean(NSD_arr)
            
            seg_metrics['DSC'].append(round(DSC, 4))
            seg_metrics['NSD'].append(round(NSD, 4))

            # Update tqdm bar with running means
            mean_dsc = np.mean(seg_metrics['DSC'])
            mean_nsd = np.mean(seg_metrics['NSD'])
            pbar.set_postfix({
                'Mean DSC': f"{mean_dsc:.4f}",
                'Mean NSD': f"{mean_nsd:.4f}"
            })
    
    # Save metrics to CSV
    dataframe = pd.DataFrame(seg_metrics)
    dataframe.to_csv(save_path, index=False)

    # Calculate and print average and std deviation for metrics
    case_avg_DSC = dataframe['DSC'].mean()
    case_avg_NSD = dataframe['NSD'].mean()
    case_std_DSC = dataframe['DSC'].std()
    case_std_NSD = dataframe['NSD'].std()

    print("\n" + 20 * '>')
    print(f'Evaluation Results:')
    print(20 * '-')
    print(f'Number of images: {len(dataframe)}')
    print(f'Average DSC: {case_avg_DSC:.4f}')
    print(f'Standard deviation DSC: {case_std_DSC:.4f}')
    print(f'Average NSD: {case_avg_NSD:.4f}')
    print(f'Standard deviation NSD: {case_std_NSD:.4f}')
    print(20 * '<')
    
    # 如果使用了配置，也打印详细结果
    if cfg:
        print(f"\nDetailed results saved to: {save_path}")
        
        # 显示前10个结果
        print("\nFirst 10 results:")
        print(dataframe.head(10))

if __name__ == "__main__":
    main()
