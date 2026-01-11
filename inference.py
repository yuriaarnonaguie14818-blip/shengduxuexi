import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize
from torchvision.utils import save_image
import utils.utils as utils
from datasets.dataloader import DatasetSegmentation, collate_fn
from utils.processor import Samprocessor
from segment_anything import build_sam_vit_b, build_textsam_vit_b, build_textsam_vit_h, build_textsam_vit_l
from utils.lora import LoRA_Sam
import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import os
import random
from utils.utils import load_cfg_from_cfg_file
import logging
from utils import utils
import pickle
import warnings

# 禁用pickle相关警告
warnings.filterwarnings("ignore", category=UserWarning, module='torch.serialization')

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str, help="Path to config file")
    parser.add_argument('--seed', type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")
    args = parser.parse_args()
    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.update({k: v for k, v in vars(args).items() if v is not None})
    return cfg

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

def load_checkpoint_safe(checkpoint_path, device):
    """安全地加载检查点，处理PyTorch 2.6+的weights_only问题"""
    try:
        # 首先尝试使用weights_only=True（PyTorch 2.6+默认）
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        print(f"成功以weights_only=True模式加载检查点")
        return checkpoint
    except (pickle.UnpicklingError, RuntimeError) as e:
        print(f"以weights_only=True加载失败: {str(e)[:100]}...")
        print("尝试使用weights_only=False模式加载（确保检查点来源可信）...")
        
        # 如果失败，尝试使用weights_only=False
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            print(f"成功以weights_only=False模式加载检查点")
            return checkpoint
        except Exception as e2:
            print(f"以weights_only=False加载也失败: {str(e2)[:100]}...")
            
            # 最后尝试传统的加载方式
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                print(f"成功以传统模式加载检查点")
                return checkpoint
            except Exception as e3:
                # 如果都失败，尝试手动解析模型状态字典
                print(f"所有加载方式都失败，尝试手动提取模型参数...")
                return load_checkpoint_manual(checkpoint_path, device)

def load_checkpoint_manual(checkpoint_path, device):
    """手动加载检查点，跳过有问题的部分"""
    try:
        # 使用pickle直接加载
        with open(checkpoint_path, 'rb') as f:
            # 跳过pickle的安全检查
            import io
            unpickler = pickle.Unpickler(io.BytesIO(f.read()))
            unpickler.encoding = 'latin1'  # 使用latin1编码，更兼容
            data = unpickler.load()
        
        # 提取模型状态字典
        if isinstance(data, dict):
            if "model" in data:
                return {"model": data["model"]}
            else:
                # 可能是直接保存的state_dict
                return {"model": data}
        else:
            # 可能是直接保存的state_dict
            return {"model": data}
    except Exception as e:
        print(f"手动加载也失败: {e}")
        raise

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    cfg = get_arguments()

    # 创建输出目录
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "seg_results", f"seed{cfg.seed}"), exist_ok=True)
    
    # 设置日志
    log_dir = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}")
    os.makedirs(log_dir, exist_ok=True)
    logger = logger_config(os.path.join(log_dir, "inference_log.txt"))

    logger.info("************")
    logger.info("** Config **")
    logger.info("************")
    logger.info(cfg)

    if cfg.seed >= 0:
        logger.info("Setting fixed seed: {}".format(cfg.seed))
        set_random_seed(cfg.seed)

    # 根据训练配置生成结果名称
    rank = getattr(cfg.SAM, 'RANK', 16)  # 默认使用16，与原始配置一致
    num_shots = getattr(cfg.DATASET, 'NUM_SHOTS', -1)  # 默认使用-1，与原始配置一致
    n_ctx_text = getattr(cfg.PROMPT_LEARNER, 'N_CTX_TEXT', 4)  # 默认使用4，与原始配置一致
    
    results_name = (
        f"LORA{rank}_"
        f"SHOTS{num_shots}_"
        f"NCTX{n_ctx_text}_"
        f"CSC{cfg.PROMPT_LEARNER.CSC}_"
        f"CTP{cfg.PROMPT_LEARNER.CLASS_TOKEN_POSITION}"
    )
    
    logger.info(f"Results name: {results_name}")

    checkpoint_type = "latest" if cfg.TEST.USE_LATEST else "best"

    with torch.no_grad():
        checkpoint_path = os.path.join(
            cfg.output_dir,
            cfg.DATASET.NAME,
            "trained_models",
            f"seed{cfg.seed}",
            f"{results_name}_{checkpoint_type}.pth"
        )

        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            # 尝试其他可能的检查点名称
            alternative_paths = [
                checkpoint_path,
                os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}", f"{results_name}_final.pth"),
                os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}", "final.pth"),
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    checkpoint_path = alt_path
                    logger.info(f"使用替代检查点: {checkpoint_path}")
                    break
        
        if not os.path.exists(checkpoint_path):
            logger.error(f"检查点不存在: {checkpoint_path}")
            return
        
        classnames = cfg.PROMPT_LEARNER.CLASSNAMES

        # 加载模型
        logger.info(f"构建SAM模型: {cfg.SAM.MODEL}")
        if cfg.SAM.MODEL == "vit_b":
            sam = build_textsam_vit_b(cfg=cfg, checkpoint=cfg.SAM.CHECKPOINT, classnames=classnames)
        elif cfg.SAM.MODEL == "vit_l":
            sam = build_textsam_vit_l(cfg=cfg, checkpoint=cfg.SAM.CHECKPOINT, classnames=classnames)
        else:
            sam = build_textsam_vit_h(cfg=cfg, checkpoint=cfg.SAM.CHECKPOINT, classnames=classnames)

        # 应用LoRA
        sam_lora = LoRA_Sam(sam, rank)
        model = sam_lora.sam
        
        # 安全地加载检查点
        try:
            checkpoint = load_checkpoint_safe(checkpoint_path, device)
            model.load_state_dict(checkpoint["model"], strict=False)
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            # 尝试仅加载可加载的参数
            try:
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint["model"].items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=False)
                logger.info("部分模型参数加载成功")
            except Exception as e2:
                logger.error(f"部分加载也失败: {e2}")
                return

        processor = Samprocessor(model)
        dice_scores = {}
        total_dice_values = []

        # 为每个类别创建输出目录
        for text_label in classnames[1:]:
            dice_scores[text_label] = []
            seg_path = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "seg_results", f"seed{cfg.seed}", text_label, results_name)
            gt_path = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "gt_masks", f"seed{cfg.seed}", text_label, results_name)
            
            os.makedirs(seg_path, exist_ok=True)
            os.makedirs(gt_path, exist_ok=True)
            logger.info(f"创建目录: {seg_path}")
            logger.info(f"创建目录: {gt_path}")

        # 加载测试数据集
        dataset = DatasetSegmentation(cfg, processor, mode="test")
        test_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=0)

        model.eval()
        model.to(device)

        logger.info(f"开始推理，测试样本数: {len(dataset)}")
        progress_bar = tqdm(test_dataloader, desc="Evaluating", dynamic_ncols=True)

        for i, batch in enumerate(progress_bar):
            # 确保batch是列表格式
            if isinstance(batch, dict):
                batch = [batch]
            
            # 推理
            outputs = model(batched_input=batch, multimask_output=False)
            
            # 获取真实掩码和预测掩码
            stk_gt = batch[0]["ground_truth_mask"]
            stk_out = torch.cat([out["masks"].squeeze(0) for out in outputs], dim=0)
            text_labels = batch[0]["text_labels"].squeeze(0)

            # 获取文件名
            mask_name = batch[0].get("mask_name", f"mask_{i}.png")
            if isinstance(mask_name, list):
                mask_name = mask_name[0]
            
            # 处理每个预测
            for j, label in enumerate(text_labels):
                if j >= stk_out.shape[0]:
                    break
                    
                label_j = int(label.detach().cpu())
                if label_j >= len(classnames):
                    continue
                
                # 获取预测掩码
                mask_pred_np = stk_out[j].detach().cpu().numpy()
                if mask_pred_np.ndim == 3:
                    mask_pred_np = mask_pred_np[0]  # 取第一个通道
                
                # 二值化处理
                mask_pred = (mask_pred_np > 0.5).astype(np.uint8) * 255
                
                # 获取真实掩码
                gt_mask_np = stk_gt[j].detach().cpu().numpy()
                if gt_mask_np.ndim == 3:
                    gt_mask_np = gt_mask_np[0]  # 取第一个通道
                gt_mask = (gt_mask_np > 0.5).astype(np.uint8) * 255
                
                # 保存结果
                class_name = classnames[label_j]
                seg_path = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "seg_results", f"seed{cfg.seed}", class_name, results_name, mask_name)
                gt_path = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "gt_masks", f"seed{cfg.seed}", class_name, results_name, mask_name)
                
                cv2.imwrite(seg_path, mask_pred)
                cv2.imwrite(gt_path, gt_mask)
                
                # 计算Dice系数（可选）
                try:
                    from monai.metrics import DiceMetric
                    dice_metric = DiceMetric(include_background=True, reduction="mean")
                    dice_metric(y_pred=torch.tensor(mask_pred_np).unsqueeze(0).unsqueeze(0),
                               y=torch.tensor(gt_mask_np).unsqueeze(0).unsqueeze(0))
                    dice_score = dice_metric.aggregate().item()
                    dice_scores[class_name].append(dice_score)
                    total_dice_values.append(dice_score)
                except:
                    pass

            progress_bar.set_postfix({"processed": f"{i+1}/{len(test_dataloader)}"})

        # 打印评估结果
        logger.info("\n" + "="*50)
        logger.info("评估结果:")
        logger.info("="*50)
        
        for class_name in dice_scores:
            if dice_scores[class_name]:
                avg_dice = np.mean(dice_scores[class_name])
                logger.info(f"{class_name}: Dice = {avg_dice:.4f} (n={len(dice_scores[class_name])})")
        
        if total_dice_values:
            logger.info(f"\n总体Dice系数: {np.mean(total_dice_values):.4f}")
        
        logger.info(f"\n推理完成！结果保存在: {os.path.join(cfg.output_dir, cfg.DATASET.NAME, 'seg_results')}")

if __name__ == "__main__":
    main()
