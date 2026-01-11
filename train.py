import torch
import torch.nn as nn
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import utils.utils as utils
from datasets.dataloader import DatasetSegmentation, collate_fn
from utils.processor import Samprocessor
from segment_anything import build_textsam_vit_b, build_textsam_vit_h, build_textsam_vit_l
from utils.lora import LoRA_Sam
import os
from time import time
import argparse
import random
import numpy as np
import logging
from utils.utils import load_cfg_from_cfg_file
from torch.cuda.amp import autocast, GradScaler
import contextlib
import gc

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        required=True,
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help="Whether to resume training"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="", 
        help="output directory"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    cfg = load_cfg_from_cfg_file(args.config_file)
    # 应用命令行参数覆盖配置
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

# Validation function with memory optimization
def evaluate_validation_loss(model, val_dataloader, device, seg_loss, ce_loss, use_amp=False):
    model.eval()
    val_losses = []
    
    autocast_context = autocast() if use_amp and device == 'cuda' else contextlib.nullcontext()
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            with autocast_context:
                outputs = model(batched_input=batch, multimask_output=False)
                stk_gt, stk_out = utils.stacking_batch(batch, outputs)
                stk_out = stk_out.squeeze(1)
                loss = seg_loss(stk_out, stk_gt.float().to(device)) + ce_loss(stk_out, stk_gt.float().to(device))
                val_losses.append(loss.item())
            
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    model.train()
    return mean(val_losses) if val_losses else 0.0

def print_memory_stats(device, logger=None):
    if device == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        message = f"GPU Memory - Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB"
        
        if logger:
            logger.info(message)
        else:
            print(message)
    return allocated if device == 'cuda' else 0

def main():
    cfg = get_arguments()
    
    # 确保输出目录存在
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}"), exist_ok=True)
    
    # 设置日志
    log_path = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}", "log.txt")
    logger = logger_config(log_path)
    logger.info("************")
    logger.info("** Config **")
    logger.info("************")
    logger.info(cfg)
    
    # 设置随机种子
    if cfg.seed >= 0:
        logger.info("Setting fixed seed: {}".format(cfg.seed))
        set_random_seed(cfg.seed)
    
    # ========== 设备设置 ==========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 清理CUDA缓存
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    print_memory_stats(device, logger)
    
    # ========== 加载模型 ==========
    classnames = cfg.PROMPT_LEARNER.CLASSNAMES
    
    logger.info(f"Loading SAM model: {cfg.SAM.MODEL}")
    try:
        if cfg.SAM.MODEL == "vit_b":
            sam = build_textsam_vit_b(cfg=cfg, checkpoint=cfg.SAM.CHECKPOINT, classnames=classnames)
        elif cfg.SAM.MODEL == "vit_l":
            sam = build_textsam_vit_l(cfg=cfg, checkpoint=cfg.SAM.CHECKPOINT, classnames=classnames)
        else:
            sam = build_textsam_vit_h(cfg=cfg, checkpoint=cfg.SAM.CHECKPOINT, classnames=classnames)
        logger.info("SAM model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading SAM model: {e}")
        raise
    
    # 创建SAM LoRA
    rank = getattr(cfg.SAM, 'RANK', 16)
    logger.info(f"Creating LoRA with rank: {rank}")
    sam_lora = LoRA_Sam(sam, rank)
    model = sam_lora.sam
    
    # 冻结backbone（可选）
    freeze_backbone = getattr(cfg.TRAIN, 'FREEZE_BACKBONE', False)
    if freeze_backbone and hasattr(model, 'image_encoder'):
        for name, param in model.image_encoder.named_parameters():
            param.requires_grad = False
        logger.info("Froze image encoder backbone")
    
    # ========== 数据准备 ==========
    processor = Samprocessor(model)
    
    logger.info("Loading training dataset...")
    try:
        train_ds = DatasetSegmentation(cfg, processor, mode="train", 
                                      num_shots=cfg.DATASET.NUM_SHOTS, 
                                      seed=cfg.seed)
    except Exception as e:
        logger.error(f"Error loading training dataset: {e}")
        raise
    
    logger.info("Loading validation dataset...")
    try:
        val_ds = DatasetSegmentation(cfg, processor, mode="val", 
                                     num_shots=cfg.DATASET.NUM_SHOTS, 
                                     seed=cfg.seed*cfg.seed)
    except Exception as e:
        logger.error(f"Error loading validation dataset: {e}")
        raise
    
    # 创建数据加载器
    num_workers = 0  # 设置为0以避免多进程问题
    pin_memory = False
    
    train_dataloader = DataLoader(train_ds, 
                                  batch_size=cfg.TRAIN.BATCH_SIZE, 
                                  shuffle=True, 
                                  collate_fn=collate_fn,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    
    val_dataloader = DataLoader(val_ds, 
                                batch_size=cfg.TRAIN.BATCH_SIZE, 
                                shuffle=False, 
                                collate_fn=collate_fn,
                                num_workers=num_workers,
                                pin_memory=pin_memory)
    
    logger.info(f"Train dataset size: {len(train_ds)}")
    logger.info(f"Val dataset size: {len(val_ds)}")
    logger.info(f"Train dataloader length: {len(train_dataloader)}")
    
    # ========== 优化器设置 ==========
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.TRAIN.LEARNING_RATE, weight_decay=0.01)
    
    # 混合精度训练设置
    use_amp = getattr(cfg.TRAIN, 'USE_AMP', False)
    scaler = GradScaler() if use_amp and device == 'cuda' else None
    
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    num_epochs = cfg.TRAIN.NUM_EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 记录可训练参数
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Number of trainable parameters: {trainable_count:,}")
    logger.info(f"Total parameters: {total_count:,}")
    logger.info(f"Trainable ratio: {trainable_count/total_count*100:.2f}%")
    
    # ========== 恢复训练功能 ==========
    results_name = (
        f"LORA{rank}_"
        f"SHOTS{cfg.DATASET.NUM_SHOTS}_"
        f"NCTX{cfg.PROMPT_LEARNER.N_CTX_TEXT}_"
        f"CSC{cfg.PROMPT_LEARNER.CSC}_"
        f"CTP{cfg.PROMPT_LEARNER.CLASS_TOKEN_POSITION}"
    )
    
    resume_path = os.path.join(
        cfg.output_dir,
        cfg.DATASET.NAME,
        "trained_models",
        f"seed{cfg.seed}",
        f"{results_name}_latest.pth"
    )
    
    start_epoch = 0
    best_loss = float("inf")
    
    if cfg.resume and os.path.exists(resume_path):
        logger.info(f"Loading checkpoint from: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint["best_loss"]
            if scaler and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
            logger.info(f"Loaded checkpoint from epoch {start_epoch}, best loss: {best_loss:.4f}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Starting training from scratch")
    
    # ========== 训练循环 ==========
    model.train()
    model.to(device)
    
    total_loss = []
    epoch_time = []
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    # 梯度累积步数
    gradient_accumulation = getattr(cfg.TRAIN, 'GRADIENT_ACCUMULATION', 1)
    
    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []
        epoch_start_time = time()
        
        accumulation_steps = 0
        
        # 训练进度条
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, batch in enumerate(train_pbar):
            # 混合精度前向传播
            if use_amp and device == 'cuda':
                with autocast():
                    outputs = model(batched_input=batch, multimask_output=False)
                    stk_gt, stk_out = utils.stacking_batch(batch, outputs)
                    stk_out = stk_out.squeeze(1)
                    loss = seg_loss(stk_out, stk_gt.float().to(device)) + ce_loss(stk_out, stk_gt.float().to(device))
                    loss = loss / gradient_accumulation  # 梯度累积归一化
                
                # 反向传播
                scaler.scale(loss).backward()
            else:
                # 标准前向传播
                outputs = model(batched_input=batch, multimask_output=False)
                stk_gt, stk_out = utils.stacking_batch(batch, outputs)
                stk_out = stk_out.squeeze(1)
                loss = seg_loss(stk_out, stk_gt.float().to(device)) + ce_loss(stk_out, stk_gt.float().to(device))
                loss = loss / gradient_accumulation  # 梯度累积归一化
                loss.backward()
            
            accumulation_steps += 1
            
            # 累积梯度后更新
            if accumulation_steps % gradient_accumulation == 0:
                if use_amp and device == 'cuda':
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                accumulation_steps = 0
            
            epoch_losses.append(loss.item() * gradient_accumulation)  # 重新缩放用于记录
            
            # 更新进度条
            train_pbar.set_postfix({
                'loss': f"{loss.item() * gradient_accumulation:.4f}",
                'acc_step': accumulation_steps
            })
            
            # 定期清理缓存
            if i % 10 == 0 and device == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
        
        # 处理剩余的梯度
        if accumulation_steps > 0:
            if use_amp and device == 'cuda':
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # 每个epoch结束后的操作
        epoch_end_time = time()
        epoch_time.append(epoch_end_time - epoch_start_time)
        mean_epoch_loss = mean(epoch_losses) if epoch_losses else 0
        
        # 验证阶段
        mean_val_loss = evaluate_validation_loss(model, val_dataloader, device, seg_loss, ce_loss, use_amp)
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        logger.info(f'EPOCH {epoch+1}/{num_epochs}: '
                   f'Train Loss: {mean_epoch_loss:.4f} | '
                   f'Val Loss: {mean_val_loss:.4f} | '
                   f'Time: {epoch_time[-1]:.1f}s | '
                   f'LR: {current_lr:.6f}')
        
        # 根据验证损失保存最佳模型
        if mean_val_loss < best_loss:
            logger.info(f"  New best validation loss: {best_loss:.4f} -> {mean_val_loss:.4f}")
            best_loss = mean_val_loss
            save_dict = {
                "model": model.state_dict(),
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
                "config": {k: v for k, v in cfg.items() if not callable(v)}  # 只保存可序列化的配置
            }
            if scaler:
                save_dict["scaler"] = scaler.state_dict()
            
            best_model_path = os.path.join(
                cfg.output_dir,
                cfg.DATASET.NAME,
                "trained_models",
                f"seed{cfg.seed}",
                f"{results_name}_best.pth"
            )
            torch.save(save_dict, best_model_path)
            logger.info(f"  Saved best model to: {best_model_path}")
        
        # 保存最新模型
        save_dict = {
            "model": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
            "config": {k: v for k, v in cfg.items() if not callable(v)}  # 只保存可序列化的配置
        }
        if scaler:
            save_dict["scaler"] = scaler.state_dict()
        
        latest_model_path = os.path.join(
            cfg.output_dir,
            cfg.DATASET.NAME,
            "trained_models",
            f"seed{cfg.seed}",
            f"{results_name}_latest.pth"
        )
        torch.save(save_dict, latest_model_path)
        
        # 记录内存使用情况
        print_memory_stats(device, logger)
        
        # 强制垃圾回收
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # 最终总结
    logger.info(f"Training completed!")
    logger.info(f"Total training time: {sum(epoch_time):.1f}s")
    logger.info(f"Average epoch time: {mean(epoch_time):.1f}s")
    logger.info(f"Best validation loss: {best_loss:.4f}")
    
    # 保存最终模型
    final_model_path = os.path.join(
        cfg.output_dir,
        cfg.DATASET.NAME,
        "trained_models",
        f"seed{cfg.seed}",
        f"{results_name}_final.pth"
    )
    
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": num_epochs,
        "best_loss": best_loss,
        "config": {k: v for k, v in cfg.items() if not callable(v)}  # 只保存可序列化的配置
    }, final_model_path)
    
    logger.info(f"Final model saved to: {final_model_path}")
    
    return best_loss

if __name__ == "__main__":
    main()
