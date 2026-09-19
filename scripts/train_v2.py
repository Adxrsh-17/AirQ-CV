import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from airq.data.v2_dataset import V2Dataset
from airq.models.v2_forecaster import V2Forecaster

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/raw_full438')
    parser.add_argument('--exp_dir', type=str, default='experiments/v2_baseline')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from best checkpoint if present')
    return parser.parse_args()

def main():
    args = get_args()
    
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)} | CuDNN Benchmark: Enabled", flush=True)
    
    print(f"Loading datasets with parallel multi-core preloading...", flush=True)
    train_ds = V2Dataset(args.data_dir, split='train', preload=True)
    val_ds = V2Dataset(args.data_dir, split='validation', preload=True)
    
    loader_kwargs = {
        "batch_size": args.batch_size,
        "pin_memory": torch.cuda.is_available(),
        "num_workers": args.num_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    
    print(f"Train sequences: {len(train_ds)} ({len(train_loader)} batches), Val sequences: {len(val_ds)} ({len(val_loader)} batches)", flush=True)
    
    model = V2Forecaster(hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss() # MAE Loss
    
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    best_val_loss = float('inf')
    ckpt_path = exp_dir / "best_model.pt"
    if args.resume and ckpt_path.exists():
        print(f"Resuming weights from existing checkpoint: {ckpt_path}", flush=True)
        try:
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
            print("Checkpoint successfully loaded! Continuing training...", flush=True)
        except Exception as e:
            print(f"Warning: could not load checkpoint ({e}), starting from scratch.", flush=True)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        optimizer.zero_grad()
        
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---", flush=True)
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            p1_x = batch['p1_x'].to(device, non_blocking=True)
            s2_x = batch['s2_x'].to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                preds = model(p1_x, s2_x)
                loss = criterion(preds, y)
                loss = loss / args.grad_accum
                
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            train_loss += loss.item() * args.grad_accum
            
            if (batch_idx + 1) % 15 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  [Batch {batch_idx+1}/{len(train_loader)}] Loss: {loss.item() * args.grad_accum:.4f}", flush=True)
                
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                p1_x = batch['p1_x'].to(device, non_blocking=True)
                s2_x = batch['s2_x'].to(device, non_blocking=True)
                y = batch['y'].to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    preds = model(p1_x, s2_x)
                    loss = criterion(preds, y)
                    
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} Completed in {epoch_time:.2f}s | Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f}", flush=True)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  --> Saved new best checkpoint to {ckpt_path} (Val MAE: {val_loss:.4f})", flush=True)
            
    print("\nTraining complete.", flush=True)

if __name__ == '__main__':
    main()
