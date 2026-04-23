"""
Phase 1: Train CNN Backbone on NIH Coreset
Fine-tunes TorchXRayVision models for Pleural Effusion detection
Includes checkpointing for 24h Slurm time limits
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class NIHCoresetDataset(Dataset):
    """Dataset for NIH coreset images"""
    
    def __init__(self, csv_path, image_dir, transform=None, split='train', target_disease='Effusion'):
        df_full = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.split = split

        # Create target_label if not present (raw NIH metadata)
        if 'target_label' not in df_full.columns:
            mask_disease = df_full['Finding Labels'].str.contains(target_disease, na=False)
            mask_healthy = df_full['Finding Labels'] == 'No Finding'
            df_full = df_full[mask_disease | mask_healthy].copy()
            df_full['target_label'] = df_full['Finding Labels'].str.contains(target_disease, na=False).astype(int)

        self.df = df_full.reset_index(drop=True)

        # Create train/val split (stratified by label and gender)
        if 'Patient Gender' in self.df.columns:
            stratify = self.df['target_label'].astype(str) + '_' + self.df['Patient Gender'].astype(str)
        else:
            stratify = self.df['target_label']
        
        train_idx, val_idx = train_test_split(
            np.arange(len(self.df)),
            test_size=0.2,
            random_state=42,
            stratify=stratify
        )
        
        if split == 'train':
            self.df = self.df.iloc[train_idx].reset_index(drop=True)
        else:
            self.df = self.df.iloc[val_idx].reset_index(drop=True)
        
        print(f"   {split.upper()} set: {len(self.df):,} images")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_dir / row['Image Index']

        # Load image as RGB (works for both xrv and torchvision models)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(float(row['target_label']), dtype=torch.float32)

        return image, label


def get_transforms(image_size=224, augment=True):
    """Get train/val transforms"""
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def get_model(model_name, num_classes=1):
    """Load pretrained torchvision model for chest X-ray classification.
    All models use ImageNet pretrained weights and accept 3-channel 224x224 input.
    """
    import torchvision.models as tv_models
    print(f"\n🏗️  Loading model: {model_name}")

    model_map = {
        'densenet121':     (tv_models.densenet121,     tv_models.DenseNet121_Weights.DEFAULT,     224),
        'resnet50':        (tv_models.resnet50,         tv_models.ResNet50_Weights.DEFAULT,         224),
        'resnet101':       (tv_models.resnet101,        tv_models.ResNet101_Weights.DEFAULT,        224),
        'efficientnet_b4': (tv_models.efficientnet_b4,  tv_models.EfficientNet_B4_Weights.DEFAULT,  224),
        'vgg16':           (tv_models.vgg16_bn,         tv_models.VGG16_BN_Weights.DEFAULT,         224),
        'inception_v3':    (tv_models.inception_v3,     tv_models.Inception_V3_Weights.DEFAULT,     299),
        'mobilenet_v2':    (tv_models.mobilenet_v2,     tv_models.MobileNet_V2_Weights.DEFAULT,     224),
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")

    model_fn, weights_enum, img_size = model_map[model_name]
    model = model_fn(weights=weights_enum)

    # Replace final classifier for binary output
    if model_name == 'densenet121':
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name in ('resnet50', 'resnet101'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'efficientnet_b4':
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == 'vgg16':
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    elif model_name == 'inception_v3':
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model.aux_logits = False
    elif model_name == 'mobilenet_v2':
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    print(f"   Source: Torchvision ImageNet pretrained")
    print(f"   Input size: {img_size}x{img_size}")
    return model, img_size


def save_checkpoint(epoch, model, optimizer, scheduler, auc, checkpoint_path):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'auc': auc,
    }, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device=None):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path,
                            map_location=device or 'cpu',
                            weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('auc', 0.0)


def process_output(outputs):
    """Flatten model output to 1D tensor for binary classification"""
    if outputs.dim() > 1:
        outputs = outputs.squeeze(-1)
    if outputs.dim() > 1:
        outputs = outputs[:, 0]
    return outputs.float()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with tqdm(dataloader, desc="Training", unit="batch") as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            outputs = process_output(outputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(dataloader)
    # Guard against all-same-class batches
    try:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        epoch_auc = 0.5
    return epoch_loss, epoch_auc


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        with tqdm(dataloader, desc="Validation", unit="batch") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                outputs = process_output(outputs)

                loss = criterion(outputs, labels)
                running_loss += loss.item()

                preds = torch.sigmoid(outputs).cpu().numpy().flatten()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / len(dataloader)
    try:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        epoch_auc = 0.5
    return epoch_loss, epoch_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['densenet121', 'resnet50', 'resnet101', 
                               'efficientnet_b4', 'vgg16', 'inception_v3', 'mobilenet_v2'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='models/checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    args = parser.parse_args()
    
    print("="*80)
    print(f"PHASE 1: TRAIN {args.model_name.upper()}")
    print("="*80)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    # Use full dataset (coreset skipped - 224x224 images are already small)
    coreset_csv = project_root / "data" / "raw" / "Data_Entry_2017_v2020.csv"
    image_dir = project_root / "data" / "raw" / "nih_images"
    checkpoint_dir = project_root / args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load model
    model, img_size = get_model(args.model_name)
    model = model.to(device)
    
    # Datasets
    print(f"\n📊 Loading datasets...")
    train_transform = get_transforms(img_size, augment=True)
    val_transform = get_transforms(img_size, augment=False)
    
    train_dataset = NIHCoresetDataset(coreset_csv, image_dir, train_transform, split='train')
    val_dataset = NIHCoresetDataset(coreset_csv, image_dir, val_transform, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Optimizer and loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume from checkpoint if exists
    start_epoch = 0
    best_auc = 0.0
    checkpoint_path = checkpoint_dir / f"{args.model_name}_latest.pt"
    
    if args.resume and checkpoint_path.exists():
        print(f"\n♻️  Resuming from checkpoint: {checkpoint_path}")
        start_epoch, best_auc = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
        print(f"   Starting from epoch {start_epoch + 1}, best AUC: {best_auc:.4f}")
    
    # Training loop
    print(f"\n🚀 Starting training...")
    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': []}
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Log metrics
        print(f"\n📊 Epoch {epoch + 1} Results:")
        print(f"   Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")
        print(f"   Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        # Save checkpoint every epoch
        save_checkpoint(epoch + 1, model, optimizer, scheduler, val_auc, checkpoint_path)
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_path = checkpoint_dir / f"{args.model_name}_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"   💾 New best model saved (AUC: {val_auc:.4f})")
    
    # Save final model
    final_path = project_root / "models" / "backbones" / f"{args.model_name}.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path)
    
    # Save training history
    history_path = checkpoint_dir / f"{args.model_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"📊 Best Val AUC: {best_auc:.4f}")
    print(f"💾 Model saved: {final_path}")
    print(f"📈 History saved: {history_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
