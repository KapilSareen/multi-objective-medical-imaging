"""
Phase 1.5: Generate Predictions Cache
Runs inference with all 7 trained models to create P_cache.npy (N x 7 matrix)
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import json


TARGET_DISEASE = "Effusion"

MODEL_NAMES = [
    'densenet121', 'resnet50', 'resnet101',
    'efficientnet_b4', 'vgg16', 'inception_v3', 'mobilenet_v2'
]


class NIHDataset(Dataset):
    """Full NIH dataset for inference"""

    def __init__(self, csv_path, image_dir, transform=None, target_disease=TARGET_DISEASE):
        df = pd.read_csv(csv_path)

        # Filter for disease + healthy
        mask_disease = df['Finding Labels'].str.contains(target_disease, na=False)
        mask_healthy = df['Finding Labels'] == 'No Finding'
        df = df[mask_disease | mask_healthy].copy()
        df['target_label'] = df['Finding Labels'].str.contains(target_disease, na=False).astype(int)
        self.df = df.reset_index(drop=True)

        self.image_dir = Path(image_dir)
        self.transform = transform

        print(f"   Total samples: {len(self.df):,}")
        print(f"   Positive ({target_disease}): {self.df['target_label'].sum():,}")
        print(f"   Negative (No Finding): {(self.df['target_label']==0).sum():,}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_dir / row['Image Index']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(float(row['target_label']), dtype=torch.float32)
        gender = str(row.get('Patient Gender', 'U'))
        return image, label, gender, str(row['Image Index'])


def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def build_model(model_name):
    """Build model architecture (same as train_backbone.py)"""
    model_map = {
        'densenet121':     (tv_models.densenet121,     None, 224),
        'resnet50':        (tv_models.resnet50,         None, 224),
        'resnet101':       (tv_models.resnet101,        None, 224),
        'efficientnet_b4': (tv_models.efficientnet_b4,  None, 224),
        'vgg16':           (tv_models.vgg16_bn,         None, 224),
        'inception_v3':    (tv_models.inception_v3,     None, 299),
        'mobilenet_v2':    (tv_models.mobilenet_v2,     None, 224),
    }

    model_fn, _, img_size = model_map[model_name]
    # Load without pretrained weights (we'll load our trained weights)
    model = model_fn(weights=None)

    # Replace classifier to match training setup
    if model_name == 'densenet121':
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 1)
    elif model_name in ('resnet50', 'resnet101'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
    elif model_name == 'efficientnet_b4':
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
    elif model_name == 'vgg16':
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, 1)
    elif model_name == 'inception_v3':
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
        model.aux_logits = False
    elif model_name == 'mobilenet_v2':
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)

    return model, img_size


def run_inference(model, dataloader, device):
    """Run inference and return predictions, labels, genders, image_ids"""
    model.eval()
    all_preds, all_labels, all_genders, all_ids = [], [], [], []

    with torch.no_grad():
        for images, labels, genders, image_ids in tqdm(dataloader, desc="Inference"):
            images = images.to(device)
            outputs = model(images)
            # Flatten to 1D
            outputs = outputs.squeeze(-1)
            if outputs.dim() > 1:
                outputs = outputs[:, 0]
            preds = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_genders.extend(list(genders))
            all_ids.extend(list(image_ids))

    return np.array(all_preds), np.array(all_labels), all_genders, all_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--models_dir', type=str, default='models/backbones')
    parser.add_argument('--output_dir', type=str, default='data/cache')
    args = parser.parse_args()

    print("="*80)
    print("PHASE 1.5: GENERATE PREDICTIONS CACHE")
    print("="*80)

    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "raw" / "Data_Entry_2017_v2020.csv"
    image_dir = project_root / "data" / "raw" / "nih_images"
    models_dir = project_root / args.models_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Device: {device}")

    all_model_preds = []
    labels = None
    genders = None
    image_ids = None

    for model_name in MODEL_NAMES:
        print(f"\n{'='*80}")
        print(f"Model: {model_name.upper()}")

        checkpoint_path = models_dir / f"{model_name}.pt"
        if not checkpoint_path.exists():
            print(f"   Checkpoint not found: {checkpoint_path} - SKIPPING")
            continue

        # Build model and load weights
        model, img_size = build_model(model_name)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model = model.to(device)
        print(f"   Loaded: {checkpoint_path}")

        # Dataset
        transform = get_transform(img_size)
        if labels is None:
            # Build dataset once, reuse for all models
            dataset = NIHDataset(csv_path, image_dir, transform)
        else:
            # Rebuild dataset with new transform (img_size may differ for inception)
            dataset = NIHDataset(csv_path, image_dir, transform)

        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

        preds, lbls, gnds, ids = run_inference(model, dataloader, device)

        print(f"   Predictions: {len(preds):,}")
        print(f"   Mean: {preds.mean():.4f}, Std: {preds.std():.4f}")

        all_model_preds.append(preds)

        if labels is None:
            labels = lbls
            genders = gnds
            image_ids = ids

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    if not all_model_preds:
        print("\nNo models found! Train models first.")
        sys.exit(1)

    # Stack into (N x num_models)
    P_cache = np.stack(all_model_preds, axis=1)
    demographics = np.array(genders)

    print(f"\n{'='*80}")
    print(f"P_cache shape: {P_cache.shape}")
    print(f"Labels shape:  {labels.shape}")
    print(f"Demographics:  {demographics.shape}")

    # Save
    np.save(output_dir / "P_cache.npy", P_cache)
    np.save(output_dir / "y_true.npy", labels)
    np.save(output_dir / "demographics.npy", demographics)

    pd.DataFrame({
        'Image Index': image_ids,
        'y_true': labels,
        'gender': genders
    }).to_csv(output_dir / "predictions_metadata.csv", index=False)

    summary = {
        'num_models': len(all_model_preds),
        'num_samples': int(len(labels)),
        'model_names': MODEL_NAMES[:len(all_model_preds)],
        'P_cache_shape': list(P_cache.shape),
        'label_distribution': {
            'positive': int(labels.sum()),
            'negative': int(len(labels) - labels.sum())
        },
        'gender_distribution': {
            'M': int(np.sum(demographics == 'M')),
            'F': int(np.sum(demographics == 'F')),
        }
    }

    with open(output_dir / "predictions_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to: {output_dir}")
    print(f"  P_cache.npy, y_true.npy, demographics.npy")
    print(f"  predictions_metadata.csv, predictions_summary.json")
    print(f"\n{'='*80}")
    print("PHASE 1.5 COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
