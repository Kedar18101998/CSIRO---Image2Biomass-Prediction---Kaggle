"""
CSIRO Biomass Prediction using DINOv3 + Metric-Optimized Ensemble

Author: Kedar Kale
Competition: CSIRO Biomass Estimation (Kaggle)
Model: DINOv3 ConvNeXt-L + Ridge + SVR + LightGBM
"""

# =========================================================================================
# CSIRO Biomass — DINOv3 Metric-Optimized Ensemble
#
# IMPROVEMENTS:
# 1. High Res (448px) for texture detail.
# 2. Metric-Aware Training: Uses SQRT transform for high-weight targets (Total/GDM)
#    to maximize the R2 score, instead of Log which suppresses it.
# 3. TTA: Horizontal Flip averaging.
# =========================================================================================

import os
import sys
import gc
import cv2
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T 

# Machine Learning Imports
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# ====================================================
# 1. CONFIGURATION
# ====================================================
# 🔴 Adjust these paths if your Kaggle input folder names are different
REPO_PATH = "/kaggle/input/dinov3-version-git/pytorch/default/1/dinov3-main"
WEIGHTS_PATH = "/kaggle/input/dinov3/pytorch/convnext_large/1/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth"

sys.path.append(REPO_PATH)

class Config:
    SEED = 42
    # Batch size 2 is safer for 448px images on Kaggle GPU
    BATCH_SIZE = 2 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 🔴 High Resolution for Texture Analysis
    IMG_SIZE = 448 
    
    DATA_DIR = Path("/kaggle/input/csiro-biomass")
    TRAIN_CSV = DATA_DIR / "train.csv"
    TEST_CSV  = DATA_DIR / "test.csv"
    IMG_DIR   = DATA_DIR / "train"
    TEST_IMG_DIR = DATA_DIR / "test"
    
    TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(Config.SEED)

# ====================================================
# 2. MANUAL TRANSFORMS
# ====================================================
def get_transforms():
    """
    Standard ImageNet normalization used by DINO.
    """
    return T.Compose([
        T.ToPILImage(),
        T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

base_transform = get_transforms()

# ====================================================
# 3. MODEL WRAPPER
# ====================================================
class DINOv3ConvNextWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, x):
        out = self.model(x)
        
        # Handle dictionary outputs common in DINOv3
        if isinstance(out, dict):
            if 'res4' in out: out = out['res4']
            elif 'x_norm_clstoken' in out: return out['x_norm_clstoken']
            else: out = list(out.values())[0]

        # Smart Pooling: Only pool if output is a feature map (4D)
        if out.ndim == 4: 
            return out.mean(dim=[-2, -1])
            
        return out

# ====================================================
# 4. LOAD MODEL
# ====================================================
def get_model():
    print(f"Loading Architecture...")
    try:
        # Try loading from local repo path using torch.hub
        model = torch.hub.load(REPO_PATH, 'dinov3_convnext_large', source='local', pretrained=False)
    except Exception as e:
        print(f"Hub load failed ({e}), falling back to direct import...")
        sys.path.append(REPO_PATH)
        from hubconf import dinov3_convnext_large
        model = dinov3_convnext_large(pretrained=False)
    
    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading Weights from {WEIGHTS_PATH}...")
        state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print("❌ ERROR: Weights file not found! Check WEIGHTS_PATH.")
    
    model.to(Config.DEVICE).eval()
    return DINOv3ConvNextWrapper(model)

# ====================================================
# 5. DATA PROCESSING
# ====================================================
def clean_and_load_image(path):
    img = cv2.imread(str(path))
    if img is None: return np.zeros((512, 512, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Crop bottom 15% to remove the color calibration card
    h, w, _ = img.shape
    new_h = int(h * 0.85) 
    return img[:new_h, :, :]

def load_data():
    train = pd.read_csv(Config.TRAIN_CSV)
    test  = pd.read_csv(Config.TEST_CSV)
    
    # Pivot to get one row per image
    train_w = train.pivot(index="image_path", columns="target_name", values="target").reset_index()
    
    test["target"] = 0.0
    test_w = test.pivot(index="image_path", columns="target_name", values="target").reset_index()
    
    return train_w, test_w, test

train_w, test_w, test_base = load_data()

# ====================================================
# 6. FEATURE EXTRACTION (TTA)
# ====================================================
def extract_features_tta(df, img_dir, model):
    all_feats = []
    paths = [img_dir / Path(p).name for p in df["image_path"]]
    
    print(f"Starting extraction for {len(paths)} images...")
    
    for i in tqdm(range(0, len(paths), Config.BATCH_SIZE), desc="Extracting"):
        batch_paths = paths[i : i + Config.BATCH_SIZE]
        
        # 1. Load Images
        imgs_np = [clean_and_load_image(p) for p in batch_paths]
        
        # 2. Transform to Tensor
        tensors_orig = torch.stack([base_transform(img) for img in imgs_np]).to(Config.DEVICE)
        
        # 3. TTA: Horizontal Flip
        tensors_flip = torch.flip(tensors_orig, dims=[3]) 
        
        with torch.no_grad():
            feat_orig = model(tensors_orig)
            feat_flip = model(tensors_flip)
            
            # Average predictions to reduce noise
            feat_avg = (feat_orig + feat_flip) / 2.0
            all_feats.append(feat_avg.cpu().numpy())
            
        if i % 10 == 0: 
            gc.collect()
            torch.cuda.empty_cache()

    if not all_feats: return np.zeros((0, 1536))
    return np.vstack(all_feats)

# --- EXECUTE EXTRACTION ---
model = get_model()

print("\n>>> 1. Extracting TRAIN Features...")
X_train_raw = extract_features_tta(train_w, Config.IMG_DIR, model)

print("\n>>> 2. Extracting TEST Features...")
X_test_raw = extract_features_tta(test_w, Config.TEST_IMG_DIR, model)

del model, base_transform
gc.collect()
torch.cuda.empty_cache()

# ====================================================
# 7. METRIC-OPTIMIZED ENSEMBLE TRAINING
# ====================================================
print("\n>>> 3. Training Metric-Optimized Ensemble...")

if X_train_raw.shape[0] != len(train_w):
    train_w = train_w.iloc[:X_train_raw.shape[0]]

# Define Heavy Targets based on Competition Weights
# Dry_Total_g (0.5) and GDM_g (0.2)
HEAVY_TARGETS = ["Dry_Total_g", "GDM_g"]

test_preds_final = np.zeros((X_test_raw.shape[0], len(Config.TARGETS)))
kf = KFold(n_splits=10, shuffle=True, random_state=Config.SEED)

for t_idx, target in enumerate(Config.TARGETS):
    y_raw_target = train_w[target].values
    
    # 🔴 KEY CHANGE: Transform Logic
    # SQRT for heavy targets helps predict larger values accurately (better R2)
    # Log1p for lighter targets smoothes noise
    if target in HEAVY_TARGETS:
        print(f"Target: {target:12s} | Mode: SQRT Transform (Maximizing R2)")
        y_curr = np.sqrt(y_raw_target)
    else:
        print(f"Target: {target:12s} | Mode: Log1p Transform (Smoothing Noise)")
        y_curr = np.log1p(y_raw_target)
        
    fold_preds = []
    fold_scores = []
    
    # Ensemble Models
    model_ridge = Ridge(alpha=0.5) 
    model_svr = SVR(C=5.0, epsilon=0.05) 
    model_lgbm = LGBMRegressor(n_estimators=2000, learning_rate=0.005, num_leaves=31, verbose=-1, random_state=42)
    
    ensemble = VotingRegressor([
        ('ridge', model_ridge), 
        ('svr', model_svr), 
        ('lgbm', model_lgbm)
    ])
    
    for tr_idx, va_idx in kf.split(X_train_raw, y_curr):
        X_tr, X_va = X_train_raw[tr_idx], X_train_raw[va_idx]
        y_tr, y_va = y_curr[tr_idx], y_curr[va_idx]
        
        ensemble.fit(X_tr, y_tr)
        
        val_pred = ensemble.predict(X_va)
        fold_scores.append(np.sqrt(mean_squared_error(y_va, val_pred)))
        fold_preds.append(ensemble.predict(X_test_raw))
        
    print(f"Target: {target:12s} | CV RMSE: {np.mean(fold_scores):.4f}")
    
    # Aggregate and Inverse Transform
    avg_preds = np.mean(fold_preds, axis=0)
    
    if target in HEAVY_TARGETS:
        # Inverse SQRT is Square
        test_preds_final[:, t_idx] = np.square(np.maximum(avg_preds, 0))
    else:
        # Inverse Log is Expm1
        test_preds_final[:, t_idx] = np.maximum(np.expm1(avg_preds), 0)

# ====================================================
# 8. SUBMISSION GENERATION
# ====================================================
test_w[Config.TARGETS] = test_preds_final

# Physics Constraints
test_w["Dry_Clover_g"] = np.maximum(test_w["GDM_g"] - test_w["Dry_Green_g"], 0)
test_w["Dry_Dead_g"]   = np.maximum(test_w["Dry_Total_g"] - test_w["GDM_g"], 0)

pred_map = {row['image_path']: {t: row[t] for t in Config.TARGETS} for _, row in test_w.iterrows()}
submission_rows = []

for _, row in test_base.iterrows():
    val = pred_map.get(row['image_path'], {}).get(row['target_name'], 0.0)
    submission_rows.append({'sample_id': row['sample_id'], 'target': val})

sub_df = pd.DataFrame(submission_rows)
sub_df.to_csv("submission.csv", index=False)

print("\n✅ Metric-Aware Submission Saved!")
print(sub_df.head())