# File serves as the pipeline executor, calling the other modules
import yaml, pandas as pd, torch
from torch.utils.data import DataLoader
from src.data.indexer import build_index, filter_views
from src.data.splitter import split_by_patient, save_splits_json
from src.data.preprocess import make_transforms
from src.data.dataset import CXRDataset

def main(cfg_path = "config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    root = cfg["paths"]["data_root"]
    cls  = cfg["data"]["class_names"]
    keep = cfg["data"]["keep_views"]
    h, w = cfg["model"]["input_size"]

    #Building the index to store the information from the csv file
    print("Building index:")
    df = build_index(root, cls)
    print(f"Images in CSV: {len(df):,}; missing paths: {df['missing'].sum():,}")

    df = df[~df["missing"]].reset_index(drop=True)
    df = filter_views(df, tuple(keep))

    print("Splitting by patient:")
    df, patient_counts = split_by_patient(df, ratios=(
        cfg["splits"]["train"], cfg["splits"]["val"], cfg["splits"]["test"]), seed=cfg["seed"])
    print("Patients per split:", patient_counts)

    #save lightweight artifacts
    df.to_parquet(cfg["paths"]["index_out"])
    save_splits_json(df, cfg["paths"]["splits_out"])

    #dataloaders for training and validation data sets
    t_train = make_transforms((h,w), cfg["data"]["use_imagenet_norm"], train=True)
    t_eval  = make_transforms((h,w), cfg["data"]["use_imagenet_norm"], train=False)

    ds_train = CXRDataset(df, "train", cls, t_train)
    ds_val   = CXRDataset(df, "val",   cls, t_eval)

    dl_train = DataLoader(ds_train, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    xb, yb, meta = next(iter(dl_train))
    print(f"Train batch: x={tuple(xb.shape)}, y={tuple(yb.shape)} (N,C,H,W) = {xb.shape}")
    print(f"Example metas: {meta[:2] if isinstance(meta, list) else meta}")

if __name__ == "__main__":
    main()

