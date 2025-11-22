import os, yaml, time, math, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.data import Subset
from tqdm.auto import tqdm

from src.data.dataset import CXRDataset
from src.data.preprocess import make_transforms
from src.data.indexer import build_index, filter_views
from src.data.splitter import split_by_patient
from train.model_factor import build_model
from train.metrics import multilabel_scores

torch.backends.cudnn.benchmark = True

def seed_all(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

def make_loaders(df, class_names, input_size, batch_size, workers, subset=None):
    t_train = make_transforms(input_size, use_imagenet_norm=True, train=True)
    t_eval = make_transforms(input_size, use_imagenet_norm=True, train=False)
    #t_train = make_transforms((h, w), use_imagenet_norm=cfg["data"]["use_imagenet_norm"], train=True)
    #t_eval = make_transforms((h, w), use_imagenet_norm=cfg["data"]["use_imagenet_norm"], train=False    )

    ds_train = CXRDataset(df, "train", class_names, t_train)
    ds_val = CXRDataset(df, "val", class_names, t_eval)


    if subset: #
        subset = int(subset)
        ds_train = Subset(ds_train, list(range(min(subset, len(ds_train)))))

    pin = torch.cuda.is_available()

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin, persistent_workers=bool(workers))
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin, persistent_workers=bool(workers))
    return dl_train, dl_val

def evaluate(model, dl, device, class_names):
    model.eval()
    y_true, y_prob = [], []

    #
    pbar = tqdm(dl, desc="Evaluation", leave=False)

    with torch.no_grad():
        for xb, yb, _ in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            prob = torch.sigmoid(logits)
            y_true.append(yb.cpu().numpy())
            y_prob.append(prob.cpu().numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)
    return multilabel_scores(y_true, y_prob)

def train_one_epoch(model, dl, device, optimizer, loss_fn):
    model.train()
    running = 0.0
    n_seen = 0

    #
    pbar = tqdm(dl, desc="Train", leave=False)

    for xb, yb, _ in pbar:
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        #
        batch_size = xb.size(0)
        running += loss.item() * xb.size(0)
        n_seen += batch_size
        pbar.set_postfix(loss=running / max(n_seen, 1))

    return running /len(dl.dataset)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--subset", type=int, default=0, help="Train on N samples (0=all) for a smoke run")
    ap.add_argument("--arch", default="densenet121")
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed_all(cfg.get("seed", 42))

    #
    root = cfg["paths"]["data_root"]
    cls = cfg["data"]["class_names"]
    keep = tuple(cfg["data"].get("keep_views", ["PA", "AP"]))

    print("Building index...")
    df = build_index(root, cls)
    df = df[~df["missing"]].reset_index(drop=True)
    df = filter_views(df, keep)
    df, _ = split_by_patient(df, ratios=(cfg["splits"]["train"], cfg["splits"]["val"], cfg["splits"]["test"]), seed=cfg["seed"])

    #
    input_h, input_w = cfg["model"]["input_size"]
    dl_train, dl_val, = make_loaders(df, cls, (input_h, input_w), args.batch, args.workers, subset=args.subset)

    #
    device = get_device()
    print("Device:", device)
    model = build_model(args.arch, num_outputs=len(cls), pretrained=bool(args.pretrained)).to(device)

    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"].get("weight_decay", 0.0))

    #
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    #optimizer = AdamW(model.parameters(), lr=cfg["train"].get("lr", 1e-4), weight_decay=cfg["train"].get("weight_decay", 1e-4))
    loss_fn = nn.BCEWithLogitsLoss()

    #
    best_auc = -1.0
    os.makedirs(cfg["paths"]["models"], exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, dl_train, device, optimizer, loss_fn)
        scores = evaluate(model, dl_val, device, cls)
        dt = time.time() - t0
        msg = f"[{epoch}/{args.epochs}] loss={train_loss:.4f} AUCROC={scores['auc_roc_macro']:.4f} AUCPR={scores['auc_pr_macro']:.4f}  time={dt:.1f}s"
        print(msg)

        #
        auc = scores.get("auc_roc_macro", float("nan"))
        if not math.isnan(auc) and auc > best_auc:
            best_auc = auc
            out = os.path.join(cfg["paths"]["models"], f"best_{args.arch}.pt")
            torch.save(model.state_dict(), out)
            print("saved", out)

if __name__ == "__main__":
    main()

