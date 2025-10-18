import torch, pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CXRDataset(Dataset):
    def __init__(self, index_df: pd.DataFrame, split: str, class_names, transforms):
        self.df = index_df[index_df["split"] == split].reset_index(drop=True)
        self.class_names = class_names
        self.t = transforms

    def __len__(self): return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(row.path).convert("RGB") #ensure 3-channel
        x = self.t(img)
        y = torch.tensor(row[self.class_names].values.astype("float32"))
        meta = {"image_id": row.image_id, "patient_id": row.patient_id}
        return x, y, meta
