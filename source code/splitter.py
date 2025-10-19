import json, numpy as np, pandas as pd
from typing import Tuple

#split dataset into train/validation/test sets using patient ids rather than individual images to avoid data leaking where a patient may appear both in training and testing
def split_by_patient(df: pd.DataFrame, ratios=(0.7,0.15,0.15), seed=42):
    #safety check to ensure ratios sum to 1.0 (with tolerance)
    assert abs(sum(ratios) - 1.0) < 1e-6
    #create a reproducible random generator
    rng = np.random.default_rng(seed)
    #collect unique patient ids
    patients = df["patient_id"].dropna().unique()
    #shuffle patient ids randomly
    rng.shuffle(patients)

    #compute how many patients go into each split
    n = len(patients)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)

    #slice the shuffled list into three sets
    train_p = set(patients[:n_train])
    val_p = set(patients[n_train:n_train+n_val])
    test_p = set(patients[n_train+n_val:])

    #helper function that assigns a split label based on patient id
    def assign(pid):
        if pid in train_p: return "train"
        if pid in val_p: return "val"
        return "test"

    #make a copy so the orginal DataFrame not modified directly
    df = df.copy()
    #apply the split assignment to every row
    df["split"] = df["patient_id"].map(assign)
    #sanity check to ensure we have exactly three split categories
    assert set(df["split"].unique()) == {"train", "val", "test"}
    #return updated DataFrame and a quick summary of patient counts
    return df, {"train": len(train_p), "val": len(val_p), "test": len(test_p)}

#write a json summary file recording how many images and unique patients are in each split.
def save_splits_json(df: pd.DataFrame, out_path: str):
    #build a dictionary with counts and unique patients total per split
    payload = {
        "counts": df["split"].value_counts().to_dict(),
        "patients_per_split": df.groupby("split")["patient_id"].nunique().to_dict()
    }
    #write that dictionary to a json file
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
