import os, pathlib, pandas as pd, numpy as np
from sys import meta_path
from tqdm import tqdm #may be used later for progress bars / graphs

#combine NIH chest x-ray metadata (CSV) with actual image files
def build_index(data_root: str, class_names):
    """
    data_root/
    images/  #NIH images
    Data_Entry_2017.csv  #NIH labels/metadata/patient info
    """

    #join the root path with CSV filename to get full path
    csv_path = os.path.join(data_root, 'Data_Entry_2017.csv')
    #check if the CSV file exists. if not, programs stops and raise error
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    #load the CSV metadata into a DataFrame
    meta = pd.read_csv(csv_path)

    #rename columns for cleaner access within python
    meta = meta.rename(columns={
        "Image Index": "image_id", #filename of x-ray image
        "Finding Labels": "label_raw", #original string of disease labels
        "Patient ID": "patient_id", #id of the patient
        "View Position": "view" #x-ray view type (PA, AP, etc)
    })

    #build the image directory path
    img_dir = os.path.join(data_root, 'images')
    if not os.path.exists(img_dir):
        img_dir = data_root

    #walk through the directory and build a map of filenames -> absolute paths
    path_map = {}
    for root, _, files in os.walk(img_dir):
        for f in files:
            #include only png/jpp/jpeg files (ignoring others)
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                path_map[f] = os.path.join(root, f)

    #map each image_id from the CSV to its actual file path
    meta["path"] = meta["image_id"].map(path_map.get)
    #create a boolean column showing which images are missing
    meta["missing"] = meta["path"].isna()

    #convert disease labels strings into encoded vectors
    def to_multihot(s):
        #split label string by '|' and remove whitespace
        tags = {t.strip() for t in str(s).split('|')}
        #return 1 if class name exists in tags, else 0
        return [1 if c in tags else 0 for c in class_names]

    #apply to all rows and create an array of one-hot encoded labels
    mh = np.stack(meta["label_raw"].apply(to_multihot).values)
    #add each disease label as a separate binary column
    for i, c in enumerate(class_names):
        meta[c] = mh[:, i]

    #normalize the 'view' column (ex: "pa " -> "PA")
    meta["view"] = meta["view"].astype(str).str.upper().str.strip()
    #return the final merged dataset with paths + patient info + encoded labels
    return meta

#keeps only images from specific x-ray views
def filter_views(df: pd.DataFrame, keep=("PA", "AP")):
    #filter DataFrame rows to include only those view types
    return df[df["view"].isin(keep)].reset_index(drop=True)
