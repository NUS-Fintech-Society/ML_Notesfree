import pandas as pd
import numpy as np

from pathlib import Path, PosixPath
import pathlib

from tqdm import tqdm
import os
import librosa
from scipy.io import wavfile
from sklearn.pipeline import Pipeline
from typing import Optional, Dict, List, Callable, Tuple
import inspect
import librosa.feature.spectral  as features

def get_feature_extractors(feats: Optional[List[str]] = None)-> Dict[str,Callable]:
    """Returns a Dictionary of Function Name: Function
    This is based on all of the feature extractions in librosa

    Returns:
        Dict[str,Callable]: [description]
    """
    if feats is None:
        return {i:features.__dict__.get(i) for i in features.__all__}
    else:
        return {i:features.__dict__.get(i) for i in feats}

def pad_data(df, SAMPLE_RATE):
    pad_df = df.copy()
    for i,row in df.iterrows():
        length = row["data_len"]
        cls = row["class_label"]
        data = row["data"]
        if length < SAMPLE_RATE:
            #Padding
            tmp = np.zeros(SAMPLE_RATE)
            tmp[:data.shape[0]]=data
            pad_df.at[i,'data']= tmp

        elif length > SAMPLE_RATE and cls != "_background_noise_":
            pad_df.iloc[i,'data'] = data[:SAMPLE_RATE]
    pad_df["data_len"] = pad_df["data"].apply(len)
    return pad_df
    
def read_files(path, SAMPLE_RATE=16000) -> pd.DataFrame:
    tmp_list = []
    for (dirpath, dirnames, filenames) in tqdm(os.walk(path),total=30):
        for file in filenames:
            if file.endswith('.wav'):
                tmp_path=Path(dirpath, file)
                class_label = tmp_path.parts[-2]
                data, samp_freq = librosa.load(tmp_path, SAMPLE_RATE)
                f_path = str("/".join(tmp_path.parts[-2:]))
                tmp_list.append([tmp_path,f_path,class_label,samp_freq,data])
            else:
                continue
    return  pd.DataFrame(tmp_list,
        columns=['full_file_path','file_path','class_label','sample_freq','data']
    )

def load_data(
        path, SAMPLE_RATE=16000, features:Optional[Dict[str,Callable]] = None
    ) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    train_df = read_files(Path(path,'audio'),SAMPLE_RATE)
    train_df["data_len"] = train_df["data"].apply(len)
    train_df = pad_data(train_df,SAMPLE_RATE)
    for name, fun in features.items():
        train_df["name"] = train_df["data"].apply(fun)

    label_map = {j:i for i,j in enumerate(train_df["class_label"].unique())}
    train_df['label_int'] = train_df["class_label"].map(label_map)
    val_txt = open(Path(path,"validation_list.txt")).readlines()
    val_files = [i.strip() for i in val_txt]

    test_txt = open(Path(path,"testing_list.txt")).readlines()
    test_files = [i.strip() for i in test_txt]
    val_df = train_df[train_df["file_path"].isin(val_files)].copy()
    test_df = train_df[train_df["file_path"].isin(test_files)].copy()

    train_df = train_df[
        ~(train_df["file_path"].isin(val_files)) & 
        ~(train_df["file_path"].isin(test_files))
    ]
    return train_df, val_df, test_df

if __name__ == "__main__":
    FEATURES = [
        "spectal_centroid"
    ]
    print(features.__all__)
    FEAT_EXT = get_feature_extractors()
    train_df, val_df, test_df = load_data("./data/train",16000,FEAT_EXT)
    print(train_df.columns)
