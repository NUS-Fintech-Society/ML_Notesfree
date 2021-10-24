import pandas as pd
import numpy as np

from pathlib import Path, PosixPath
import pathlib

from tqdm import tqdm
import os
import librosa
from scipy.io import wavfile
from sklearn.pipeline import Pipeline


def load_data(path, SAMPLE_RATE=16000):
    train_df = read_files(Path(path,'audio'),SAMPLE_RATE)

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
    print(train_df["data"].apply(lambda x: x.shape))
    return train_df, val_df, test_df

def read_files(path, SAMPLE_RATE=16000) -> pd.DataFrame:
    tmp_list = []
    for (dirpath, dirnames, filenames) in tqdm(os.walk(path),total=30):
        for file in filenames:
            if file.endswith('.wav'):
                tmp_path=Path(dirpath, file)
                class_label = tmp_path.parts[-2]
                data, samp_freq = librosa.load(tmp_path)
                f_path = str("/".join(tmp_path.parts[-2:]))
                tmp_list.append([tmp_path,f_path,class_label,samp_freq,data])
            else:
                continue
    return  pd.DataFrame(tmp_list,
        columns=['full_file_path','file_path','class_label','sample_freq','data']
    )

if __name__ == "__main__":
    load_data("./data/train")