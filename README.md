# ML_Notesfree


## Preprocessing

Loading of Data uses the librosa library to read audio files and perform a variety of feature extraction. 

```py
features = ['spectral_centroid', 'spectral_bandwidth', 'spectral_contrast', 'spectral_rolloff', 'spectral_flatness', 'poly_features', 'rms', 'zero_crossing_rate', 'chroma_stft', 'chroma_cqt', 'chroma_cens', 'melspectrogram', 'mfcc', 'tonnetz']
from notesfree.preprocessing import load
FEAT_EXT = load.get_feature_extractors(features)
# If fewer features are required, edit the features variable, if no argument is given for get_feature_extractors, the code will load all the features
train_df, val_df, test_df = load_data("./data/train",16000,FEAT_EXT)
```

