import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def make_dataset(datasets,features,target):
    rus = RandomUnderSampler(random_state=10)

    final_dataset = pd.DataFrame()
    final_labels = pd.Series()

    for id,dataset in datasets.items():
        X = dataset[features]
        y = dataset[target]
        X_resampled, y_resampled = rus.fit_resample(X, y)
        final_dataset = pd.concat([final_dataset,X_resampled], ignore_index=True)
        final_labels = pd.concat([final_labels,y_resampled], ignore_index=True)
        # print(X_resampled.shape,type(y_resampled))

    # shuffle_idx = np.random.permutation(len(final_dataset))
    # X = final_dataset.iloc[shuffle_idx]
    # y = final_labels.iloc[shuffle_idx]
    X = final_dataset
    y = final_labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    # X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=21)
    return X_train, X_test, y_train, y_test