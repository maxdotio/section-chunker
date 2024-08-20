from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier

def build_LR_model():
    lr = LogisticRegression(random_state = 15)
    return lr

def build_LRCV_model():
    params = {"Cs":[1, 10, 100], "cv": [5, 10, None]}
    lr_cv = LogisticRegressionCV(random_state = 26, max_iter = 500) # ,solver="liblinear"
    return lr_cv, params

def build_RF_model():
    params = {"n_estimators":[10, 100, 200, 500], "max_features": [2, 4, 6, "sqrt"]}
    rf = RandomForestClassifier(random_state = 45)
    return rf, params