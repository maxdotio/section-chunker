from feature_engineering import *
from make_dataset import *
from models import *
from evaluate import *
from visualize import *
from train import *
import joblib
import pandas as pd

if __name__== "__main__":
    # cursor = connect_dbase()
    # records = get_records(cursor)
    # unq_ids = get_unique_ids(records)

    records = pd.read_csv('app\\section-identifier\\data\\label_data.csv')
    unq_ids = records["documentId"].unique()

    patterns = [r"^\d{1,3}[\.,]+\s*[A-Za-z\s|\-,_]+",r"^\([A-Z]\)\s*[\w\s]+",r"^[A-Z]\.\s*[\w\s]+",r"^Part\s+[IVXLC]+\s*-\s*[\w\s]+",r"^Part\s+[ivxlc]+\s*-\s*[\w\s]+",r"^\d{1,3}\.?\s*[\w\s]+[\.,]",r"^\d{1,3}\.\d+\s*[\w\s]+\.",r"^[0-9]+\.[0-9]+",r"^Section\s+\d+",r"^Section\s+\d+\.\d+",r"^Section\s+\d+\.\d+\.",r"^Part\s+\d+[a-zA-Z]",r"^Part\s+\d+[a-zA-Z]\.",r"^Article\s+\d+\.\d+",r"^Article\s+[IVXLC]+",r"^Article\s+[IVXLC]+\s*-\s*[\w\s]+",r"^\s*[I\d]+[\.]?\s*$"]
    discards = [r"[\$\%]+",r"^\d+(st|rd|th|nd)",r"^\d+\s*(sq\s*ft|square feet|sq\.\s*ft\.)",r"^\d{5}(-\d{4})?"]

    columns = ["length","content","label","line_height","line_gap","sorted_line_gap","left-align","prev_line_diff","next_line_diff","pattern","discarded_pattern", "normalised_line_height","normalised_line_gap", "line_number", "pattern_type"] # ,"offset","x1","x2","y1","y2"

    datasets, pages_data = make_features(records, unq_ids, patterns, discards, columns)

    features = ["normalised_line_height","normalised_line_gap","sorted_line_gap","left-align","next_line_diff", "pattern", "line_number", "pattern_type"] 
    target = "label"

    X_train, X_test, y_train, y_test = make_dataset(datasets, features, target)

    model, params = build_RF_model()

    model = train_model(model, params, 5, X_train, y_train)

    model = model.fit(X_train, y_train)

    joblib.dump(model, r'C:\Users\Aashrith\CoE Internship\real-assistant\ML experiments\section_chunker_model_v2.pkl')

    make_eval_report(model, X_test, y_test)

    ## Testing

    # train_dataset = X_train
    # train_dataset["label"] = y_train
    # print(train_dataset.isna().all())
    # make_report(train_dataset,1)