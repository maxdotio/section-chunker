import sys
from pathlib import Path
sys.path.insert(1, 'section-identifier')
sys.path.insert(2, 'section-merger')

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
import joblib
from feature_engineering import extract_features
from make_sections import *
from merge_sections import *
import json
from typing import List, Dict

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

patterns = [r"^\d{1,3}[\.,]+\s*[A-Za-z\s|\-,_]+",r"^\([A-Z]\)\s*[\w\s]+",r"^[A-Z]\.\s*[\w\s]+",r"^Part\s+[IVXLC]+\s*-\s*[\w\s]+",r"^Part\s+[ivxlc]+\s*-\s*[\w\s]+",r"^\d{1,3}\.?\s*[\w\s]+[\.,]",r"^\d{1,3}\.\d+\s*[\w\s]+\.",r"^[0-9]+\.[0-9]+",r"^Section\s+\d+",r"^Section\s+\d+\.\d+",r"^Section\s+\d+\.\d+\.",r"^Part\s+\d+[a-zA-Z]",r"^Part\s+\d+[a-zA-Z]\.",r"^Article\s+\d+\.\d+",r"^Article\s+[IVXLC]+",r"^Article\s+[IVXLC]+\s*-\s*[\w\s]+",r"^\s*[I\d]+[\.]?\s*$"]
discards = [r"[\$\%]+",r"^\d+(st|rd|th|nd)",r"^\d+\s*(sq\s*ft|square feet|sq\.\s*ft\.)",r"^\d{5}(-\d{4})?"]
columns = ["length","content","line_height","line_gap","sorted_line_gap","left-align","prev_line_diff","next_line_diff","pattern","discarded_pattern", "normalised_line_height","normalised_line_gap", "line_number", "pattern_type"] # ,"offset","x1","x2","y1","y2"
features = ["normalised_line_height","normalised_line_gap","sorted_line_gap","left-align","next_line_diff", "pattern", "line_number", "pattern_type"] 

app = FastAPI()

model_path = Path('section-identifier/section_chunker_model_v1.pkl')

model = joblib.load(model_path)

class PredictRequest(BaseModel):
    json_data: Dict
    merge: bool

class PredictResponse(BaseModel):
    sections: List

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Section Identification Service"}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest): # request: Request
    try:
        # raw_data = await request.body()
        # data = json.loads(raw_data)
        logger.info("Started extracting features")

        dataset = extract_features(patterns, discards, columns, request.json_data) #

        logger.info("Features extracted!!")

        predictions = model.predict(dataset[features]).tolist()

        logger.info("Made predictions!!")

        sections = get_sections(request.json_data, predictions)

        logger.info("Extracted sections!!")

        if request.merge:
            questions = get_questions(sections)
            logger.info("Created questions!!")

            new_predictions = get_predictions(questions)
            logger.info("Made new predictions!!")

            sections = merge_sections(new_predictions, sections)
            logger.info("Merged sections!!")

        return PredictResponse(sections=sections)
    except Exception as e:
        logger.info(e)
        raise HTTPException(status_code=400, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)