# -*- coding: utf-8 -*-
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn
from fastapi import BackgroundTasks, FastAPI
import random
from train import trainer

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sentiment_class_predictor = trainer()

class Sentiment(BaseModel):
    sentence: str

@app.get("/hello")
def hello():
   return {"Hello": "World!"}

@app.post("/sentiment/")
async def predict_nearest_title(sentiment: Sentiment):
    probs = sentiment_class_predictor.predict(sentence=sentiment.sentence)['probs']
    '''
    vocab._token_to_index["labels"]
    {'0': 0, '-1': 1, '1': 2}
    '''

    return {'probs':
                {
                    'neutral': probs[0],
                    'negative': probs[1],
                    'positive': probs[2]
                 }
            }

if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=8000,
                log_level="trace", debug=True)