import pickle
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
from tqdm import tqdm
import os

class SentimentClassPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        # This method is implemented in the base class.
        return self.predict_json({"context": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        context = json_dict["context"]
        return self._dataset_reader.text_to_instance(mention_uniq_id=None,
                                                     data={'context': context})