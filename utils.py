# -*- coding: utf-8 -*-
import hashlib
import os
import requests
import json
import configparser
import copy
import time
from tqdm import tqdm
import re
import emoji
import neologdn
import unicodedata
from typing import Dict, Iterable, List, Tuple
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder

from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.trainer import Trainer, GradientDescentTrainer

CODE_REGEX = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％\s]')

def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)


def build_data_loaders(config,
    train_data: List[Instance],
    dev_data: List[Instance],
    test_data: List[Instance]) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_loader = SimpleDataLoader(train_data, config.batch_size_for_train, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, config.batch_size_for_eval, shuffle=False)
    test_loader = SimpleDataLoader(test_data, config.batch_size_for_eval, shuffle=False)

    return train_loader, dev_loader, test_loader

def build_trainer(
    config,
    model: Model,
    train_loader: DataLoader,
    dev_loader: DataLoader,
) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters, lr=config.lr)  # type: ignore
    model.cuda()
    trainer = GradientDescentTrainer(
        model=model,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=config.num_epochs,
        optimizer=optimizer,
        cuda_device=0,
        serialization_dir=config.serialization_dir
    )
    return trainer

def emb_returner(config):
    if config.bert_name == 'japanese-bert':
        huggingface_model = 'cl-tohoku/bert-base-japanese'
    else:
        huggingface_model = 'dummy'
        print(config.bert_name,'are not supported')
        exit()
    bert_embedder = PretrainedTransformerEmbedder(model_name="cl-tohoku/bert-base-japanese")
    return bert_embedder, bert_embedder.get_output_dim(), BasicTextFieldEmbedder({'tokens': bert_embedder})


def normalizer(text):
    text = neologdn.normalize(text)
    text = ''.join(['' if c in emoji.UNICODE_EMOJI else c for c in text])
    text = CODE_REGEX.sub('', text)
    text = re.compile(u"([\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF])").sub('', text)
    text = unicodedata.normalize("NFKC", text)

    return text
