from overrides import overrides
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import SpanField, ListField, TextField, MetadataField, ArrayField, SequenceLabelField, LabelField
from allennlp.data.fields import LabelField, TextField
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from parameters import Params
import random
from tqdm import tqdm
from tokenizer import CustomTokenizer
import numpy as np
import pickle
import json
from utils import normalizer
random.seed(42)

class TwitterDatasetReader(DatasetReader):
    def __init__(
        self,
        config,
        max_tokens: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.custom_tokenizer_class = CustomTokenizer(config=config)
        self.token_indexers = self.custom_tokenizer_class.token_indexer_returner()
        self.max_tokens = max_tokens
        self.config = config
        self.class2id = {}
        self.train_mention_ids, self.dev_mention_ids, self.test_mention_ids, self.mention_id2data = \
            self.mention_ids_returner()

    @overrides
    def _read(self, train_dev_test_flag: str):
        '''
        :param train_dev_test_flag: 'train', 'dev', 'test'
        :return: list of instances
        '''
        mention_ids, instances = list(), list()
        if train_dev_test_flag == 'train':
            mention_ids += self.train_mention_ids
            # Because Iterator(shuffle=True) has bug, we forcefully shuffle train dataset here.
            random.shuffle(mention_ids)
        elif train_dev_test_flag == 'dev':
            mention_ids += self.dev_mention_ids
        elif train_dev_test_flag == 'test':
            mention_ids += self.test_mention_ids

        for idx, mention_uniq_id in tqdm(enumerate(mention_ids)):
            instances.append(self.text_to_instance(mention_uniq_id, data=self.mention_id2data[mention_uniq_id]))

        return instances

    @overrides
    def text_to_instance(self, mention_uniq_id, data=None) -> Instance:
        if mention_uniq_id == None:
            tokenized = [Token('[CLS]')]
            tokenized += [Token(split_token) for split_token in self.custom_tokenizer_class.tokenize(
                                              txt=
                                              normalizer(data['context']))][:self.config.max_token_length]
            tokenized += [Token('[SEP]')]
            context_field = TextField(tokenized, self.token_indexers)
            fields = {"context": context_field}
        else:
            tokenized = [Token('[CLS]')]
            tokenized += [Token(split_token) for split_token in self.custom_tokenizer_class.tokenize(
                                              txt=normalizer(data['context']))][:self.config.max_token_length]
            tokenized += [Token('[SEP]')]
            context_field = TextField(tokenized, self.token_indexers)
            fields = {"context": context_field }
            fields['label'] = LabelField(data['label'])
            fields['mention_uniq_id'] = ArrayField(np.array(mention_uniq_id))

        return Instance(fields)


    def mention_ids_returner(self):
        mention_id2data = {}
        train_mention_ids, dev_mention_ids, test_mention_ids = [], [], []

        pos_data, neutral_data, neg_data = [], [], []
        with open(self.config.dataset_path, 'rb') as f:
            raw_data = pickle.load(f)

        pos, neutral, neg = 0, 0, 0
        for line_idx, line in tqdm(enumerate(raw_data)):
            context = line['text']
            if context.strip() == "":
                continue
            label = line['label']
            orig_id = line['id']
            topic = line['topic']
            status = line['status']
            if label == [0, 1, 0, 0, 0]: # pos
                true_label = '1'
                data = {'label': true_label, 'context': context, 'id': orig_id,  'topic': topic, 'status': status}
                pos_data.append(data)
                pos += 1
            elif label == [0, 0, 0, 1, 0]: # neutral
                true_label = '0'
                neutral += 1
                data = {'label': true_label, 'context': context, 'id': orig_id,  'topic': topic, 'status': status}
                if neutral > self.config.max_neutral_data_num:
                    continue
                neutral_data.append(data)
            elif label == [0, 0, 1, 0, 0]: # neg
                true_label = '-1'
                neg += 1
                data = {'label': true_label, 'context': context, 'id': orig_id,  'topic': topic, 'status': status}
                neg_data.append(data)
            else:
                continue

        random.shuffle(pos_data)
        random.shuffle(neutral_data)
        random.shuffle(neg_data)

        for one_class_dataset in [pos_data, neutral_data, neg_data]:
        # train : dev : test = 8 : 1 : 1
            data_num = len(one_class_dataset)
            if self.config.debug:
                data_num = data_num // 8
            data_frac = data_num // 10
            train_tmp_ids = [i for i in range(0, data_frac * 8)]
            dev_tmp_ids = [j for j in range(data_frac * 8, data_frac * 9)]
            test_tmp_ids = [k for k in range(data_frac * 9, data_num)]

            for idx, data in enumerate(one_class_dataset):
                tmp_idx_for_all_data = len(mention_id2data)
                mention_id2data.update({tmp_idx_for_all_data: data})

                if idx in train_tmp_ids:
                    train_mention_ids.append(tmp_idx_for_all_data)
                elif idx in dev_tmp_ids:
                    dev_mention_ids.append(tmp_idx_for_all_data)
                elif idx in test_tmp_ids:
                    test_mention_ids.append(tmp_idx_for_all_data)
                else:
                    if self.config.debug:
                        continue
                    else:
                        exit()
        with open('./dataset/train_mention_ids.pkl', 'wb') as trn_:
            pickle.dump(train_mention_ids, trn_)
        with open('./dataset/dev_mention_ids.pkl', 'wb') as dev_:
            pickle.dump(dev_mention_ids, dev_)
        with open('./dataset/test_mention_ids.pkl', 'wb') as test_:
            pickle.dump(test_mention_ids, test_)
        with open('./dataset/mention_id2data.json', 'w') as m2d:
            json.dump(mention_id2data, m2d, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))

        return train_mention_ids, dev_mention_ids, test_mention_ids, mention_id2data

if __name__ == '__main__':
    params = Params()
    config = params.opts
    dsr = TwitterDatasetReader(config=config)
    dsr._read('train')