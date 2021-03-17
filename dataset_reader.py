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
import glob
import re
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
    def _read(self, train_dev_test_flag: str) -> list:
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
            instances.append(self.text_to_instance(mention_uniq_id,
                                                   data=self.mention_id2data[mention_uniq_id]))

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
            fields = {"context": context_field}

            fields['label'] = LabelField(data['label'])
            fields['mention_uniq_id'] = ArrayField(np.array(mention_uniq_id))

        return Instance(fields)


    def mention_ids_returner(self):
        mention_id2data = {}
        train_mention_ids, dev_mention_ids, test_mention_ids = [], [], []

        dataset = []
        with open(self.config.dataset_path, 'r') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if line.strip() != '':
                    label = line.split(',')[0]
                    if label not in ['0', '1']:
                        print('ErrorLabel:', label)
                        continue
                    context = ''.join(line.split(',')[1:])
                    data = {'label': label, 'context': context}
                    dataset.append(data)

        # train : dev : test = 7 : 1 : 2
        data_num = len(dataset)
        if self.config.debug:
            data_num = data_num // 8
        data_frac = data_num // 10
        train_tmp_ids = [i for i in range(0, data_frac * 7)]
        dev_tmp_ids = [j for j in range(data_frac * 7, data_frac * 8)]
        test_tmp_ids = [k for k in range(data_frac * 8, data_num)]

        for idx, data in enumerate(dataset):
            mention_id2data.update({idx: data})

            if idx in train_tmp_ids:
                train_mention_ids.append(idx)
            elif idx in dev_tmp_ids:
                dev_mention_ids.append(idx)
            elif idx in test_tmp_ids:
                test_mention_ids.append(idx)
            else:
                if self.config.debug:
                    continue
                else:
                    print('Error')
                    exit()

        return train_mention_ids, dev_mention_ids, test_mention_ids, mention_id2data

if __name__ == '__main__':
    params = Params()
    config = params.opts
    dsr = TwitterDatasetReader(config=config)
    dsr._read('train')