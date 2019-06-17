#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-17 14:21
# @Author  : zhangzhen
# @Site    : 
# @File    : 2019_ccks_erl.py
# @Software: PyCharm
import numpy as np
from tqdm import tqdm
import codecs
import json
import os

root = '/Users/zhangzhen/data/ccks2019_el.v1'


def read_train_data(mode='test', train_num=1000):
    train_data = []
    with codecs.open(root + os.sep + 'train.json', encoding='utf-8') as f:
        count = 0
        for l in tqdm(f):
            try:
                _ = json.loads(l, encoding='utf-8')
                train_data.append({
                    'text': _['text'].lower(),
                    'mention_data': [(x['mention'].lower(), int(x['offset']), x['kb_id'])
                                     for x in _['mention_data'] if x['kb_id'] != 'NIL'
                                     ]
                })
                count += 1
            except ValueError:
                print("乱码字符串: ", l)

            if mode == 'test' and count > train_num:
                break
    return train_data


def read_kb(mode='train', kb_num=1000):
    id2kb, kb2ids = {}, {}
    with codecs.open(root + os.sep + 'kb_data', encoding='utf-8') as f:
        count = 0
        for l in tqdm(f):
            _ = json.loads(l)
            subject_id = _['subject_id']
            subject_alias = list(set([_['subject']] + _.get('alias', [])))
            subject_alias = [alias.lower() for alias in subject_alias]
            subject_desc = '\n'.join(u'%s：%s' % (i['predicate'], i['object']) for i in _['data'])
            subject_desc = subject_desc.lower()
            if subject_desc:
                id2kb[subject_id] = {'subject_alias': subject_alias, 'subject_desc': subject_desc}

            count += 1
            if mode == 'test' and count > kb_num:
                break

    # entity -> subject_id:List
    for i, j in id2kb.items():
        for k in j['subject_alias']:
            if k not in kb2ids:
                kb2ids[k] = []
            kb2ids[k].append(i)
    return id2kb, kb2ids


mode = 0
min_count = 2
char_size = 128


class CCKS2019:

    def __init__(self, mode='test'):
        self.id2kb, self.kb2ids = read_kb(mode=mode)
        self.train_data = read_train_data(mode=mode)
        self.id2char, self.char2id = self.load_char2int()
        self.train, self.dev = self.random_split_data(seed=1111)

    def load_char2int(self, filename='all_chars_me.json'):
        if not os.path.exists(filename):
            chars = {}
            for d in tqdm(iter(self.id2kb.values())):
                for c in d['subject_desc']:
                    chars[c] = chars.get(c, 0) + 1
            for d in tqdm(iter(self.train_data)):
                for c in d['text']:
                    chars[c] = chars.get(c, 0) + 1
            chars = {i: j for i, j in chars.items() if j >= min_count}
            id2char = {i + 2: j for i, j in enumerate(chars)}  # 0: mask, 1: padding
            char2id = {j: i for i, j in id2char.items()}
            json.dump([id2char, char2id], codecs.open(filename, 'w', encoding='utf-8'))
        else:
            id2char, char2id = json.load(codecs.open(filename, encoding='utf-8'))

        return id2char, char2id

    def random_split_data(self, filename='random_order.json', seed=1234):
        if not os.path.exists(filename):
            np.random.seed(seed=seed)

            random_order = list(range(len(self.train_data)))
            np.random.shuffle(random_order)
            json.dump(random_order, codecs.open(filename, 'w', encoding='utf-8'), indent=4)
        else:
            random_order = json.load(codecs.open(filename, encoding='utf-8'))

        dev = [self.train_data[j] for i, j in enumerate(random_order) if i % 9 == mode]
        train = [self.train_data[j] for i, j in enumerate(random_order) if i % 9 != mode]
        return train, dev


if __name__ == '__main__':
    ccks = CCKS2019()

    print()
    print(ccks.train_data[0])
    print(ccks.train_data[1])
