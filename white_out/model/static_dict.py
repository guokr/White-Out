#!/usr/bin/env python
# encoding: utf-8

import pickle

class StaticDictionary(object):
    def __init__(self, dict_path: str):
        static_dict_obj = pickle.load(open(dict_path, "rb"))
        self.alphabet = static_dict_obj["alphabet"]
        self.words_set = static_dict_obj["words_set"]
        self.words_trie = static_dict_obj["words_trie"]
