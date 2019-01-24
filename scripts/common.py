#!/usr/bin/env python
# encoding: utf-8

from collections import defaultdict
import fire
import pickle
import os

def build_StaticDictionary(raw_dictionary_filepath: str):
    dir_path, filename = os.path.split(raw_dictionary_filepath)

    with open(raw_dictionary_filepath, 'r', newline='', encoding='utf8') as f:
        words = [line.strip() for line in f]

    words = {word.strip().lower() for word in words}

    alphabet = {c for w in words for c in w} # set

    words_trie = defaultdict(set)
    for word in words:
        for i in range(len(word)):
            words_trie[word[:i]].add(word[:i+1])
            words_trie[word] = set()
    words_trie = {k: sorted(v) for k, v in words_trie.items()}

    static_dict_obj = {"alphabet": alphabet,
                       "words_set": words,
                       "words_trie": words_trie}

    static_dict_obj_path = os.path.join(dir_path, "static_dict.pkl")
    pickle.dump(static_dict_obj, open(static_dict_obj_path, "wb"))

if __name__ == "__main__":
    fire.Fire({"build_dict": build_StaticDictionary})
