#!/usr/bin/env python
# encoding: utf-8

from white_out.model import LevenshteinModel
from white_out.model import BrillMooreModel

words = ["hi", "there", "therea","our", "friend"]

lm_model = LevenshteinModel(words=words, max_distance=2)
lm_res = lm_model([["ho", "thera"], ["thore"], ["oui"]])
for _ in lm_res:
    print(_)

bm_model = BrillMooreModel(dictionary_path="/home/guokr/DeepPav/deeppav/dictionary/new_static_dict.pkl",
                           model_path="./sample_checkpoint/error_model.tsv", candidates_count=1)


origin_sent = "so do you kno where is the opple ?"
bm_res = bm_model([origin_sent.split()])
correct_sent = [_[0][1] for _ in bm_res[0]]

print(correct_sent)



