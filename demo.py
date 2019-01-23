#!/usr/bin/env python
# encoding: utf-8

from whiteout.model import LevenshteinModel

words = ["hi", "there", "therea","our", "friend"]
mm = LevenshteinModel(words=words, max_distance=2)
kk = mm([["ho", "thera"], ["thore"], ["oui"]])

for _ in kk:
    print(_)
