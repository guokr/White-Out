#!/usr/bin/env python
# encoding: utf-8

import string
from math import log10
from typing import Iterable, List, Tuple

from .levenshtein_searcher import LevenshteinSearcher

class LevenshteinModel():
    _punctuation = frozenset(string.punctuation)

    def __init__(self, words: Iterable[str], max_distance: int=1, error_probability: float=1e-4, *args, **kwargs):
        words = list({word.strip().lower().replace('ё', 'е') for word in words})
        alphabet = sorted({letter for word in words for letter in word})
        self.max_distance = max_distance
        self.error_probability = log10(error_probability)
        self.vocab_penalty = self.error_probability * 2
        self.searcher = LevenshteinSearcher(alphabet, words, allow_spaces=True, euristics=2)

    def _infer_instance(self, tokens: Iterable[str]) -> List[List[Tuple[float, str]]]:
        candidates = []
        for word in tokens:
            if word in self._punctuation:
                candidates.append([(0, word)])
            else:
                c = {candidate: self.error_probability * distance
                     for candidate, distance in self.searcher.search(word, d=self.max_distance)}
                c[word] = c.get(word, self.vocab_penalty)
                candidates.append([(score, candidate) for candidate, score in c.items()])
        return candidates

    def __call__(self, batch: Iterable[Iterable[str]], *args, **kwargs) -> List[List[List[Tuple[float, str]]]]:
        """Propose candidates for tokens in sentences

        Args:
            batch: batch of tokenized sentences

        Returns:
            batch of lists of probabilities and candidates for every token
        """
        return [self._infer_instance(tokens) for tokens in batch]
