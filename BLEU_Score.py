import collections
import math
import torch
from nltk.util import ngrams

def _compute_ngram_counter(tokens, max_n):
   ngrams_counter = collections.Counter()
   for n in range(1, max_n + 1):
       ngrams_counter.update(ngrams(tokens, n))
   return ngrams_counter
# ngrams_counter = collections.Counter(tuple(x.split(" ")) for x in ngrams_iterator(tokens, max_n))
# 였던 것을 nltk 사용하는 것으로 바꿈. ngrams_iterator가 torchtext 안에 있는데 torchtext 설치에 문제가 있어서..

def bleu_score(candidate_corpus, references_corpus, max_n=4, weights=[0.25] * 4):
   assert max_n == len(weights), 'Length of the "weights" list has be equal to max_n'
   assert len(candidate_corpus) == len(references_corpus), "The length of candidate and reference corpus should be the same"

   clipped_counts = torch.zeros(max_n)
   total_counts = torch.zeros(max_n)
   weights = torch.tensor(weights)

   candidate_len = 0.0
   refs_len = 0.0

   for (candidate, refs) in zip(candidate_corpus, references_corpus):
       current_candidate_len = len(candidate)
       candidate_len += current_candidate_len

       refs_len_list = [float(len(ref)) for ref in refs]
       refs_len += min(refs_len_list, key=lambda x: abs(current_candidate_len - x))

       reference_counters = _compute_ngram_counter(refs[0], max_n)
       for ref in refs[1:]:
           reference_counters = reference_counters | _compute_ngram_counter(ref, max_n)

       candidate_counter = _compute_ngram_counter(candidate, max_n)

       clipped_counter = candidate_counter & reference_counters

       for ngram, count in clipped_counter.items():
           clipped_counts[len(ngram) - 1] += count

       for i in range(max_n):
           total_counts[i] += max(current_candidate_len - i, 0)

   if min(clipped_counts) == 0:
       return 0.0
   else:
       pn = clipped_counts / total_counts
       log_pn = weights * torch.log(pn)
       score = torch.exp(sum(log_pn))

       bp = math.exp(min(1 - refs_len / candidate_len, 0))

       return bp * score.item()