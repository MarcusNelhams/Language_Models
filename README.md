# CSE 143 (Winter 2020) HW 1 data

Language Models and Perplexity Classifiers

# Usage

This folder contains 3 files, a subset of the 1 Billion Word Benchmark's
heldout set.

Specifically, `1b_benchmark.train.tokens` is taken from sections 0-9,
`1b_benchmark.dev.tokens` is taken from sections 10 and 11, and
`1b_benchmark.test.tokens` is taken from sections 12 and 13.

## To Run

python3 .\main --model {unigram, bigram, trigram} --data {train, dev, test} --alpha {0.1, 0.5, 0.1, optimize}
