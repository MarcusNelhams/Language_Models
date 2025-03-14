'''
Language model classes
'''

import numpy as np

def parse(path):
    with open(path, 'r', encoding='utf8') as file:
            lines = file.readlines()

    tokens = []
    for line in lines:
        tokens_in_line = line.split()
        tokens.extend(['<START>'] + tokens_in_line + ['<STOP>'])

    return tokens

class LanguageModel(object):
    """Base class for feature extraction.
    """
    def __init__(self):
        pass
    def transform(self, tokens):
        pass
    def fit(self, tokens):
        pass
    def evaluate(self, test):
        pass

class Unigram(LanguageModel) :
    def __init__(self, alpha=0, dropVal = 3):
        self.unigram = {}
        self.probs = []
        self.word_counts = []
        self.total_words = 0
        self.alpha = alpha
        self.dropVal = dropVal

    def transform(self, tokens):
        word_counts = np.zeros(len(self.unigram))
        for tok in tokens:
            if tok in self.unigram:
                word_counts[self.unigram[tok]] += 1
            else:
                word_counts[self.unigram['<UNK>']] += 1
        
        # smoothing
        word_counts += self.alpha

        # handle words with c(w) < self.dropVal
        for index, count in enumerate(word_counts):
            if count < self.dropVal + self.alpha and index != self.unigram['<UNK>']:
                word_counts[self.unigram["<UNK>"]] += count - self.alpha
                word_counts[index] = 0
            
        return word_counts

    def fit(self, tokens):
        index = 0
        for tok in tokens:
            if tok not in self.unigram:
                self.unigram[tok] = index
                index += 1
            else:
                continue
        self.unigram["<UNK>"] = index

        self.word_counts = self.transform(tokens)
        
        self.total_words = self.word_counts.sum() - self.word_counts[self.unigram['<START>']]

        self.probs = self.word_counts / self.total_words
    
    def evaluate(self, test):
        total_prob = 0
        M = len([x for x in test if x != '<START>'])
        for tok in test:
            if tok in self.unigram and self.word_counts[self.unigram[tok]] > 0:
                #print(f"probs: {tok} {np.log2(self.probs[self.unigram[tok]])}")
                if (tok != "<START>"):
                    total_prob += np.log2(self.probs[self.unigram[tok]])
            elif tok in self.unigram and self.word_counts[self.unigram[tok]] == 0:
                total_prob += np.log2(self.probs[self.unigram['<UNK>']])
            elif self.alpha > 0:
                total_prob += np.log2(self.alpha / (self.word_counts.sum() - self.word_counts[self.unigram['<START>']]))

        l = total_prob / M
        return 2**-l
    
class Bigram(LanguageModel):
    def __init__(self, alpha=0, dropVal = 3):
        self.bigram = {}
        self.probs = []
        self.gram_counts = []
        self.total_words = 0
        self.alpha = alpha
        self.unigram = Unigram(alpha, dropVal)
        self.vocab_size = 0

    def transform(self, tokens):
        gram_counts = np.zeros(len(self.bigram))
        for tok in tokens:
            if tok == '<START>':
                init_prev = '<START>'
            init_curr = tok
            gram = (init_curr, init_prev)

            if gram in self.bigram:
                curr, prev = gram
                if self.unigram.word_counts[self.unigram.unigram[curr]] == 0:
                    curr = '<UNK>'
                if self.unigram.word_counts[self.unigram.unigram[prev]] == 0:
                    prev = '<UNK>'
                new_gram = (curr, prev)
                if new_gram not in self.bigram:
                    self.bigram[new_gram] = self.bigram[gram]
                gram_counts[self.bigram[new_gram]] += 1
            else:
                gram_counts[self.bigram[('<UNK>','<UNK>')]] += 1
            
            init_prev = init_curr

        # smoothing
        gram_counts += self.alpha

        return gram_counts

    def fit(self, tokens):
        self.unigram.fit(tokens)
        self.vocab_size = len([x for x in self.unigram.word_counts if x > 0])

        index = 0
        for tok in tokens:
            if tok == '<START>':
                prev = '<START>'
            curr = tok
            gram = (curr, prev)
            if gram not in self.bigram:
                self.bigram[gram] = index
                index += 1
            prev = curr
        self.bigram[('<UNK>', '<UNK>')] = index

        self.gram_counts = self.transform(tokens)

        self.probs = np.zeros(len(self.gram_counts))
        for gram in self.bigram.keys():
            w_curr, w_prev = gram
            if self.unigram.word_counts[self.unigram.unigram[w_curr]] == 0:
                w_curr = '<UNK>'
            if self.unigram.word_counts[self.unigram.unigram[w_prev]] == 0:
                w_prev = '<UNK>'
            self.probs[self.bigram[(w_curr, w_prev)]] = self.gram_counts[self.bigram[(w_curr, w_prev)]]
            c_prev = self.unigram.word_counts[self.unigram.unigram[w_prev]] + self.alpha * self.vocab_size
            # handle w_prev == <UNK>
            if c_prev > 0:
                self.probs[self.bigram[(w_curr, w_prev)]] /= c_prev
            else:
                self.probs[self.bigram[(w_curr, w_prev)]] = 0

    def evaluate(self, test):
        grams = []
        zero_prob_total = 0
        zero_prob_encounters = 0
        for tok in test:
            if tok == '<START>':
                init_prev = '<START>'
            init_curr = tok
            gram = (init_curr, init_prev)

            curr, prev = gram
            if curr not in self.unigram.unigram or self.unigram.word_counts[self.unigram.unigram[curr]] == 0:
                curr = '<UNK>'
            if prev not in self.unigram.unigram or self.unigram.word_counts[self.unigram.unigram[prev]] == 0:
                prev = '<UNK>'
            new_gram = (curr, prev)

            # handle prob(x|y) == 0

            if new_gram in self.bigram:
                grams.append(new_gram)
            elif self.alpha > 0:
                zero_prob_total += np.log2(self.alpha / (self.unigram.word_counts[self.unigram.unigram[prev]] + self.alpha * self.vocab_size))
                zero_prob_encounters += 1
            
            init_prev = init_curr

        total_prob = 0
        M = len(grams) + zero_prob_encounters - len([x for x in grams if x == ('<START>', '<START>')])
        for gram in grams:
            curr, prev = gram
            if gram in self.bigram:
                if (gram != ("<START>", "<START>") and self.probs[self.bigram[gram]] > 0):
                    #print(f"probs: {gram} {np.log2(self.probs[self.bigram[gram]])}")
                    total_prob += np.log2(self.probs[self.bigram[gram]])

        # add zero probs
        #print(zero_prob_total)
        #print(total_prob)
        total_prob += zero_prob_total


        l = total_prob / M
        return 2**-l

class Trigram(LanguageModel):
    def __init__(self, alpha=0, dropVal=3):
        self.trigram = {}
        self.probs = []
        self.gram_counts = []
        self.total_words = 0
        self.alpha = alpha
        self.unigram = Unigram(self.alpha, dropVal)
        self.bigram = Bigram(self.alpha, dropVal)
        self.vocab_size = 0

    def transform(self, tokens):
        gram_counts = np.zeros(len(self.trigram))
        for tok in tokens:
            if tok == '<START>':
                prev_init = '<START>'
                prev_prev_init = '<START>'
            curr_init = tok
            gram = (curr_init, prev_init, prev_prev_init)
            if gram in self.trigram:
                curr, prev, prev_prev = gram
                if self.unigram.word_counts[self.unigram.unigram[curr]] == 0:
                    curr = '<UNK>'
                if self.unigram.word_counts[self.unigram.unigram[prev]] == 0:
                    prev = '<UNK>'
                if self.unigram.word_counts[self.unigram.unigram[prev_prev]] == 0:
                    prev_prev = '<UNK>'
                new_gram = (curr, prev, prev_prev)
                if new_gram not in self.trigram:
                    self.trigram[new_gram] = self.trigram[gram]
                gram_counts[self.trigram[new_gram]] += 1
            else:
                gram_counts[self.trigram[('<UNK>','<UNK>','<UNK>')]] += 1
            
            prev_prev_init = prev_init
            prev_init = curr_init

        # smoothing
        gram_counts = gram_counts + self.alpha

        return gram_counts

    def fit(self, tokens):
        self.bigram.fit(tokens)
        self.unigram.fit(tokens)
        self.vocab_size = len([x for x in self.unigram.word_counts if x > 0])

        index = 0
        for tok in tokens:
            if tok == '<START>':
                prev = '<START>'
                prev_prev = '<START>'
            curr = tok
            gram = (curr, prev, prev_prev)
            if gram not in self.trigram:
                self.trigram[gram] = index
                index += 1
            prev_prev = prev
            prev = curr

        self.trigram[('<UNK>', '<UNK>', '<UNK>')] = index

        self.gram_counts = self.transform(tokens)

        self.probs = np.zeros(len(self.gram_counts))
        for gram in self.trigram.keys():
            w_curr, w_prev, w_prev_prev = gram
            if self.unigram.word_counts[self.unigram.unigram[w_curr]] == 0:
                w_curr = '<UNK>'
            if self.unigram.word_counts[self.unigram.unigram[w_prev]] == 0:
                w_prev = '<UNK>'
            if self.unigram.word_counts[self.unigram.unigram[w_prev_prev]] == 0:
                w_prev_prev = '<UNK>'
            gram = (w_curr, w_prev, w_prev_prev)
            self.probs[self.trigram[(w_curr, w_prev, w_prev_prev)]] = self.gram_counts[self.trigram[(w_curr, w_prev, w_prev_prev)]]
            c_prev = self.bigram.gram_counts[self.bigram.bigram[(w_prev, w_prev_prev)]] + (self.alpha * self.vocab_size)
            if c_prev == 0:
                self.probs[self.trigram[(w_curr, w_prev, w_prev_prev)]] = 0
            else:
                self.probs[self.trigram[(w_curr, w_prev, w_prev_prev)]] /= c_prev

    def evaluate(self, test):
        grams = []
        zero_probs_encountered = 0
        zero_probs_total = 0
        for tok in test:
            if tok == '<START>':
                prev_init = '<START>'
                prev_prev_init = '<START>'
            curr_init = tok
            gram = (curr_init, prev_init, prev_prev_init)

            curr, prev, prev_prev = gram
            if curr not in self.unigram.unigram or self.unigram.word_counts[self.unigram.unigram[curr]] == 0:
                curr = '<UNK>'
            if prev not in self.unigram.unigram or self.unigram.word_counts[self.unigram.unigram[prev]] == 0:
                prev = '<UNK>'
            if prev_prev not in self.unigram.unigram or self.unigram.word_counts[self.unigram.unigram[prev_prev]] == 0:
                prev_prev = '<UNK>'
            new_gram = (curr, prev, prev_prev)

            if new_gram in self.trigram:
                grams.append(new_gram)
            elif self.alpha > 0 and curr in self.unigram.unigram and prev in self.unigram.unigram and prev_prev in self.unigram.unigram:
                if (prev, prev_prev) in self.bigram.bigram and self.bigram.probs[self.bigram.bigram[(prev, prev_prev)]] > 0:
                    zero_probs_total += np.log2(self.alpha / (self.bigram.gram_counts[self.bigram.bigram[(prev, prev_prev)]] + self.alpha * self.vocab_size))
                else:
                    zero_probs_total += np.log2(self.alpha / self.alpha * self.vocab_size)
                zero_probs_encountered += 1

            prev_prev_init = prev_init
            prev_init = curr_init

        total_prob = 0
        M = len(grams) + zero_probs_encountered - len([x for x in grams if x == ('<START>', '<START>', '<START>')])
        for gram in grams:
            if gram in self.trigram:
                #print(gram)
                if (gram[0] != '<START>' and gram[1] == '<START>' and gram[2] == '<START>'):
                    gram = (gram[0], gram[1])
                    #print(f"probs: {gram} {np.log2(self.bigram.probs[self.bigram.bigram[gram]])}")
                    total_prob += np.log2(self.bigram.probs[self.bigram.bigram[gram]])
                elif (gram != ("<START>", "<START>", "<START>")):
                    #print(f"probs: {gram} {np.log2(self.probs[self.trigram[gram]])}")
                    if self.probs[self.trigram[gram]] > 0:
                        total_prob += np.log2(self.probs[self.trigram[gram]])

        total_prob += zero_probs_total

        l = total_prob / M
        return 2**-l

class Interpolated(LanguageModel):
    def __init__(self, lambs=[.1, .3, .6], dropVal=3):
        self.unigram = Unigram(0, dropVal)
        self.bigram = Bigram(0, dropVal)
        self.trigram = Trigram(0, dropVal)
        self.probs = []
        self.lambdas = lambs
    
    def transform(self, tokens):
        uni_probs = self.lambdas[0] * self.unigram.probs
        bi_probs = self.lambdas[1] * self.bigram.probs
        tri_probs = self.lambdas[2] * self.trigram.probs

        probs = np.zeros(len(self.trigram.trigram))
        for w1, w2, w3 in self.trigram.trigram:
            tri_prob = tri_probs[self.trigram.trigram[(w1, w2, w3)]]
            bi_prob = bi_probs[self.bigram.bigram[(w1, w2)]]
            uni_prob = uni_probs[self.unigram.unigram[w1]]

            probs[self.trigram.trigram[(w1, w2, w3)]] = uni_prob + bi_prob + tri_prob

        return probs
    
    def fit(self, tokens):
        self.trigram.fit(tokens)
        self.bigram = self.trigram.bigram
        self.unigram = self.trigram.unigram

        self.probs = self.transform(tokens)

    def evaluate(self, test):
        grams = []
        for tok in test:
            if tok == '<START>':
                prev_init = '<START>'
                prev_prev_init = '<START>'
            curr_init = tok
            gram = (curr_init, prev_init, prev_prev_init)

            curr, prev, prev_prev = gram
            if curr not in self.unigram.unigram or self.unigram.word_counts[self.unigram.unigram[curr]] == 0:
                curr = '<UNK>'
            if prev not in self.unigram.unigram or self.unigram.word_counts[self.unigram.unigram[prev]] == 0:
                prev = '<UNK>'
            if prev_prev not in self.unigram.unigram or self.unigram.word_counts[self.unigram.unigram[prev_prev]] == 0:
                prev_prev = '<UNK>'
            new_gram = (curr, prev, prev_prev)

            grams.append(new_gram)
            
            prev_prev_init = prev_init
            prev_init = curr_init

        total_prob = 0
        M = len(grams) - len([x for x in grams if x == ('<START>', '<START>', '<START>')])
        for gram in grams:
            if gram in self.trigram.trigram:
                #print(gram)
                if (gram[0] != '<START>' and gram[1] == '<START>' and gram[2] == '<START>'):
                    gram = (gram[0], gram[1])
                    #print(f"probs: {gram} {np.log2(self.bigram.probs[self.bigram.bigram[gram]])}")
                    total_prob += np.log2((self.lambdas[1] + self.lambdas[2]) * self.bigram.probs[self.bigram.bigram[gram]] + self.lambdas[0] * self.unigram.probs[self.unigram.unigram[gram[0]]])
                elif (gram != ("<START>", "<START>", "<START>") and self.probs[self.trigram.trigram[gram]] > 0):
                    #print(f"probs: {gram} {np.log2(self.probs[self.trigram[gram]])}")
                    total_prob += np.log2(self.probs[self.trigram.trigram[gram]])
                    #print(self.probs[self.trigram.trigram[gram]])

        l = total_prob / M
        return 2**-l     
