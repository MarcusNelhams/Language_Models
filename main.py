'''
Main file for running
'''

import numpy as np
import time
import argparse
from models import *
import random
import matplotlib.pyplot as plt

def findHyperParams(args):
    train_file = '1b_benchmark.train.tokens'
    tokens = parse(train_file)

    test_file = '1b_benchmark.dev.tokens'
    #test_file = 'test.tokens'
    test = parse(test_file)
    start_time = time.time()
    perps = []
    alphas = []
    drops = []
    bestPerp = None
    bestAlpha = None
    bestDrop = None
    for i in range(15):
        alpha = .0005 + (random.random() * (.002 - .0005))
        drop = 3
        #drop = 3
        if args.model == "unigram":
            feat_extractor = Unigram(alpha, drop)
        elif args.model == "bigram":
            feat_extractor = Bigram(alpha, drop)
        elif args.model == "trigram":
            feat_extractor = Trigram(alpha, drop)
        elif args.model == "interpolated":
            feat_extractor = Interpolated(drop)
        else:
            raise Exception("Pass unigram, bigram or trigram to --model")

        feat_extractor.fit(tokens)

        perplexity = feat_extractor.evaluate(test)

        perps.append(perplexity)
        alphas.append(alpha)
        drops.append(drop)
        
        print(f"{i+1}-- {alpha:.5f}, {drop}: {perplexity:.5f}")

        if bestPerp is None or perplexity < bestPerp:
            bestPerp = perplexity
            bestAlpha = alpha
            bestDrop = drop

    print()
    print("Time for training and evaluate: %.2f seconds" % (time.time() - start_time))
    print(f"Best alpha, drop, and perplexity: {bestAlpha}, {bestDrop}: {bestPerp}")

    # Create the first subplot
    plt.subplot(2, 1, 1)  # (rows, columns, panel number)
    plt.scatter(drops, perps, label='drops v perp', color='blue')
    plt.xlabel('drops')
    plt.ylabel('perps')
    plt.title('drops v Perps')
    plt.legend()

    # Create the second subplot
    plt.subplot(2, 1, 2)  # (rows, columns, panel number)
    plt.scatter(alphas, perps, label='Alpha v Perplexities', color='red')
    plt.xlabel('Alpha')
    plt.ylabel('Perplexity')
    plt.title('Perplexities of Unigram depenendinig on Alpha Value of Laplace Smoothing')
    plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plots
    plt.show()

def findLambdas():
    train_file = '1b_benchmark.train.tokens'
    tokens = parse(train_file)

    test_file = '1b_benchmark.dev.tokens'
    #test_file = 'test.tokens'
    test = parse(test_file)
    start_time = time.time()
    perps = []
    l1s = []
    l2s = []
    l3s = []
    bestPerp = None
    bestLams = None
    for i in range(25):
        l1 = random.random()
        l2 = random.random() * (1 - l1)
        l3 = 1 - l1 - l2
        
        model = Interpolated([l1, l2, l3])

        model.fit(tokens)

        perplexity = model.evaluate(test)

        perps.append(perplexity)
        l1s.append(l1)
        l2s.append(l2)
        l3s.append(l3)
        
        print(f"{i+1}-- lambdas({l1:.5f}, {l2:.5}, {l3:.5}): {perplexity:.5f}")

        if bestPerp is None or perplexity < bestPerp:
            bestPerp = perplexity
            bestLams = (l1, l2, l3)

    print("Time for training and evaluate: %.2f seconds" % (time.time() - start_time))
    print(f"Best lambdas and perplexity: {bestLams}: {bestPerp}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'trigram', 'interpolated'])
    parser.add_argument('--alpha', '-a', type=str, default='0',
                        choices=['0', '1', '0.5', '0.1', '.00001', 'optimize'])
    parser.add_argument('--data', '-d', type=str, default='train',
                        choices=['train', 'dev', 'test'])
    args = parser.parse_args()
    print(args)

    # Convert text into features
    if args.alpha == 'optimize':
        if args.model == 'interpolated':
            findLambdas()
        else:
            findHyperParams(args)
        return

    alpha = float(args.alpha)
    if args.model == "unigram":
        feat_extractor = Unigram(alpha)
    elif args.model == "bigram":
        feat_extractor = Bigram(alpha)
    elif args.model == "trigram":
        feat_extractor = Trigram(alpha)
    elif args.model == "interpolated":
        feat_extractor = Interpolated([.1,.03,.87], 0)
    else:
        raise Exception("Pass unigram, bigram or trigram to --model")

    # Tokenize text into tokens
    start_time = time.time()

    train_file = '1b_benchmark.train.tokens'
    #train_file = 'halftrain.tokens'
    tokens = parse(train_file)
    feat_extractor.fit(tokens)

    test_file = '1b_benchmark.' + args.data + '.tokens'
    #test_file = 'test.tokens'
    test = parse(test_file)

    perplexity = feat_extractor.evaluate(test)
    print(f"perplexity of test: {perplexity}")
    print("Time for training and evaluate: %.2f seconds" % (time.time() - start_time))


if __name__ == '__main__':
    main()