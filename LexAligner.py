#!/usr/bin/env python
from __future__ import division
import optparse
import sys
import os
import logging
from collections import defaultdict


def Lexical_Aligner(bitext, numepochs):

    f_count = defaultdict(int)
    e_count = defaultdict(int)
    fe_count = defaultdict(int)

    for (n, (f, e)) in enumerate(bitext):
        for f_i in set(f):
            f_count[f_i] += 1
            for e_j in set(e):
                fe_count[(f_i, e_j)] = 0
        for e_j in set(e):
            e_count[e_j] = 0
        if n % 500 == 0:
            sys.stderr.write(".")
    # ==========================================
    #                   Training
    # ==========================================
    t = fe_count
    t = dict.fromkeys(t, float(1/len(f_count)))
    for i in range(numepochs):
        e_count = dict.fromkeys(e_count, 0)
        fe_count = dict.fromkeys(fe_count, 0)
        for (f, e) in bitext:
            for f_i in f:
                Z = 0
                for (j, e_j) in enumerate(e):
                    Z += t[(f_i, e_j)]
                for (j, e_j) in enumerate(e):
                    c = float(t[(f_i, e_j)]/Z)
                    fe_count[(f_i, e_j)] += c
                    e_count[(e_j)] += c
        for (f_i, e_j) in fe_count:
            t[(f_i, e_j)] = float(fe_count[(f_i, e_j)]/e_count[(e_j)])
    # ==========================================
    #                   Decoding
    # ==========================================
    for (f, e) in bitext:
        for (i, f_i) in enumerate(f):
            bestp = 0
            bestj = 0
            for (j, e_j) in enumerate(e):
                if t[(f_i, e_j)] > bestp:
                    bestp = t[(f_i, e_j)]
                    bestj = j
            sys.stdout.write("%i-%i " % (i, bestj))
        sys.stdout.write("\n")


if __name__ == '__main__':

    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
    optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
    optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English filename (default=en)")
    optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
    optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
    optparser.add_option("-i", "--numepochs", dest="numepochs", default=int(1), help="number of epochs of training; in each epoch we iterate over over training examples")

    (opts, _) = optparser.parse_args()
    f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
    e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

    if opts.logfile:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

    sys.stderr.write("Training with Dice's coefficient...")

    bitext = [[sentence.strip().split() for sentence in pair] for pair in
              zip(open(f_data), open(e_data))[:opts.num_sents]]

    Lexical_Aligner(bitext, int(opts.numepochs))
