import perc
import sys
import optparse
import os
from collections import defaultdict


def distance(var1, var2):
    if len(var1) != len(var2):
        return 1
    elif var1 == var2:
        return 0
    elif var1[1:] == var2[1:]:
        return 0.9
    else:
        return 1


def perc_train(train_data, tagset, numepochs):
    # perceptron train
    T = float(len(train_data))
    step = numepochs*T
    feat_vec_cache = defaultdict(int)
    # feat_vec stores the weights for the features
    # of a sentence, initially all weights are 0
    feat_vec = defaultdict(int)
    # default_tag = 'B-NP'
    default_tag = tagset[0]
    # for each epoch/iteration
    for i in range(0, numepochs):
        # pdb.set_trace()
        # for each item (e.g tuple=([labeled words for each sentence]
        # ,[features for those words of sentence])) in train_data
        for (label_list, feat_list) in train_data:
            # cur = list of best tag for each word in
            # sentence found using viterbi algo
            cur = perc.perc_test(feat_vec, label_list,
                                 feat_list, tagset, default_tag)
            # gold = list of reference/true tag for each word in sentence
            gold = [entry.split()[2] for entry in label_list]
            if cur != gold:
                cur.insert(0, 'B_-1')
                gold.insert(0, 'B_-1')
                cur.append('B_+1')
                gold.append('B_+1')
                cur_len = len(cur)
                gold_len = len(gold)
                if cur_len != gold_len:
                    raise ValueError("output length is not the same \
                                      with the input sentence")
                feat_index = 0
                # perceptron update
                # for each tag/word of a sentence
                for j in range(1, cur_len):
                    # for each word in a sentence, (feat_index, features)
                    # is a tuple, where feat_index=endindex of the list
                    # of features for that word, and features=list of
                    # features for that word
                    (feat_index, features) = perc.feats_for_word(feat_index,
                                                                 feat_list)
                    # update the weights of the features for that word,
                    # by rewarding the features seen in reference, while
                    # penalizing the ones not seen in reference but
                    # returned by viterbi
                    for f in features:
                        feat_vec[(f, cur[j])] = feat_vec[(f, cur[j])] - distance(cur[j], gold[j])
                        feat_vec[(f, gold[j])] = feat_vec[(f, gold[j])] + distance(cur[j], gold[j])
                        feat_vec_cache[(f, cur[j])] = feat_vec_cache[
                                    (f, cur[j])] - distance(cur[j], gold[j])*(float(step/numepochs*T))
                        feat_vec_cache[(f, gold[j])] = feat_vec_cache[
                                    (f, gold[j])] + distance(cur[j], gold[j])*(float(step/numepochs*T))
            step -= 1
        print >> sys.stderr, "iteration %d done." % i
    return feat_vec_cache


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs",
                         default=int(1), help="number of epochs of training; \
                         in each epoch we iterate over over training examples")
    optparser.add_option("-m", "--modelfile", dest="modelfile",
                         default=os.path.join("data", "default.model"),
                         help="weights for all features stored on disk")
    (opts, _) = optparser.parse_args()

    # each element in the feat_vec dictionary is:
    # key=feature_id value=weight
    feat_vec = {}
    tagset = []
    train_data = []

    # tagset contains list of the tags in tagset.txt
    # ['B-NP', 'I-NP', 'O', 'B-VP', 'B-PP', 'I-VP', 'B-ADVP',
    # 'B-SBAR', 'B-ADJP', 'I-ADJP', 'B-PRT', 'I-ADVP', 'I-PP',
    # 'I-CONJP', 'I-SBAR', 'B-CONJP', 'B-INTJ', 'B-LST', 'I-INTJ',
    # 'I-UCP', 'I-PRT', 'I-LST', 'B-UCP']
    tagset = perc.read_tagset(opts.tagsetfile)
    # print 'tagset', tagset
    print >>sys.stderr, "reading data ..."
    # print 'train_data', type(train_data), len(train_data)
    train_data = perc.read_labeled_data(opts.trainfile, opts.featfile)
    # for i in train_data:
    #     print type(i), len(i), i
    print >>sys.stderr, "done."
    feat_vec = perc_train(train_data, tagset, int(opts.numepochs))
    perc.perc_write_to_file(feat_vec, opts.modelfile)
