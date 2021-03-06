from __future__ import division
import sys
import codecs
import optparse
import os
import math
import Queue as Q

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w',
                     default=os.path.join('data', 'count_1w.txt'),
                     help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w',
                     default=os.path.join('data', 'count_2w.txt'),
                     help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input",
                     default=os.path.join('data', 'input'),
                     help="input file to segment")
(opts, _) = optparser.parse_args()


class Entry(object):
    def __init__(self, word, start, p, e):
        self.word = word
        self.start = start
        self.logp = p
        self.pre = e
        return

    def __cmp__(self, other):
        return cmp(self.start, other.start)


class Pdist(dict):
    "A probability distribution estimated from counts in datafile."

    def __init__(self, filename, sep='\t', N=None, missingfn=None):
        self.maxlen = 0
        for line in file(filename):
            (key, freq) = line.split(sep)
            try:
                utf8key = unicode(key, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key] = self.get(utf8key, 0) + int(freq)
            self.maxlen = max(len(utf8key), self.maxlen)
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1./N)

    def __call__(self, key):
        if key in self:
            return float(self[key])/float(self.N)
        # else: return self.missingfn(key, self.N)
        elif len(key) == 1:
            return self.missingfn(key, self.N)
        else:
            return None


def findlogp1(word):
    num = Pw(word)
    if num is None:
        return 0
    else:
        return math.log(num)


def findlogp2(word1, word2):
    lamb = 0.95
    logp = Pw(word1)
    logp2 = Pw2(word2+'\t'+word1)
    if logp2 is None:
        logp2 = 0
    if logp is None:
        logp = 0
    return math.log(lamb*logp2 + (1-lamb)*logp)


def traceSeq(chart, sent):
    output = []
    entry = chart[len(sent)-1]
    while entry is not None:
        output.insert(0, entry.word)
        entry = entry.pre
    return output


def segment(sent, Pw):
    # counter = 0
    heap = Q.PriorityQueue()
    chart = [None]*len(sent)
    for i in range(len(sent)):
        word = sent[:i+1]
        if len(word) <= 5:
            logp = findlogp1(word)
            if logp != 0:
                heap.put(Entry(word, 0, logp, None))
    while not heap.empty():
        entry = heap.get()
        endindex = entry.start + len(entry.word) - 1
        if chart[endindex] is None:
            chart[endindex] = entry
        else:
            if chart[endindex].logp >= entry.logp:
                continue
            else:
                chart[endindex] = entry
        nstart = endindex + 1
        for i in range(nstart, len(sent)):
            word = sent[nstart:i+1]
            if len(word) <= 5:
                if findlogp1(word) != 0:
                    e = Entry(word, nstart,
                              findlogp2(word, entry.word) + entry.logp, entry)
                    heap.put(e)
    output = traceSeq(chart, sent)
    return output

if __name__ == "__main__":

    Pw = Pdist(opts.counts1w)
    Pw2 = Pdist(opts.counts2w)
    # print len(Pw2), len(Pw)
    old = sys.stdout
    sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
    # ignoring the dictionary provided in opts.counts
    with open(opts.input) as f:
        for line in f:
            utf8line = unicode(line.strip(), 'utf-8')
            output = segment(utf8line, Pw)
            print " ".join(output)
    sys.stdout = old
