
import sys
import codecs
import optparse
import os
from math import log
from operator import attrgetter

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
(opts, _) = optparser.parse_args()


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


class Entry():
    def __init__(self, word, start, end, logprob, backptr):
        self.word = word
        self.start = start
        self.end = end
        self.logprob = logprob
        self.backptr = backptr

# the default segmenter does not use any probabilities, but you could ...
Pw = Pdist(opts.counts1w)


def NewStart(x):
    if x is None:
        return 0
    else:
        return x+1


def WordFinder(heap, endindx, prob, seq, entry):
    word = ''
    for i in range(NewStart(endindx), len(seq)):
        word = word + seq[i]
        if word in Pw:
            newentry = Entry(word, NewStart(endindx),
                             NewStart(endindx)+len(word),
                             prob+log(Pw(word)), entry)
            heap.append(newentry)
            heap = sorted(heap, key=attrgetter('logprob'), reverse=True)
    return heap

old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
# ignoring the dictionary provided in opts.counts

with open(opts.input) as f:
    for line in f:
        heap = []
        utf8line = unicode(line.strip(), 'utf-8')
        output = [i for i in utf8line]
        chart = [None]*len(output)
        heap = WordFinder(heap, None, 0, output, None)
        while len(heap):
            entry = heap[0]
            heap.remove(heap[0])
            endindex = entry.end - 1
            if chart[endindex] == None:
                chart[endindex] = entry
            else:
                if chart[endindex].logprob < entry.logprob:
                    chart[endindex] = entry
                # else:
                #    continue
            heap = WordFinder(heap, endindex, entry.logprob, output, entry)

        finalindex = len(output)-1
        finalentry = chart[finalindex]
        segment = []
        while finalentry:
            segment.append(finalentry.word)
            finalentry = finalentry.backptr
        segment.reverse()
        print " ".join(segment)

# sys.stdout = old
