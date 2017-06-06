from __future__ import division
from __future__ import print_function


import os
import re
from gensim.models import Word2Vec
from pyltp import Segmentor


# sentences = [['hello', 'my', 'name', 'is', 'michael'], ['how', 'are', 'you']]
# model = Word2Vec(sentences, min_count=0)
# For more information refer to https://rare-technologies.com/word2vec-tutorial/

# LTP is used for segmentation
# For more information refer to http://pyltp.readthedocs.io/zh_CN/latest/api.html#id4


class MySentences(object):
    def __init__(self, datadir, modeldir):
        self.datadir = datadir
        self.modelpath = os.path.join(modeldir, "cws.model")
        self.segmentor = Segmentor()
        self.segmentor.load(self.modelpath)

    def __iter__(self):
        for fname in ["chineseQ_segmented.txt", "news_sohusite.txt":
            fpath = os.path.join(self.datadir, fname)
            with open(fpath) as f:
                for line in f:

                    # print(line.decode('utf8'))
                    # chinese = re.findall('[\u4e00-\u9fff]+', line)
                    # chinese = "".join(chinese)
                    # words = self.segmentor.segment(line)
                    words = line.split()
                    print("\t".join(words))
                    yield words

def main():
    curdir = os.path.dirname(os.path.realpath(__file__))
    datadir = os.path.join(curdir, "data")
    print(datadir)
    modeldir = os.path.join(curdir, "ltp_data")
    print(modeldir)
    sentences = MySentences(datadir, modeldir)
    model = Word2Vec(sentences, min_count=5, size=300, workers=12)
    outdir = os.path.join(curdir, "model.out")
    print(outdir)
    model.save(outdir)

if __name__ == "__main__":
    main()
