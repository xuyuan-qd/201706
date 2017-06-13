from __future__ import print_function
from __future__ import absolute_import

import json
import cPickle

from gensim.models import Word2Vec
from pyltp import Segmentor

from util import eprint, iprint, dprint

w2v_model = Word2Vec.load("./word2vec/model.out")
segmentor = Segmentor()
segmentor.load("./word2vec/ltp_data/cws.model")


def translate():
    fin = open("./data/C_train_Q.json")
    jin = json.load(fin)

    fout = open("./data/C_train_Q_split.dat", "wb")
    for item in jin["questions"]:
        cPickle.dump(item, fout, protocol=cPickle.HIGHEST_PROTOCOL)

    fin.close()
    fout.close()


def to_vec():
    fin = open("./data/C_train_Q_split.dat", "rb")

    fout = open("./data/C_train_Q_split_vec.dat", "wb")

    while True:
        try:
            item = cPickle.load(fin)
            words = segmentor.segment(item["C_question"].encode("utf8"))
            words_vec = []
            for word in words:
                words_vec.append(w2v_model.wv[word])
            item["C_question_vec"] = words_vec
            cPickle.dump(item, fout, protocol=cPickle.HIGHEST_PROTOCOL)
        except (AttributeError, KeyError):
            eprint(item)
            continue
        except EOFError:
            break


if __name__ == "__main__":
    # translate()
    to_vec()

