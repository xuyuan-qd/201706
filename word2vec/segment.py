from pyltp import Segmentor

import os

curdir = os.path.dirname(os.path.realpath(__file__))
modelpath = os.path.join(curdir, "ltp_data/cws.model")
segmentor = Segmentor()
segmentor.load(modelpath)


with open('chineseQ.txt', 'r') as in_file:
    with open('data/chineseQ_segmented.txt', 'w') as out_file:
        for line in in_file:
            words = segmentor.segment(line)
            out_file.write(" ".join(words) + "\n")

