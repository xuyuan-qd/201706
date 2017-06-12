# coding: utf8

from __future__ import print_function

import json
import random
import keras
import numpy as np
import heapq
import os
import shutil
from gensim.models import Word2Vec
from pyltp import Segmentor

from util import eprint, iprint, dprint
from iat import iat

answers = [u'0', u'1', u'10', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'blue', u'brown', u'cube', u'cyan', u'cylinder', u'gray', u'green', u'large', u'metal', u'no', u'purple', u'red', u'rubber', u'small', u'sphere', u'yellow', u'yes']
model = keras.models.load_model("./output/lstm.h5")

w2v_model = Word2Vec.load("./word2vec/model.out")
segmentor = Segmentor()
segmentor.load("./word2vec/ltp_data/cws.model")

q_file = open("./data/C_train_Q.json")
q_json = json.load(q_file)
i_file = open("./data/img_train_vec.json")
i_json = json.load(i_file)


def format_x(img_key, sentence):
    try:
        img = i_json[img_key]
    except KeyError:
        print("ERROR image not found", img_key, file=sys.stderr)
        raise
    img_v = np.repeat(img, 5)
    x = np.empty((1, 300))
    x[0] = img_v

    try:
        words = segmentor.segment(sentence)
        for word in words:
            try:
                word_v = w2v_model.wv[word]
            except KeyError:
                print("ERROR illegal in word", word, file=sys.stderr)
                raise
            x = np.append(x, [word_v], axis=0)
        # print("INFO x vector shape", x.shape)
    except KeyError:
        raise

    maxlen = 50
    if len(x) > maxlen:
        print("ERROR sentence too long", sentence, file=sys.stderr)
        raise ValueError("sentence too long")

    for i in range(len(x), maxlen):
        x = np.append(x, np.zeros((1, 300)), axis=0)

    return x


def predict(question, img_key):
    xs = []
    x = format_x(img_key, question)
    xs.append(x)
    xs = np.array(xs)

    ret = model.predict(xs)[0]
    ans_key = heapq.nlargest(3, xrange(len(ret)), key=ret.__getitem__)
    ans = [answers[i] for i in ans_key]
    return ans


def main():
    files = os.listdir("./data/wav")
    # q_key = random.choice(files)
    q_key = "相比紫色大金属立方体，绿色大物体更多么.wav"
    iprint("question file", q_key)
    shutil.copy("./data/wav/" + q_key, "./wav/iflytek02.wav")
    question = iat()
    iprint("question", question)

    for rep in range(1):
        images = i_json.keys()
        img_key = random.choice(images)
        img_key = "CLEVR_train_000000.png"
        iprint("image", img_key, i_json[img_key])

        ans = predict(question, img_key)
        iprint("top 3 answers:", " ".join(ans))


if __name__ == "__main__":
    main()
