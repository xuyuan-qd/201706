from __future__ import print_function

import json
import numpy as np
import sys

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.utils import plot_model
from gensim.models import Word2Vec
from pyltp import Segmentor


maxlen = 20


def nn():
    model = Sequential()

    model.add(LSTM(28, input_shape=(None, 300)))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    return model


def main():
    model = nn()
    plot_model(model, to_file="./output/lstm.png", show_shapes=True)

    w2v_model = Word2Vec.load("./word2vec/model.out")
    segmentor = Segmentor()
    segmentor.load("./word2vec/ltp_data/cws.model")

    q_file = open("./data/C_train_Q.json")
    q_json = json.load(q_file)
    i_file = open("./data/img_train_vec.json")
    i_json = json.load(i_file)

    for batch in range(10000):
        xs = []
        ys = []
        print("INFO starting batch", batch)
        for item in q_json["questions"][batch * 32: batch * 32 + 32]:
            img_key = item["image_filename"]
            try:
                img = i_json[img_key]
            except KeyError:
                print("ERROR image not found", img_key, file=sys.stderr)
                continue
            img_v = np.repeat(img, 5)
            x = np.empty((1, 300))
            x[0] = img_v

            try:
                sentence = item["C_question"].encode('utf8')
            except AttributeError:
                print("ERROR illegal C_question", item, file=sys.stderr)
                continue

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
                continue

            if len(x) > 25:
                print("ERROR sentence too long", sentence, file=sys.stderr)
                continue

            maxlen = 25
            for i in range(len(x), maxlen):
                x = np.append(x, np.zeros((1, 300)), axis=0)
            xs.append(x)

            answer = item["vec_answer"]
            ys.append(answer)

        xs = np.array(xs)
        ys = np.array(ys)
        print("INFO xs vector shape", xs.shape)
        print("INFO ys vector shape", ys.shape)
        model.fit(xs, ys)

        if batch % 100 == 0:
            model.save("./output/lstm.h5")
            print("INFO model saved")


if __name__ == "__main__":
    main()
