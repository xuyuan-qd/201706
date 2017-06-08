from __future__ import print_function

import json
import numpy as np
import sys

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.utils import plot_model
from gensim.models import Word2Vec
from pyltp import Segmentor


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def nn():
    model = Sequential()

    model.add(LSTM(300, input_shape=(None, 300)))
    model.add(Dense(28, activation="softmax"))
    # model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    return model


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


def format_input(start, end):
    xs = []
    ys = []
    for item in q_json["questions"][start: end]:
        img_key = item["image_filename"]

        try:
            sentence = item["C_question"].encode('utf8')
        except AttributeError:
            print("ERROR illegal C_question", item, file=sys.stderr)
            continue

        try:
            x = format_x(img_key, sentence)
        except (KeyError, ValueError):
            continue

        xs.append(x)

        answer = item["vec_answer"]
        ys.append(answer)

    xs = np.array(xs)
    ys = np.array(ys)
    print("INFO xs vector shape", xs.shape)
    print("INFO ys vector shape", ys.shape)

    return xs, ys


def main():
    model = nn()
    plot_model(model, to_file="./output/lstm.png", show_shapes=True)

    batch_size = 32
    train_batch_start = 0
    train_batch_end = 20000
    test_batch_start = 20000
    test_batch_end = test_batch_start + batch_size

    for batch in range(train_batch_start, train_batch_end):
        xs, ys = format_input(batch * batch_size, (batch + 1) * batch_size)
        model.fit(xs, ys)

        if batch % 1000 == 0:
            model.save("./output/lstm.h5")
            print("INFO model saved")

        if batch % 100 == 0:
            xtest, ytest = format_input(test_batch_start, test_batch_end)
            print("============================================", file=sys.stderr)
            print("=================test=======================", file=sys.stderr)
            print("================", model.metrics_names, "================", file=sys.stderr)
            score = model.evaluate(xs, ys)
            print("================", score, "=================", file=sys.stderr)


if __name__ == "__main__":
    main()
