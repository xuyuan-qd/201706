# coding: utf8

from __future__ import print_function

import random
import keras
import numpy as np
import heapq
import os
import shutil

from lstm import q_json, i_json, format_x
from iat import iat

answers = [u'0', u'1', u'10', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'blue', u'brown', u'cube', u'cyan', u'cylinder', u'gray', u'green', u'large', u'metal', u'no', u'purple', u'red', u'rubber', u'small', u'sphere', u'yellow', u'yes']
model = keras.models.load_model("./output/lstm.h5")


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


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
    print("INFO question file", q_key)
    shutil.copy("./data/wav/" + q_key, "./wav/iflytek02.wav")
    question = iat()
    print("INFO question", question)
    print(q_json["questions"][0])

    for rep in range(1):
        images = i_json.keys()
        img_key = random.choice(images)
        img_key = "CLEVR_train_000000.png"
        print("INFO image", img_key, i_json[img_key])

        image = i_json[img_key]
        image_valid = [i for i in image if i != -1]
        correct_answer = len(image_valid) // 6
        print("INFO correct answer:", correct_answer)

        ans = predict(question, img_key)
        print("INFO top 3 answers:", " ".join(ans))

        # if ans.count(str(correct_answer)):
            # print("INFO verdict: " + color.GREEN + "correct" + color.END + " in top3")
        # else:
            # print("INFO verdict: " + color.RED + "wrong" + color.END + " in top3")


if __name__ == "__main__":
    main()
