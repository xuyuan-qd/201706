from __future__ import print_function

import json


i = open("./data/img_train_vec.json")
o = open("./data/img_train_vec_1000.json", "w")

j = json.load(i)
l = {}

for item in j:
    # CLEVR_train_067254.png
    img_id = int(item[12:18])
    if img_id < 1000:
        l[item] = j[item]

print(l)
json.dump(l, o)

i.close()
o.close()

