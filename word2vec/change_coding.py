from pyltp import Segmentor

import os

curdir = os.path.dirname(os.path.realpath(__file__))
modelpath = os.path.join(curdir, "ltp_data/cws.model")
segmentor = Segmentor()
segmentor.load(modelpath)


with open('data/news_sohusite_xml.dat', 'rb') as in_file:
    with open('data/news_sohusite.txt', 'w') as out_file:
        for line in in_file:
            pos = line.find('content>')
            if pos != -1:
                new_line = line[9:-11].decode('gb18030').encode('utf8')
                words = segmentor.segment(new_line)
                out_file.write(" ".join(words) + "\n")
            else:
                print(line.decode('gb18030').encode('utf8'))
