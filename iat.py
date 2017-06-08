from __future__ import print_function

import os

def iat():
    exe = os.popen('./demo')
    question = exe.read()
    exe.close()
    return question


if __name__ == "__main__":
    print(iat())
