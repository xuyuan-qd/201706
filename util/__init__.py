from __future__ import print_function

import sys


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


def eprint(*args, **kwargs):
    print(color.RED + "ERROR" + color.END, *args, file=sys.stderr, **kwargs)

def iprint(*args, **kwargs):
    print(color.YELLOW + "INFO" + color.END, *args, file=sys.stderr, **kwargs)

def dprint(*args, **kwargs):
    print(color.BLUE + "DEBUG" + color.END, *args, file=sys.stderr, **kwargs)
