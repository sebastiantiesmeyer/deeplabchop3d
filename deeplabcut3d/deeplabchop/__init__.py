import os

DEBUG = True and 'DEBUG' in os.environ and os.environ['DEBUG']

from deeplabchop import new, status, extract, label, util, wizard#, #predict, draw#, training
