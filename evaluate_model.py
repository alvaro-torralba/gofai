from sklearn.metrics import confusion_matrix

import math
import os.path
import pickle
from sys import exit

#from learning.translate.pddl_parser import *
#from learning.translate.pddl_parser import lisp_parser
#from learning.translate.pddl_parser import parsing_functions

import importlib
rule_evaluator = importlib.import_module("./fd-partial-grounding/src/subdominization-training/")

def evaluate_model(model, training_data):

    for task in training_data:
        print(model, task)
    exit()
