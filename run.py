
# -*- coding: utf-8 -*-


import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
import networkx as nx
from scipy.special import logit as nplogit
from numpy import random
import pandas as pd
import multiprocess as mp

import os,sys,optparse,time,pathlib,pickle,math,copy,random,subprocess,re,itertools

def run_clear(case_num):
    text_path = '/path/to/your/data/' + str(case_num)
    
    string = 'mpirun -np 10 python Color_based_code.py --AF ' + text_path + '/AF.txt  --Data ' + text_path  +'/data.pkl --S_es ' + text_path + '/S_0.txt --W_es ' + text_path + '/W_0.txt'
    print(string)
    Run = subprocess.run(string, shell=True)
    return 'Done'

if __name__ == '__main__':
    case_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Only case numbers now
    p = mp.Pool(1)
    p.map(run_clear, case_numbers)
    p.close()
    p.join()
