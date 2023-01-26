#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 22/11/2022 10:09 am
# @Author  : Wizard Chenhan Zhang
# @FileName: util.py
# @Software: PyCharm

import torch
import numpy as np
import pandas as pd
import networkx as nx
from texttable import Texttable
import csv
from scipy.sparse.linalg import eigs
import random
import os


def set_seed(seed):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True