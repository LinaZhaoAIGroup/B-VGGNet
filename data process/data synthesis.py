#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

if not os.path.exists('temp'):
    os.mkdir('temp')


# In[2]:


def DataSynthesis(i, Ts, Es):
    synthetic_data = []
    for j in range(Es.shape[0]):
        e = random.uniform(0, 1)
        T = Ts.values[i, 1:-1]
        if Es.values[j, -1] != Ts.values[i, -1]:
            continue
        else:
            E = Es.values[j, 1:-1]
            last_column = Es.values[j, -1]
            Xac = [E_val + e * T_val for E_val, T_val in zip(E, T)]
            max_value = max(Xac)
            min_value = min(Xac)
            scaled_Xac = [100 * (X_val - min_value) / (max_value - min_value) if i < len(Xac) - 1 else X_val for i, X_val in enumerate(Xac)]
            scaled_Xac.append(last_column)
            synthetic_data.append(np.array(scaled_Xac))

    if synthetic_data:
        columns = list(Es.columns[1:-1]) + [Es.columns[-1]]
        df = pd.DataFrame(np.array(synthetic_data), columns=columns)
        df.to_csv('temp/{}.csv'.format(i), index=False)
        
    else:
        print("No data for index {}".format(i))

