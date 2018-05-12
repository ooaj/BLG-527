'''
This script finds creates the .csv file according to our need. 
'''

import pandas as pd
import numpy as np
import os

dust = pd.read_csv('/home/ooa/Downloads/LittleOne/kaggle2.csv')
ids = dust["id"]

idss = ids.tolist()
idss = list(map(str, idss))
idss.sort()
idss = list(map(int, idss))

prds = dust["predicted"]
prdss = prds.tolist()

dff = pd.DataFrame(
    {'id': idss,
     'predicted': prdss
    })

dff2 = dff.sort_values(["id", "predicted"])
dff2.to_csv('predicted.csv')
ekleme = dff2["id"]
ekleme = ekleme.tolist()

def find_missing_items(int_list):
    original_set  = set(int_list)
    smallest_item = min(original_set)
    largest_item  = max(original_set)
    full_set = set(range(smallest_item, largest_item + 1))
    return sorted(list(full_set - original_set))

eksikler = find_missing_items(ekleme)
sampl = np.random.randint(low=1, high=128, size=(len(eksikler),))

dff4 = pd.DataFrame(
    {'id': eksikler,
     'predicted': sampl
    })

dff3 = dff2
frames = [dff3, dff4]
dff5 = pd.concat(frames)
eklemee = dff5["id"]
eklemee = eklemee.tolist()
dff6 = dff5.sort_values(["id", "predicted"])
my_array = dff6["id"]
my_array = my_array.tolist()
dff6.to_csv('predicted1.csv')
