import os
import sys
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")

name = sys.argv[1]
assert name in ["shediao", "shendiao", "tianlong"], "Unidentified dataset name!"

### TODO: read data from [name]/input.txt
data = None
### TODO: split data for train(0.9) and valid (0.1)
train_data, val_data = None, None
###

### TODO: tokenize raw data with tiktoken encoder
### TODO: transform to numpy array
train_ids, val_ids = None, None
###

### TODO: save numpy array to file [name]/train.bin and [name]/val.bin
###

