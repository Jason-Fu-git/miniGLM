import os
import sys
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")

names = sys.argv[1:]

### TODO: read data from ([name]/input.txt for name in names)
### TODO: combine multiple books into one single data file
### TODO: split data for train(0.9) and valid (0.1)
train_data, val_data = None, None
###

### TODO: tokenize raw data with tiktoken encoder
### TODO: transform to numpy array
train_ids, val_ids = None, None
###

# save numpy array to file [name]/train.bin and [name]/val.bin
train_ids.tofile(os.path.join("processed_pretrain", "train.bin"))
val_ids.tofile(os.path.join("processed_pretrain", 'val.bin'))
