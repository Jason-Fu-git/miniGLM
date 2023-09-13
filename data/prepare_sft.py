import os
import sys
import tiktoken
import numpy as np
import json

# run this file to divide data into train and valid

enc = tiktoken.get_encoding("gpt2")

# read data from "sft_data.jsonl"
# split data for train(0.9) and valid (0.1)
# tokenize raw data with tiktoken encoder
encoded_train_data = []
encoded_val_data = []
with open("sft_data.jsonl", 'r', encoding='utf-8') as f:
    for i, line in enumerate(f.readlines()):
        dic = json.loads(line)
        question = dic["Question"]
        answer = dic["Answer"]
        if i % 10 == 9:
            encoded_train_data.append(enc.encode_ordinary(question + answer) + [enc.eot_token])
        else:
            encoded_val_data.append(enc.encode_ordinary(question + answer) + [enc.eot_token])

print(len(encoded_train_data), len(encoded_val_data))

# get max length
max_len = max(len(max(encoded_train_data, key=lambda x: len(x))), len(max(encoded_val_data, key=lambda x: len(x))))

# calculate mask
train_mask = np.zeros((len(encoded_train_data), max_len), dtype=np.uint16)
val_mask = np.zeros((len(encoded_val_data), max_len), dtype=np.uint16)

# transform to numpy array
train_ids = np.ones((len(encoded_train_data), max_len), dtype=np.uint16) * enc.eot_token
for i, line in enumerate(encoded_train_data):
    index = np.argwhere(np.array(line) == enc.encode_ordinary('？')[0])
    assert index.shape[1] == 1
    index = index[0, 0] + len(enc.encode_ordinary('？'))
    train_mask[i, index:len(line)] = 1
    train_ids[i, :len(line)] = line
print(train_ids.shape, train_ids.dtype)

val_ids = np.ones((len(encoded_val_data), max_len), dtype=np.uint16) * enc.eot_token
for i, line in enumerate(encoded_val_data):
    index = np.argwhere(np.array(line) == enc.encode_ordinary('？')[0])
    assert index.shape[1] == 1
    index = index[0, 0] + len(enc.encode_ordinary('？'))
    val_mask[i, index:len(line)] = 1
    val_ids[i, :len(line)] = line
print(val_ids.shape, val_ids.dtype)

assert val_ids.shape[1] == train_ids.shape[1]
assert val_ids.shape[1] == val_mask.shape[1]
assert val_ids.shape[1] == train_mask.shape[1]

# save numpy array to file [name]/train.bin and [name]/val.bin
train_ids.tofile(os.path.join("processed_finetune", "train.bin"))
val_ids.tofile(os.path.join("processed_finetune", 'val.bin'))
train_mask.tofile(os.path.join("processed_finetune", 'train_mask.bin'))
val_mask.tofile(os.path.join("processed_finetune", 'val_mask.bin'))
np.array([val_ids.shape[1]], dtype=np.uint16).tofile(os.path.join("processed_finetune", 'dim1.bin'))
