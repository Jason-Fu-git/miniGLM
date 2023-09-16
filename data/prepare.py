import os
import sys
import tiktoken
import numpy as np

# run this file using command (>>>python prepare.py [name1] [name2] ...)

# run this file to divide data into train and valid

enc = tiktoken.get_encoding("gpt2")

names = sys.argv[1:]

# read data from ([name]/input.txt for name in names)
os.remove('joined_input.txt')
for name in names:
    assert name in ["shediao", "shendiao", "tianlong", "shujian", "xiake", "yitian", "baima", "bixue", "xiaoao",
                    "yuenv",
                    "liancheng", "xueshan", "feihu", "yuanyang", "luding"], "Unidentified dataset name!"

    with open(os.path.join(name, "input.txt"), "r") as f:
        raw_data = f.read().replace('\n\n', '\n').replace(' ', '').replace('	', '').replace('\u0020', '').replace(
            '\u3000', '')  # strip the unwanted symbols
        with open("joined_input.txt", "a") as jf:
            jf.write(raw_data)  # combine multiple books into one single data file

# split data for train(0.9) and valid (0.1)
train_data = []
val_data = []
with open("joined_input.txt", "r") as f:  # load data
    raw_data = f.read()
    # format raw.txt data
    paragraphs = [para.strip() for para in
                  raw_data.split("\n")]
    # split data into train and valid
    for i, paragraph in enumerate(paragraphs):
        if len(paragraph) == 0:  # empty paragraph
            continue
        # every ten lines, the 10th line is used for validation
        if i % 10 == 9:
            val_data.append(paragraph)
        else:
            train_data.append(paragraph)
print(len(train_data), len(val_data))

# tokenize raw.txt data with tiktoken encoder
encoded_train_data = [enc.encode_ordinary('\n'.join(train_data))]
encoded_val_data = [enc.encode_ordinary('\n'.join(val_data))]
print(len(encoded_train_data), len(encoded_val_data))

# transform to numpy array
train_ids = np.array(encoded_train_data, dtype=np.uint16).T
print(train_ids.shape, train_ids.dtype)

val_ids = np.array(encoded_val_data, dtype=np.uint16).T
print(val_ids.shape, val_ids.dtype)

# save numpy array to file [name]/train.bin and [name]/val.bin
train_ids.tofile(os.path.join("processed_pretrain", "train.bin"))
val_ids.tofile(os.path.join("processed_pretrain", 'val.bin'))
