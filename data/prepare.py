import os
import sys
import tiktoken
import numpy as np
from tqdm import tqdm

from dataset_infos import infos

name = sys.argv[1]
assert name in ["sanguo", "xiyou", "honglou", "shuihu"], "Unidentified dataset name!"
info = infos[name]

# encode
enc = tiktoken.get_encoding("gpt2")

train_ids, val_ids = [], []
for i in tqdm(range(info["filenum"])):
    input_file_path = f"{name}/raw_data/{i}.html"
    with open(input_file_path, 'r') as f:
        data = f.read()
    para_ids = enc.encode_ordinary(data)
    assert data == enc.decode(para_ids)
    if i % 10 == 9:
        val_ids.extend(para_ids + [enc._special_tokens['<|endoftext|>']])
    else:
        train_ids.extend(para_ids + [enc._special_tokens['<|endoftext|>']])

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(name, 'train.bin'))
val_ids.tofile(os.path.join(name, 'val.bin'))
