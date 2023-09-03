import os
import sys

from dataset_infos import infos

name = sys.argv[1]
assert name in ["sanguo", "xiyou", "honglou", "shuihu"], "Unidentified dataset name!"
info = infos[name]

os.makedirs(name, exist_ok=True)
os.makedirs(os.path.join(name, "raw_data"), exist_ok=True)
for i in range(info["filenum"]):
    os.system(f"wget --no-check-certificate https://raw.githubusercontent.com/luoxuhai/chinese-novel/master/resources/{info['keyword']}/{i}.html -O {name}/raw_data/{i}.html")
    