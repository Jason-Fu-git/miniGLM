import os
import sys

from dataset_infos import infos, prompts

name = sys.argv[1]
assert name in ["sanguo", "xiyou", "honglou", "shuihu"], "Unidentified dataset name!"
prompt = prompts[name]

with open(os.path.join(name, "prompts.txt"), 'w') as prompts_file:
    for p in prompt:
        prompts_file.write(p + '\n')
