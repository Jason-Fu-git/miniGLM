import os
import sys

name = sys.argv[1]
assert name in ["shediao", "shendiao", "tianlong", "shujian", "xiake", "yitian", "baima", "bixue", "xiaoao", "yuenv",
                "liancheng", "xueshan", "feihu", "yuanyang", "luding"], "Unidentified dataset name!"

# run this file to fetch data from GitHub

links = {
    "shendiao": "https://github.com/loyalpartner/jywxFenxi/raw/master/神雕侠侣.txt",
    "shediao": "https://github.com/loyalpartner/jywxFenxi/raw/master/射雕英雄传.txt",
    "tianlong": "https://github.com/loyalpartner/jywxFenxi/raw/master/天龙八部.txt",
    "shujian": "https://github.com/loyalpartner/jywxFenxi/raw/master/书剑恩仇录.txt",
    "xiake": "https://github.com/loyalpartner/jywxFenxi/raw/master/侠客行.txt",
    "yitian": "https://github.com/loyalpartner/jywxFenxi/raw/master/倚天屠龙记.txt",
    "baima": "https://github.com/loyalpartner/jywxFenxi/raw/master/白马啸西风.txt",
    "bixue": "https://github.com/loyalpartner/jywxFenxi/raw/master/碧血剑.txt",
    "xiaoao": "https://github.com/loyalpartner/jywxFenxi/raw/master/笑傲江湖.txt",
    "yuenv": "https://github.com/loyalpartner/jywxFenxi/raw/master/越女剑.txt",
    "liancheng": "https://github.com/loyalpartner/jywxFenxi/raw/master/连城诀.txt",
    "xueshan": "https://github.com/loyalpartner/jywxFenxNi/raw/master/雪山飞狐.txt",
    "feihu": "https://github.com/loyalpartner/jywxFenxi/raw/master/飞狐外传.txt",
    "yuanyang": "https://github.com/loyalpartner/jywxFenxi/raw/master/鸳鸯刀.txt",
    "luding": "https://github.com/loyalpartner/jywxFenxi/raw/master/鹿鼎记.txt"
}

os.makedirs(name, exist_ok=True)
os.system(f"wget --no-check-certificate {links[name]} -O {os.path.join(name, 'input.txt')}")
print(os.path.join(name, 'input.txt'))
