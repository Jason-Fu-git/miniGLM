import os
from contextlib import nullcontext
import torch
import tiktoken
from model import GLMConfig, MiniGLM
import json

# -----------------------------------------------------------------------------
# 注：如欲评估困惑度，Rough_L 等，请使用 --start=FILE: 模式

out_dir = 'finetune-0916'  # ignored if init_from is not 'resume'
input_data = 'input.jsonl'
output_data = 'output.jsonl'
max_new_tokens = 1500  # number of tokens generated in each sample
temperature = 0.9  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 100  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1234
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
eval_mode = False  # if True, run in eval mode (will visualize perplexity and Rough_L)
exec(open('configurator.py').read())  # overrides from command line or config file

# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
config = GLMConfig(**checkpoint['model_args'])
model = MiniGLM(config)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)  # 这里增加了关键词

model.eval()
model.to(device)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

output_file = open(output_data, 'w+')
input_file = open(input_data, 'r')

# encode the beginning of the prompt
starts = [json.loads(line)['question'].strip() for line in input_file.readlines()]  # 开始序列列表

for start in starts:
    question = start
    # 追加中文问号
    if not question.endswith('？'):
        if question.endswith('?'):
            question.replace('?', '？')
        else:
            question += '？'
    start_ids = encode(question)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    # run generation
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print("Prompt:", start)
            output_tokens = y[0].tolist()
            try:
                end_idx = output_tokens.index(50256)
                output_tokens = output_tokens[len(start_ids):end_idx]
            except:
                pass
            output = decode(output_tokens)
            output = output.replace('\uFFFD', '')  # 去除因截断产生的乱码
            print(output)
            dic = {'question': start, 'answer': output}
            output_file.write(json.dumps(dic, ensure_ascii=False) + '\n')
            print('---------------')

output_file.close()
input_file.close()
