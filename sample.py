import os
from contextlib import nullcontext
import torch
import tiktoken
from model import GLMConfig, MiniGLM
import evaluations
import visualize

# -----------------------------------------------------------------------------
# 注：如欲评估困惑度，Rough_L 等，请使用 --start=FILE: 模式

out_dir = 'finetune-0916'  # ignored if init_from is not 'resume'
start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 5  # number of samples to draw
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

save_path = os.path.join(out_dir, 'samples.txt')  # answer output file
label_path = os.path.join(out_dir, 'labels.txt')  # label file

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

save_file = open(save_path, 'w')
if eval_mode:
    label_file = open(label_path, 'r')

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        starts = [line.strip() for line in f.readlines()]

    if eval_mode:
        with open(label_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines()]

    rouge_ls = []  # rough_l 值列表
    perplexity_ls = []  # perplexity 列表
    for index in range(len(starts)):
        start = starts[index]
        start_ids = encode(start)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        # run generation
        with torch.no_grad():
            with ctx:
                if not eval_mode:  # 不是评估模型模式
                    for k in range(num_samples):
                        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                        print("Prompt:", start)
                        output_tokens = y[0].tolist()
                        try:
                            end_idx = output_tokens.index(50256)
                            output_tokens = output_tokens[:end_idx]
                        except:
                            pass
                        output = decode(output_tokens)
                        output = output.replace('\uFFFD', '')  # 去除因截断产生的乱码
                        print(output)
                        save_file.write(output)
                        print('---------------')
                else:  # 评估模型模式
                    rouge_l = 0
                    perplexity = 0
                    for k in range(num_samples):
                        probs = 1
                        print("Prompt:", start)
                        for i in range(max_new_tokens):
                            y, prob = model.streaming_generate(x, temperature=temperature, top_k=top_k)
                            probs *= prob  # 连乘条件概率
                            output_tokens = y[0].tolist()[len(start_ids):]  # 去除问题部分
                            if output_tokens[-1] == 50256:  # 输出终止
                                break
                            x = y
                            try:
                                end_idx = output_tokens.index(50256)
                                output_tokens = output_tokens[:end_idx]
                            except:
                                pass
                            output = decode(output_tokens)
                            output = output.replace('\uFFFD', '')  # 去除因截断产生的乱码
                        print(output)
                        save_file.write(output + '\n')
                        print('---------------')

                        label = labels[index]  # 获取标签
                        rouge_l += evaluations.rouge_l(output, label)  # 计算rouge-l
                        perplexity += evaluations.perplexity(probs)  # 计算困惑度
                    rouge_l /= num_samples
                    perplexity /= num_samples
                    rouge_ls.append(rouge_l)
                    perplexity_ls.append(perplexity)

    if eval_mode:
        visualize.visualize_rouge_l(rouge_ls, out_dir)  # 可视化 rouge_l
        visualize.visualize_perplexity(perplexity_ls, out_dir)  # 可视化 perplexity

else:
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                output_tokens = y[0].tolist()
                try:
                    end_idx = output_tokens.index(50256)
                    output_tokens = output_tokens[:end_idx]
                except:
                    pass
                output = decode(output_tokens)
                output = output.replace('\uFFFD', '')  # 去除因截断产生的乱码

                print(output)
                save_file.write(output)
                print('---------------')

save_file.close()
