import os
from contextlib import nullcontext
import torch
import tiktoken
import gradio
from model import GLMConfig, MiniGLM

# -----------------------------------------------------------------------------

out_dir = 'pretrain-0912'  # ignored if init_from is not 'resume'
max_new_tokens = 1000  # number of tokens generated in each sample
temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1234
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
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


# encode the beginning of the prompt

# run generation

def answer_generator(question: str, history):
    """
    流式生成
    """
    with torch.no_grad():
        with ctx:
            # 追加中文问号
            if not question.endswith('？'):
                if question.endswith('?'):
                    question.replace('?', '？')
                else:
                    question += '？'
            start_ids = encode(question)
            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

            for i in range(max_new_tokens):
                y = model.streaming_generate(x, temperature=temperature, top_k=top_k)
                output_tokens = y[0].tolist()[len(start_ids):]  # 去除问题部分
                if output_tokens[-1] == 50256:  # 输出终止
                    return
                x = y
                try:
                    end_idx = output_tokens.index(50256)
                    output_tokens = output_tokens[:end_idx]
                except:
                    pass
                output = decode(output_tokens)
                yield output


def QA(question, history):
    """
    一次性生成，目前已经弃用
    """
    with torch.no_grad():
        with ctx:
            start_ids = encode(question)
            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            output_tokens = y[0].tolist()[len(start_ids):]  # 去除问题部分
            try:
                end_idx = output_tokens.index(50256)
                output_tokens = output_tokens[:end_idx]
            except:
                pass
            output = decode(output_tokens)

            return output


chat_box = gradio.ChatInterface(answer_generator)
chat_box.queue()
chat_box.launch()
