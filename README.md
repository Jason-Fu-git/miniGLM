# MiniGLM

MiniGLM目前已提供or已给出框架的部分内容如下列举。

## 数据预处理
首先进入数据目录:
```bash
cd data/
```

- 下载数据：

    ```bash
    python fetch_data.py [dataset_names] # download
    ```
    其中，你可以通过参数`[dataset_name]`来指定需要下载的数据集。可以下载的数据集包括：
    - shediao: 射雕英雄传
    - shendiao: 神雕侠侣
    - tianlong: 天龙八部

- 数据预处理（需实现）：

    ```
    python prepare.py [dataset_names] # tokenize
    ```
    通过`[dataset_names]`指定若干个数据集，将他们统一处理为一份数据（包含训练集`train.bin`与验证集`val.bin`）。

## 模型训练

通过运行如下命令启动训练：
```bash
python train.py config/train_config.py --dataset=[dataset_name]
```
其中`--dataset`参数指定使用数据在`data/`下的二级目录名。

## 模型推理

通过运行如下命令加载训练完毕的模型权重进行推理：

```bash
python sample.py --out_dir=[/dir/to/training/output] --save_path=/path/to/save/output # or add prompts by --start=FILE:/path/to/prompts.txt
```

其中：
- `--out_dir`参数指定使用的模型权重的目录（由模型训练过程生成）。
- `--save_path`参数指定生成文本的保存路径，不设置则不保存仅打印。
- `--start`参数可以设置指导模型生成的prompt。可以在`prompts.txt`文件中逐行给出输入的各个prompt
