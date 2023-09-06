# MiniGLM

prepare data:
```bash
cd data/
python fetch_data.py [dataset_name] # download
python prepare.py [dataset_name] # tokenize
```

train scripts:
```bash
python train.py config/train_config.py --dataset=[dataset_name]
```

sample scripts:
```bash
python sample.py --out_dir=[/dir/to/training/output] --save_path=/path/to/save/output # or add prompts by --start=FILE:/path/to/prompts.txt
```
