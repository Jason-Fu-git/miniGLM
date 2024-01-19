# miniGLM

清华大学计算机系2023年暑期程序设计训练大作业

miniGLM是一个基于GLM的中文问答系统。数据来源为金庸的小说，sft数据为人工+机器生成的关于金庸小说的问题。因模型规模有限，效果一般。


## 一、数据规模

#### 预训练

预训练仅使用了《神雕侠侣》《射雕英雄传》《天龙八部》三部小说，共8.75MB，其中$\frac{9}{10}$作为训练集，$\frac{1}{10}$作为验证集。

#### 微调

微调数据共1814个“问答对”，由包括我在内4位同学得到的数据汇总而成。其中$\frac{9}{10}$作为训练集，$\frac{1}{10}$作为验证集。对于我个人负责的部分，数据构造过程如下：

1. 个人参照阅读记忆，手写30条，与助教所给的20条合并，作为ChatGPT和ChatGLM学习的数据。
2. ChatGPT学习后，分多个批次生成关于《神雕侠侣》《射雕英雄传》《天龙八部》的各100条左右的数据。
3. 同ChatGPT的使用，首先让ChatGLM学习这50个问题。
4. 利用ChatGLM api 生成关于这三部小说的各100条左右的问题，再利用ChatGLM api 生成答案。

数据汇总后，起初共2000条左右，经人工筛选，去除重复的、质量不佳的问题后，剩余1814条。



## 二、预训练效果

#### loss曲线

预训练共20000个循环，其余参数保持不变，loss曲线收敛到2以下。

<img src="assets\loss.png" alt="loss" style="zoom:80%;" />

#### 生成效果

预训练后“续写”效果如下：（已设置续写部分不会再重复prompt部分）

![image-20230916172953559](assets\image-20230916172953559.png)

续写内容基本符合原作风格，没有出现乱码，也没有出现大段重复的文字，基本达到要求。

#### 生成速度

在`max_new_tokens=1500`的条件下，本地首次生成的速度大概在10s左右，之后每次生成的速度在2s以内。



## 三、微调

从微调数据转为可供模型训练数据的过程如下：

1. 逐行读取`sft_data.jsonl`文件，利用`tiktoken`编码，拼接问题与答案，并在最后追加`[enc.eot_token]`。每十行中，前九行作为训练集，最后一行作为验证集。分别存到两个列表中。

2. 计算最长行的长度，把其他所有行填充`enc.eot_token`到最长行的长度，形成`numpy`数组
3. 构建`mask`，仅答案部分的值设为1（包括答案结尾部分的**一个**`enc.eot_token`）
4. 把上述4个数组存储到文件里，并存储最长行的长度，方便reshape

每一批次数据获取过程如下：

1. 首先读取数组，并reshape成原先的形状。
2. 随机挑选`batch_size`行，从每行中随机挑选`block_size`个连续的文字作为`X`，向后错开一个文字作为标签，并获取对应的`mask`

保证选取的`batch_size`和`block_size`不越界。



## 四、微调效果

#### loss曲线

微调共10000个循环，`batch_size`设为64，loss曲线收敛到0.1左右。

<img src="assets\loss-1694852412861-2.png" alt="loss" style="zoom:80%;" />

#### 问答效果

![image-20230916162431344](assets\image-20230916162431344.png)

答案没有提前截断，没有乱码，部分情况下存在离题现象，不过仍有一部分问题回答得还算不错。

#### 生成速度

在`max_new_tokens=1500`的条件下，本地首次生成的速度大概在10s左右，之后每次生成的速度在2s以内。



## 五、`gradio` 可视化对话

在`gradio`可视化构造的过程中，设置了问题自动补全问号，答案文本不自动重复问题等功能。可视化过程采用chatbot主流的流式生成过程，即**动态依次生成各个文字**，而不是一次性生成所有文字，以提高用户体验。该功能的实现依赖于在`model.py`添加的`streaming_generate()`接口。



## 六、模型效果评估

生成内容及生成速度详见“预训练效果”及“微调效果”部分。

#### `Rouge_L`评估

核心代码如下：

```python
def rouge_l(X, Y, beta=1):
    """
    Calculate ROUGE-L score of two text collections of sentences.
    :param X: 参考答案
    :param Y: 模型生成的回答
    :param beta: 超参数
    :return: ROUGE-L score of two text collections of sentences
    """
    # 采用动态规划算法计算LCS
    print(X)
    m, n = len(X), len(Y)
    dp = np.zeros((m, n), dtype=np.uint16)
    for i in range(1, m):
        for j in range(1, n):
            if X[i] == Y[j]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m - 1][n - 1]  # 最长公共子序列
    # 计算ROUGE-L
    R_lcs = lcs / m
    P_lcs = lcs / n
    return (1 + beta ** 2) * R_lcs * P_lcs / (R_lcs + beta ** 2 * P_lcs)
```

上述代码中，首先利用动态规划算法计算最长子序列，再套用公式计算。

从问题中抽样25个，得到的`Rouge_L`频数直方图如下：

<img src="assets\rouge_l.png" alt="rouge_l" style="zoom:80%;" />

平均值为0.46，大概有40% 的问题能够获得`Rouge_L`值大于0.6的较满意回答。

#### 困惑度评估

核心算法如下：

```python
@torch.no_grad()
    def streaming_generate(self, idx, temperature=1.0, top_k=None):
        """
        此函数与generate()相似，用于流式生成。返回生成拼接后的Tensor和生成该字符的条件概率
        :param idx: 前一部分文字在字典里的index
        :param temperature:
        :param top_k:
        :return: 元组 (idx, prob_idx_next)
        """
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = self(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
        return idx, probs[0, idx_next]
    
def perplexity(probs):
    """
    Calculate perplexity
    :param probs 各步的条件概率, 是一维torch向量
    :return: perplexity
    """
    m = len(probs)
    return math.exp(- math.log(torch.prod(probs).item()) / m)  # 这样算是为了减少误差  
```

该算法巧妙地利用了可视化部分自定义的`streaming_generate()`函数，每次打包返回生成内容和新生成`token`的条件概率，在`perplexity()`函数中累乘这些条件概率，再套用公式得到困惑度。

从问题中抽样25个，得到的困惑度频数直方图如下：

<img src="assets\perplexity.png" alt="perplexity" style="zoom:80%;" />

参数`top_K`设置为100，困惑度均值为1.82。

#### 评估功能使用方式

在`sample.py`中增加了接口，只需在命令行中输入

```shell
python sample.py --out_dir=finetune-0916 --start=FILE:finetune-0916/input.txt --eval_mode=True
```

即可。

注：`start`必须采用`FILE`模式，生成的频数直方图保存在`out_dir`路径下。



## 七、实验感想

受微调数据量的限制，最后得到的对话模型没能尽善尽美。但通过本次实验，我仍对大模型训练中的预训练部分及微调部分有了更加深刻的认识，对LLM的架构也有了了解，并尝试模仿写出逐字生成的函数，收获颇丰。

用时而言，归功于服务器强大的GPU，训练用时并不长，基本在半小时内结束训练。

