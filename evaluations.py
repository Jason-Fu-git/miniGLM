# Implement metrics Perplexity, Rouge-L, etc.
import math
import numpy as np
import torch


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


def perplexity(probs):
    """
    Calculate perplexity
    :param probs 各步的条件概率, 是一维torch向量
    :return: perplexity
    """
    m = len(probs)
    return math.exp(- math.log(torch.prod(probs).item()) / m)  # 这样算是为了减少误差
