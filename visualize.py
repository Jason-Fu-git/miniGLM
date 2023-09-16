import matplotlib.pyplot as plt
import numpy as np


def visualize_loss(train_loss_list, train_interval, val_loss_list, val_interval, dataset, out_dir):
    # visualize loss of training & validation and save to [out_dir]/loss.png
    train_loss_len = len(train_loss_list)
    val_loss_len = len(val_loss_list)
    train_x_ticks = [(i + 1) * train_interval for i in range(train_loss_len)]
    val_x_ticks = [i * val_interval for i in range(val_loss_len)]  # 注意区别
    plt.plot(train_x_ticks, train_loss_list, label="train loss")
    plt.plot(val_x_ticks, val_loss_list, label="validation loss")
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.title(f"loss of training & validation in dataset {dataset}")
    plt.legend()
    plt.savefig(out_dir + "/loss.png")
    plt.close()


def visualize_perplexity(perplexities, out_dir):
    # visualize perplexity of validation and save to perplexity.png
    perplexities = np.array(perplexities)
    plt.hist(perplexities)  # 绘制频数直方图
    plt.xlabel("perplexity")
    plt.ylabel("frequency")
    plt.title(f"perplexity evaluation, mean_value = {np.mean(perplexities)}")
    plt.savefig(out_dir + "/perplexity.png")
    plt.close()


def visualize_rouge_l(rouge_ls, out_dir):
    # visualize rouge_l of validation and save to rouge_l.png
    rouge_ls = np.array(rouge_ls)
    plt.hist(rouge_ls)  # 绘制频数直方图
    plt.xlabel("rouge_l")
    plt.ylabel("frequency")
    plt.title(f"rouge_l evaluation, mean value = {np.mean(rouge_ls)}")
    plt.savefig(out_dir + "/rouge_l.png")
    plt.close()
