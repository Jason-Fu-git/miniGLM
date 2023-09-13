import matplotlib.pyplot as plt


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


def visualize_perplexity(perplexities, val_interval):
    # visualize perplexity of validation and save to perplexity.png
    perplexities_len = len(perplexities)
    x_ticks = [i * val_interval for i in range(perplexities_len)]
    plt.plot(x_ticks, perplexities)
    plt.xlabel("iter")
    plt.ylabel("perplexity")
    plt.title("perplexity of validation")
    plt.savefig("perplexity.png")
    plt.close()


def visualize_rough_l(rough_ls, val_interval):
    # visualize rough_l of validation and save to rough_l.png
    rough_ls_len = len(rough_ls)
    x_ticks = [i * val_interval for i in range(rough_ls_len)]
    plt.plot(x_ticks, rough_ls)
    plt.xlabel("iter")
    plt.ylabel("rough_l")
    plt.title("rough_l of validation")
    plt.savefig("rough_l.png")
    plt.close()
