from matplotlib import pyplot as plt


def plot_loss(loss, path):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    x = list(range(len(loss)))
    ax.plot(x, loss)
    plt.savefig(path, dpi=400)
