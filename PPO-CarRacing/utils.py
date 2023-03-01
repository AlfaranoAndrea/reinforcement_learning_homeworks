import numpy as np
import matplotlib.pyplot as plt
import torch
def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]

    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    if norm:
        # normalize
        gray = gray / 128. - 1.
    return gray

def plot_scores(scores,avg_scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores, label="Score")
    plt.plot(np.arange(1, len(avg_scores)+1), avg_scores, label="Avg on 100 episodes")
    plt.legend(bbox_to_anchor=(1.05, 1)) 
    plt.ylabel('Score')
    plt.xlabel('Episodes #')
    plt.savefig('plot.png')

img_stack = 4
transition = np.dtype(
    [
        ("s", np.float64, (img_stack, 96, 96)),
        ("a", np.float64, (3,)),
        ("a_logp", np.float64),
        ("r", np.float64),
        ("s_", np.float64, (img_stack, 96, 96)),
    ]
)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)