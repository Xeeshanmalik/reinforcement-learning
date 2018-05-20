import numpy as np


'''helper taken from'''
'''https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5'''


def prepro(I):
    I = I[35:195] # crop
    I = I[::2, ::2, 0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


'''take ID float array of rewards and compute discounted rewards'''

def discount_rewards(r, gamma):
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r.tolist()
