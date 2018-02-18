import numpy as np
import matplotlib.pyplot as plt

from reparam_vae import train as vae_train
from lr_vae import train as lrvae_train


NB_EPOCHS = 50

vae_bounds, lrvae_bounds = [], []
for epoch in range(1, NB_EPOCHS + 1):
    vae_loss = vae_train(epoch, 'vae. ')
    lrvae_loss = lrvae_train(epoch, 'lr-vae. ')
    
    vae_bounds.append(-vae_loss)
    lrvae_bounds.append(-lrvae_loss)

    # plot loss graphs
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1, epoch+1), vae_bounds, label='reparam-vae')
    ax.plot(np.arange(1, epoch+1), lrvae_bounds, label='lr-vae')
    ax.set_title('MNIST train statistics, N=20')
    ax.set_xlabel('Epoch')
    ax.set_xlim(xmin=1)
    ax.set_ylabel('Lower bound')
    ax.legend()
    fig.savefig('results.png')

