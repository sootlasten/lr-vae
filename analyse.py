import numpy as np
import matplotlib.pyplot as plt

from reparam_vae import train as vae_train
from lr_vae import train as lrvae_train
from lr_vae import get_network


lrvae1_enc, lrvae1_dec, lrvae1_encopt, lrvae1_decopt = get_network()
lrvae5_enc, lrvae5_dec, lrvae5_encopt, lrvae5_decopt = get_network()
lrvae15_enc, lrvae15_dec, lrvae15_encopt, lrvae15_decopt = get_network()

NB_EPOCHS = 30

vae_bounds, lrvae1_bounds, lrvae5_bounds, lrvae15_bounds = [], [], [], []
for epoch in range(1, NB_EPOCHS + 1):
    vae_loss = vae_train(epoch, 'vae. ')

    lrvae1_loss = lrvae_train(lrvae1_enc, lrvae1_dec, lrvae1_encopt, lrvae1_decopt, 
                              epoch, 1, prefix='lr-vae (1 sample). ')
    lrvae5_loss = lrvae_train(lrvae5_enc, lrvae5_dec, lrvae5_encopt, lrvae5_decopt, 
                              epoch, 5, prefix='lr-vae (5 samples). ')
    lrvae15_loss = lrvae_train(lrvae15_enc, lrvae15_dec, lrvae15_encopt, lrvae15_decopt, 
                              epoch, 15, prefix='lr-vae (15 samples). ')
    
    vae_bounds.append(-vae_loss)
    lrvae1_bounds.append(-lrvae1_loss)
    lrvae5_bounds.append(-lrvae5_loss)
    lrvae15_bounds.append(-lrvae15_loss)

    # plot loss graphs
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1, epoch+1), vae_bounds, label='reparam-vae')
    ax.plot(np.arange(1, epoch+1), lrvae1_bounds, label='lr-vae (1 sample)')
    ax.plot(np.arange(1, epoch+1), lrvae5_bounds, label='lr-vae (5 samples)')
    ax.plot(np.arange(1, epoch+1), lrvae15_bounds, label='lr-vae (15 samples)')
    ax.set_title('MNIST train statistics, N=20')
    ax.set_xlabel('Epoch')
    ax.set_xlim(xmin=1)
    ax.set_ylabel('Lower bound')
    ax.legend()
    fig.savefig('results.png')

