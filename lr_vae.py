from __future__ import print_function
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.distributions import Normal
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Load MNIST data
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.ToTensor()), 
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(784, 400)  
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

        self.relu = nn.ReLU()

    def sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp()
            return Variable(torch.normal(mu.data, logvar.data), requires_grad=True)
        else:
            return mu

    def encode(self, x):
        h = self.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        std = logvar.mul(0.5).exp()
        return Normal(mu, std)
            

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(20, 400)
        self.fc2 = nn.Linear(400, 784)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        return self.sigmoid(self.fc2(h))
        

encoder = Encoder()
decoder = Decoder()
if args.cuda:
    encoder.cuda()
    decoder.cuda()

enc_opt = optim.Adam(encoder.parameters(), lr=1e-3)
dec_opt = optim.Adam(decoder.parameters(), lr=1e-3)


def loss_function(recon_x, x, mu, var):
    """Reconstruction + KL divergence losses summed over all elements and batch."""
    bce = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False, reduce=False).sum(dim=1)
    kl = -0.5 * torch.sum(1 + var.log() - mu.pow(2) - var, dim=1)
    return kl, bce, (bce + kl).sum()


def train(epoch, prefix=''):
    encoder.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda: data = data.cuda()

        # pass datapoints through
        normal = encoder(data)
        z = normal.sample()
        recon_batch = decoder(z)
        kl, bce, dec_loss = loss_function(recon_batch, data, normal.mean, normal.variance)
    
        # decoder backward
        dec_opt.zero_grad()
        dec_loss.backward(retain_graph=True)
        dec_opt.step() 
        
        # encoder backward
        enc_opt.zero_grad()
        logprobs = normal.log_prob(z).sum(dim=1)
        enc_loss = (logprobs*bce + kl).sum()
        enc_loss.backward()
        enc_opt.step()
        
        train_loss += dec_loss.data[0]
        if batch_idx % args.log_interval == 0:
            print('{}Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                prefix, epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                dec_loss.data[0] / len(data)))

    train_loss /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss))
    return train_loss


def test(epoch):
    encoder.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data)
        normal = encoder(data)
        z = normal.sample()
        recon_batch = decoder(z)
        _, _, loss = loss_function(recon_batch, data, normal.mean, normal.variance)
        test_loss += loss.data[0]

        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                  recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


if __name__ == '__main__':
    if not os.path.isdir('results'):
        os.makedirs('results')
        
    train_bounds = []
    test_bounds = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)
             
        train_bounds.append(-train_loss)
        test_bounds.append(-test_loss)
    
        # generate sample in numpy and convert into torch Variable, to get around mysterious
        # bug, where invoking torch's rand num gen somehow causes training loss to diverge 
        sample = np.random.normal(size=(64, 20)).astype(np.float32)
        sample = Variable(torch.from_numpy(sample))
    
        if args.cuda:
            sample = sample.cuda()
        sample = decoder(sample).cpu()
        save_image(sample.data.view(64, 1, 28, 28), 'results/sample_' + str(epoch) + '.png')
    
        # plot loss graphs
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(train_bounds, label='Train')
        ax.plot(test_bounds, label='Test')
        ax.set_title('MNIST (with reparameterization), N=20')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Lower bound')
        ax.legend()
        fig.savefig('results.png')

