import torch, joblib

import numpy as np

from utils.loaders import DatasetClass
from config import device

from torch.utils.data import DataLoader
from models.model import Network
from utils.trainer import train

import matplotlib.pyplot as plt

epochs = 100
lr = 1e-3
train_split = .8
batch_size = 128

data = joblib.load('data/db/database.pt')

permu = np.random.permutation(list(range(data['pose'].shape[0])))

pose = data['pose'][permu, :]
trans = data['trans'][permu, :]

sep = int(train_split*data['pose'].shape[0])

train_data = {'pose':pose[:sep, :], 'trans':trans[:sep, :]}
val_data = {'pose':pose[sep:, :], 'trans':trans[sep:, :]}

train_loader = DataLoader(DatasetClass(train_data), batch_size = batch_size, num_workers=2)
val_loader = DataLoader(DatasetClass(val_data), batch_size = batch_size, num_workers=2)

model = Network(63, 50).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 1e-5, 1e-2)
train_hist, val_hist = train(model, train_loader, val_loader, optimizer, scheduler, epochs)

fig, ax = plt.subplots(1, 2, figsize = (12, 5))

ax[0].plot(train_hist)
ax[0].set_title('train loss')

ax[1].plot(val_hist)
ax[1].set_title('validation loss')

plt.show()