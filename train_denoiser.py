import torch, joblib

import numpy as np

from utils.loaders import DatasetClass
from config import device

from torch.utils.data import DataLoader
from models.denoiser import Denoiser
from utils.trainer_denoiser import train

import matplotlib.pyplot as plt

epochs = 200
lr = 1e-2
train_split = .8
batch_size = 32
seqlen=200

data = joblib.load('data/db/database.pt')
for k, v in data.items():
    data[k] = data[k][::5, :]

pose = data['pose']
trans = data['trans']

sep = int(train_split*data['pose'].shape[0])

train_data = {'pose':pose[:sep, :], 'trans':trans[:sep, :]}
val_data = {'pose':pose[sep:, :], 'trans':trans[sep:, :]}

train_loader = DataLoader(DatasetClass(train_data), batch_size = batch_size*seqlen, num_workers=2, drop_last = True)
val_loader = DataLoader(DatasetClass(val_data), batch_size = batch_size*seqlen, num_workers=2, drop_last=True)

model = Denoiser(input_dim = 63, batch_size = batch_size, hidden_dim = 256, seqlen=60, n_layers= 3).to(device)
# model.load_state_dict(torch.load('weights/ckpt.pth', map_location = 'cpu'))

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = lr, momentum=0.9, nesterov=True)

# optimizer = torch.optim.LBFGS(model.parameters(), lr = lr, max_iter = 50, line_search_fn='strong_wolfe')
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = .5, patience = 100)
train_hist, val_hist = train(model, train_loader, val_loader, optimizer, scheduler, epochs)

fig, ax = plt.subplots(1, 2, figsize = (12, 5))

ax[0].plot(train_hist)
ax[0].set_title('train loss')

ax[1].plot(val_hist)
ax[1].set_title('validation loss')

plt.show()