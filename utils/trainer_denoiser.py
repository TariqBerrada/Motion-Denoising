import torch, tqdm, os

from config import device
from utils.render import get_joints

import matplotlib.pyplot as plt

criterion = torch.nn.MSELoss()

batch_size = 256
seqlen = 60

def fit(model, loader, optimizer, scheduler):
    model.train()
    
    running_loss = 0.0

    for i, data in enumerate(loader):
        
        optimizer.zero_grad()

        pose = data['pose'].float().to(device)
        pose = pose.reshape(batch_size, seqlen, -1)
        noise =  torch.normal(0, 0.03, size = pose.shape, device = pose.device, dtype = torch.float32)
        reconstruction = model(pose+noise)

        loss =criterion(pose, reconstruction)

        # gloss = GradLoss(model, reconstruction, pose)
        
        # loss  = loss + 1000*gloss

        running_loss += loss.item()

        loss.backward()
        
        optimizer.step()
        
    running_loss /= len(loader.dataset)
    
    return running_loss

def validate(model, loader):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            pose = data['pose'].float().to(device)

            pose = pose.reshape(batch_size, seqlen, -1)
            noise =  torch.normal(0, 0.03, size = pose.shape, device = pose.device, dtype = torch.float32)
            reconstruction = model(pose+noise)

            loss =criterion(pose, reconstruction)

            running_loss += loss.item()
    
    running_loss /= len(loader.dataset)
    return running_loss


def train(model, train_loader, val_loader, optimizer, scheduler, n_epochs, save_dir = "./weights"):
    train_hist, val_hist = [], []
    
    for epoch in tqdm.tqdm_notebook(range(n_epochs)):
        
        tloss = fit(model, train_loader, optimizer, scheduler)
        vloss = validate(model, val_loader)

        scheduler.step(vloss)

        train_hist.append(tloss)
        val_hist.append(vloss)
        
        print(f"{epoch} - train_loss : {tloss} - validation_loss : {vloss}")

        if epoch%5 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, "ckpt_denoiser.pth"))
    
            fig, ax = plt.subplots(2, 1, figsize = (23, 10))

            ax[0].semilogy(train_hist)
            ax[0].set_title('train loss')

            ax[1].semilogy(val_hist)
            ax[1].set_title('validation loss')

            plt.savefig('learning.jpg')
            plt.close()
    
    return train_hist, val_hist
             