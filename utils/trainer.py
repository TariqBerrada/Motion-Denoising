import torch, tqdm, os

from config import device
from utils.render import get_joints

import matplotlib.pyplot as plt

criterion = torch.nn.MSELoss()
criterion2 = torch.nn.MSELoss()
criterion_grad = torch.nn.MSELoss()

def GradLoss(model, data, target, eps = 0.1):
    directions = torch.rand_like(data).float().to(device)
    pred_p, _, _ = model(target + eps*directions)
    pred_m, _, _ = model(target - eps*directions)

    grad_p = torch.abs(pred_p - data)
    grad_m = torch.abs(pred_m - data)

    return criterion_grad(grad_p, grad_m)


def VAELoss(data, target, mu, logvar, weight = 1.):

    data_J = get_joints(data).reshape(data.shape[0], -1)
    target_J = get_joints(target).reshape(data.shape[0], -1)

    recloss = criterion(data, target)
    recloss2 = criterion(data_J, target_J)
    KLdiv = -0.5*torch.sum(1+logvar - mu.pow(2) - logvar.exp(), dim = 1)

    return 1000*recloss + recloss2 + 0.01*weight*KLdiv.mean(), 1000*recloss +recloss2, 0.1*weight*KLdiv.mean()

def fit(model, loader, optimizer, scheduler):
    model.train()
    
    running_loss = 0.0
    running_rec = 0.0
    running_kl = 0.0

    for i, data in enumerate(loader):
        
        optimizer.zero_grad()

        pose = data['pose'].float().to(device)
        reconstruction, mu, logvar = model(pose)

        loss, rec, kl = VAELoss(pose, reconstruction, mu, logvar)

        gloss = GradLoss(model, reconstruction, pose)
        
        loss  = loss + 1000*gloss

        running_loss += loss.item()
        running_rec += rec.item()
        running_kl += kl.item()

        loss.backward()
        

        optimizer.step()
        
    
    running_loss /= len(loader.dataset)
    running_rec /= len(loader.dataset)
    running_kl /= len(loader.dataset)
    return running_loss, running_rec, running_kl

def validate(model, loader):
    model.eval()
    running_loss = 0.0
    running_rec = 0.0
    running_kl = 0.0
    with torch.no_grad():
        for i, data in enumerate(loader):
            pose = data['pose'].float().to(device)

            reconstruction, mu, logvar = model(pose)

            loss, rec, kl = VAELoss(pose, reconstruction, mu, logvar)
            running_loss += loss.item()
            running_rec += rec.item()
            running_kl += kl.item()

    
    running_loss /= len(loader.dataset)
    running_rec /= len(loader.dataset)
    running_kl /= len(loader.dataset)
    return running_loss, running_rec, running_kl


def train(model, train_loader, val_loader, optimizer, scheduler, n_epochs, save_dir = "./weights"):
    train_hist, val_hist = [], []
    train_rec, val_rec = [], []
    train_kl, val_kl = [], []

    for epoch in tqdm.tqdm(range(n_epochs)):
        
        tloss, trec, tkl = fit(model, train_loader, optimizer, scheduler)
        vloss, vrec, vkl = validate(model, val_loader)

        scheduler.step(vloss)

        train_hist.append(tloss)
        val_hist.append(vloss)
        train_rec.append(trec)
        val_rec.append(vrec)
        train_kl.append(tkl)
        val_kl.append(vkl)

        print(f"{epoch} - train_loss : {tloss} - validation_loss : {vloss}")

        if epoch%5 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, "ckpt.pth"))
    
            fig, ax = plt.subplots(2, 3, figsize = (23, 10))

            ax[0][0].semilogy(train_hist)
            ax[0][0].set_title('train loss')

            ax[0][1].semilogy(train_rec)
            ax[0][1].set_title('train reconstruction loss')

            ax[0][2].semilogy(train_kl)
            ax[0][2].set_title('train KL loss')

            ax[1][0].semilogy(val_hist)
            ax[1][0].set_title('validation loss')

            ax[1][1].semilogy(val_rec)
            ax[1][1].set_title('validation reconstruction loss')

            ax[1][2].semilogy(val_kl)
            ax[1][2].set_title('validation KL loss')

            plt.savefig('learning.jpg')
            plt.close()
    
    return train_hist, val_hist
             