import torch, tqdm, os

from config import device

import matplotlib.pyplot as plt

criterion = torch.nn.MSELoss()

def VAELoss(data, target, mu, logvar, weight = 1.):

    recloss = criterion(data, target)
    KLdiv = -0.5*torch.sum(1+logvar - mu.pow(2) - logvar.exp())

    return 1000*recloss + 0.01*weight*KLdiv

def fit(model, loader, optimizer, scheduler):
    model.train()
    
    running_loss = 0.0

    for i, data in enumerate(loader):
        pose = data['pose'].float().to(device)

        optimizer.zero_grad()
        reconstruction, mu, logvar = model(pose)

        loss = VAELoss(pose, reconstruction, mu, logvar)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
    
    running_loss /= len(loader.dataset)
    return running_loss

def validate(model, loader):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(loader):
            pose = data['pose'].float().to(device)

            reconstruction, mu, logvar = model(pose)

            loss = VAELoss(pose, reconstruction, mu, logvar)
            running_loss += loss.item()

    
    running_loss /= len(loader.dataset)
    return running_loss

def train(model, train_loader, val_loader, optimizer, scheduler, n_epochs, save_dir = "./weights"):
    train_hist, val_hist = [], []
    for epoch in tqdm.tqdm(range(n_epochs)):
        
        tloss = fit(model, train_loader, optimizer, scheduler)
        vloss = validate(model, val_loader)

        train_hist.append(tloss)
        val_hist.append(vloss)

        print(f"{epoch} - train_loss : {tloss} - validation_loss : {vloss}")

        if epoch%5 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, "ckpt.pth"))
    
            fig, ax = plt.subplots(1, 2, figsize = (12, 5))

            ax[0].semilogy(train_hist)
            ax[0].set_title('train loss')

            ax[1].semilogy(val_hist)
            ax[1].set_title('validation loss')
            plt.savefig('learning.jpg')
            plt.close()
    
    return train_hist, val_hist
             