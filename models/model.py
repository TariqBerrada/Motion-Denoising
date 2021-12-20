import torch
import torch.nn.functional as F

class Network(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Network, self).__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = torch.nn.Linear(input_dim, 256)
        self.fc2 = torch.nn.Linear(256, 512)
        self.fc3 = torch.nn.Linear(512, 512)
        self.fc4 = torch.nn.Linear(512, 512)
        self.fc5 = torch.nn.Linear(512, 512)
        self.fc6 = torch.nn.Linear(512, 512)
        self.fc7 = torch.nn.Linear(512, latent_dim*2)


        # Decoder
        self.dc1 = torch.nn.Linear(latent_dim, 256)
        self.dc2 = torch.nn.Linear(256, 512)
        self.dc3 = torch.nn.Linear(512, 512)
        self.dc4 = torch.nn.Linear(512, 512)
        self.dc5 = torch.nn.Linear(512, 512)
        self.dc6 = torch.nn.Linear(512, 512)
        self.dc7 = torch.nn.Linear(512, input_dim)


    def reparametrize(self, mu, logvar):
        std = torch.exp(.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        # Encode
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x0 = x
        x = F.leaky_relu(self.fc4(x)) + x0
        x1 = x
        x = F.leaky_relu(self.fc5(x)) + x1
        x2 = x
        x = F.leaky_relu(self.fc6(x)) + x2
        x = self.fc7(x)

        # get latent code params
        mu, logvar = x.view(-1, 2, self.latent_dim)[:, 0, :], x.view(-1, 2, self.latent_dim)[:, 1, :]

        # get latent code
        z = self.reparametrize(mu, logvar)

        # Decode
        x = F.leaky_relu(self.dc1(z))
        x = F.leaky_relu(self.dc2(x))
        x = F.leaky_relu(self.dc3(x))
        x3 = x
        x = F.leaky_relu(self.dc4(x)) + x3
        x4 = x 
        x = F.leaky_relu(self.dc5(x)) + x4
        x5 = x
        x = F.leaky_relu(self.dc6(x)) + x5
        x = self.dc7(x)

        return x, mu, logvar

    def decode(self, z):
        # Decode
        x = F.leaky_relu(self.dc1(z))
        x = F.leaky_relu(self.dc2(x))
        x = F.leaky_relu(self.dc3(x))
        x3 = x
        x = F.leaky_relu(self.dc4(x)) + x3
        x4 = x 
        x = F.leaky_relu(self.dc5(x)) + x4
        x5 = x
        x = F.leaky_relu(self.dc6(x)) + x5
        x = self.dc7(x)

        return x



class CosNetwork(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(CosNetwork, self).__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = torch.nn.Linear(input_dim, 256)
        self.fc2 = torch.nn.Linear(256, 512)
        self.fc3 = torch.nn.Linear(512, 512)
        self.fc4 = torch.nn.Linear(512, 512)
        self.fc5 = torch.nn.Linear(512, 512)
        self.fc6 = torch.nn.Linear(512, 512)
        self.fc7 = torch.nn.Linear(512, latent_dim*2)


        # Decoder
        self.dc1 = torch.nn.Linear(latent_dim, 256)
        self.dc2 = torch.nn.Linear(256, 512)
        self.dc3 = torch.nn.Linear(512, 512)
        self.dc4 = torch.nn.Linear(512, 512)
        self.dc5 = torch.nn.Linear(512, 512)
        self.dc6 = torch.nn.Linear(512, 512)
        self.dc7 = torch.nn.Linear(512, input_dim)


    def reparametrize(self, mu, logvar):
        std = torch.exp(.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        # Encode
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x0 = x
        x = F.leaky_relu(self.fc4(x)) + x0
        x1 = x
        x = F.leaky_relu(self.fc5(x)) + x1
        x2 = x
        x = F.leaky_relu(self.fc6(x)) + x2
        x = self.fc7(x)

        # get latent code params
        mu, logvar = x.view(-1, 2, self.latent_dim)[:, 0, :], x.view(-1, 2, self.latent_dim)[:, 1, :]

        # get latent code
        z = self.reparametrize(mu, logvar)

        # Decode
        x = F.leaky_relu(self.dc1(z))
        x = F.leaky_relu(self.dc2(x))
        x = F.leaky_relu(self.dc3(x))
        x3 = x
        x = F.leaky_relu(self.dc4(x)) + x3
        x4 = x 
        x = F.leaky_relu(self.dc5(x)) + x4
        x5 = x
        x = F.leaky_relu(self.dc6(x)) + x5
        x = self.dc7(x)

        return x, mu, logvar

    def decode(self, z):
        # Decode
        x = F.leaky_relu(self.dc1(z))
        x = F.leaky_relu(self.dc2(x))
        x = F.leaky_relu(self.dc3(x))
        x3 = x
        x = F.leaky_relu(self.dc4(x)) + x3
        x4 = x 
        x = F.leaky_relu(self.dc5(x)) + x4
        x5 = x
        x = F.leaky_relu(self.dc6(x)) + x5
        x = self.dc7(x)

        return x

if __name__ == '__main__':
    model = Network(20, 2)

    inp = torch.zeros(32, 20).float()
    out, _, _ = model(inp)
    
    print(out.shape)