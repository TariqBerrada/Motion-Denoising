import torch
import torch.nn.functional as F

import numpy as np

from models.model import Network

class Denoiser(torch.nn.Module):
    def __init__(self, input_dim, batch_size, hidden_dim, seqlen, n_layers, bidirectional = True):
        super(Denoiser, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.seqlen = seqlen
        
        device = 'cuda'

        self.nh = 3*28 # number of harmonics
        self.Fs = 1000 # sampling frequency
        self.f0 = self.Fs//(4*self.seqlen) # fundamental frequency

        self.nz = 28 # dimension of latent code

        self.vae = Network(63, 28)
        self.vae.load_state_dict(torch.load('weights/ckpt.pth', map_location='cpu'))

        for p in self.vae.parameters():
            p.requires_grad = False

        self.lstm = torch.nn.LSTM(
            input_size = self.input_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.n_layers,
            bidirectional = True,
            batch_first = True
        )

        self.fc1 = torch.nn.Linear(2*self.hidden_dim, 28*(2*self.nh+1)) 
        self.out = torch.nn.Linear(28*(2*self.nh+1), 28*(2*self.nh+1)) 

        S = np.zeros((self.batch_size, self.nz, self.seqlen, self.nh))
        for i in range(self.seqlen):
            for j in range(self.nh):
                S[:, :, i, j] = 2*np.pi*((i*self.f0)/self.Fs)*j

        # freq has shape 28, seqlen, nh
        self.freq = torch.tensor(S, device = 'cuda', dtype = torch.float32, requires_grad = False)

    def get_code(self, A, phi, b, k = 10):
        freq = self.freq.transpose(-2, -1)
        # Impose sparsness by keep only n frequencies .
        # A : [b, 28, nh]
        s = A.shape
        A = A.reshape(-1, A.shape[-1])
        shift = torch.topk(A, k=2, dim = -1, sorted = True).values[:, -1]
        A = F.relu(A - shift) + shift
        A = A.reshape(s)

        freq = torch.sin(((freq.T+phi.T).T).transpose(-2, -1)) # calculate the frequencies for each harmonic.
        code = torch.einsum('bzsh, bzh -> bzs', self.freq, A) + b # multiply by the amplitudes.
        return code
        
    def forward(self, x):

        # Initialize cell and hidden states
        self.hidden = torch.zeros(2*self.n_layers, self.batch_size, self.hidden_dim, device = 'cuda') # [D*nlayers, batch_size, hidden_size] D = 2 if bidir else 1
        self.state = torch.zeros(2*self.n_layers, self.batch_size, self.hidden_dim, device = 'cuda') # [D*nlayers, batch_size, hidden_size]

        self.hidden, self.state = self.lstm(x, (self.hidden, self.state))
        
        x = self.hidden[:, -1, :] # 32, 60, 512 -> 32, 512 # keep only last hidden state
        x = F.elu(self.fc1(x))
        
        x = self.out(x) # output phase and amplitude features (batch_size, 2*nh+1)
        x = x.reshape(x.shape[0], 28, 2*self.nh+1)
        
        z = self.get_code(x[:, :, :self.nh], x[:, :, self.nh:2*self.nh], x[:, :, -1]) # didn't add the bias here.
        z = z.reshape(-1, 28)
        x = self.vae.decode(z)

        x = x.reshape(self.batch_size, self.seqlen, 63)
        return x
        
if __name__ == '__main__':
    model = Denoiser(3, 8, 12, 10, 2)

    data = torch.zeros(8, 10, 3)

    out = model(data)
    print(out.shape)
