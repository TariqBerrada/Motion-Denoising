import torch

import torch.nn.functional as F

class Denoiser(torch.nn.Module):
    def __init__(self, input_dim, batch_size, hidden_dim, seqlen, n_layers, bidirectional = True):
        super(Denoiser, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.seqlen = seqlen

        m = (self.seqlen*self.input_dim)//2

        self.mapping1 = torch.nn.Linear(self.seqlen*self.input_dim, m, bias = False)
        self.mapping2 = torch.nn.Linear(self.seqlen*self.input_dim, self.seqlen*self.input_dim - m, bias = False)

        self.mapping1.weight.data /= .2*self.mapping1.weight.data.norm()
        self.mapping2.weight.data /= .2*self.mapping2.weight.data.norm()


        self.mapping1.requires_grad = False
        self.mapping2.requires_grad = False

        self.lstm = torch.nn.LSTM(
            input_size = self.input_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.n_layers,
            bidirectional = True,
            batch_first = True
        ) # input [batch_size, seqlen, nfeatures]

        # self.hidden = torch.randn(2*self.n_layers, self.batch_size, self.hidden_dim) # [D*nlayers, batch_size, hidden_size] D = 2 if bidir else 1
        # self.state = torch.randn(2*self.n_layers, self.batch_size, self.hidden_size) # [D*nlayers, batch_size, hidden_size]

        self.fc1 = torch.nn.Linear(2*self.seqlen*self.hidden_dim, self.seqlen*self.input_dim) # [batch_size, seqlen, D*hidden_dim]
        self.out = torch.nn.Linear(self.seqlen*self.input_dim, self.seqlen*self.input_dim) # [batch_size, seqlen, D*hidden_dim]

    def forward(self, x):

        # Initialize cell and hidden states
        self.hidden = torch.zeros(2*self.n_layers, self.batch_size, self.hidden_dim) # [D*nlayers, batch_size, hidden_size] D = 2 if bidir else 1
        self.state = torch.zeros(2*self.n_layers, self.batch_size, self.hidden_size) # [D*nlayers, batch_size, hidden_size]


        x= x.reshape(x.shape[0], -1)
        x1 = torch.cos(self.mapping1(x))
        x2 = torch.sin(self.mapping2(x))

        xf = F.relu(torch.cat((x1, x2), dim = -1))

        xf = xf.reshape(xf.shape[0], self.seqlen, self.input_dim)
        self.hidden, self.state = self.lstm(xf, (self.hidden, self.state))
        
        x = self.hidden.view(self.hidden.shape[0], -1)

        # x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        
        x = self.out(x)
        x = x.reshape(x.shape[0], self.seqlen, self.input_dim)

        return x
        
if __name__ == '__main__':
    model = Denoiser(3, 8, 12, 10, 2)

    data = torch.zeros(8, 10, 3)

    out = model(data)
    print(out.shape)
