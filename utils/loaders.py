from torch.utils.data import Dataset

class DatasetClass(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return self.data['pose'].shape[0]
    
    def __getitem__(self, idx):
        sample = {'pose': self.data['pose'][idx, 3:66], 'trans': self.data['trans'][idx, :]}
        return sample