from torch.cuda import is_available

datasets_path = "data/datasets"
db_path = "data/db"

device = 'cpu' # 'cuda' if is_available() else 'cpu'
