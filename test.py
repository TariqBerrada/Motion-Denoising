import joblib, torch
from utils.render import render_pose, render_pose_sequence
from config import device

from models.model import Network

import matplotlib.pyplot as plt

model = Network(63, 50).to(device)
model.load_state_dict(torch.load('weights/ckpt.pth', map_location = 'cpu'))

data = joblib.load('data/db/database.pt')
pose = torch.tensor(data['pose'][98:248, 3:66]).float().to(device)

pred, _, _ = model(pose)

print(pose.shape, pred.shape)

render_pose_sequence(pose, fps = 120, out_dir="renderings/gt.mp4")
render_pose_sequence(pred, fps = 120, out_dir="renderings/pred.mp4")

# gt = render_pose(pose)
# p = render_pose(pred[[0], 3:66])

# fig,ax = plt.subplots(1, 2, figsize = (15, 7))
# ax[0].imshow(gt)
# ax[0].set_title('ground truth')
# ax[0].axis('off')
# ax[1].imshow(p)
# ax[1].set_title('reconstruction')
# ax[1].axis('off')
# plt.show()