import joblib, torch
import numpy as np
from utils.render import render_pose, render_pose_sequence
from config import device

from models.model import Network

import matplotlib.pyplot as plt

model = Network(63, 28).to(device)
model.load_state_dict(torch.load('weights/ckpt.pth', map_location = 'cpu'))

data = joblib.load('data/db/database.pt')
pose = torch.tensor(data['pose'][98:248, 3:66]).float().to(device)

pred, _, _ = model(pose)

# print(pose.shape, pred.shape)
print('Running reconstruction test.')

render_pose_sequence(pose, fps = 60, out_dir="renderings/gt.mp4")
render_pose_sequence(pred, fps = 60, out_dir="renderings/pred.mp4")

print('Running interpolation test.')

pose0 = torch.tensor(data['pose'][[105300], 3:66]).float().to(device)
pose1 = torch.tensor(data['pose'][[1600], 3:66]).float().to(device)

sequence = torch.cat([w*pose0 + (1-w)*pose1 for w in np.linspace(0, 1, 250)])
sequence_p, _, _ = model(sequence)

render_pose(pose0, out_dir = 'renderings/pose0.jpg')
render_pose(pose1, out_dir = 'renderings/pose1.jpg')

render_pose_sequence(sequence, fps = 60, out_dir="renderings/interpolation_gt.mp4")
render_pose_sequence(sequence_p, fps = 60, out_dir="renderings/interpolation_p.mp4")

def pred_set(nx=4, ny = 4):
    n = nx*ny
    poses = torch.tensor(data['pose'][np.random.randint(0, data['pose'].shape[0], size = n), 3:66]).float().to(device)
    poses_rec, _, _  = model(poses)
    
    frames_gt = []
    frames_pred = []

    for i in range(poses.shape[0]):
        frame = render_pose(poses[[i], :])
        frames_gt.append(frame)

        frame_pred = render_pose(poses_rec[[i], :])
        frames_pred.append(frame_pred)

    frames = np.array(frames_gt)
    frames_gt = frames.reshape((nx, ny, *frame.shape))[:, :, 100:-100, 100:-100, :]

    frames_pred = np.array(frames_pred)
    frames_pred = frames_pred.reshape((nx, ny, *frame.shape))[:, :, 100:-100, 100:-100, :]
    
    fig, ax = plt.subplots(ny, 2*nx, figsize = (40, 20))
    
    for i in range(nx):
        for j in range(0, 2*ny-1, 2):
            
            ax[i][j].imshow(frames_gt[i, j//2])
            ax[i][j+1].imshow(frames_pred[i, j//2])
            ax[i][j].axis('off')
            ax[i][j+1].axis('off')
    fig.tight_layout(pad = 0)
    plt.savefig('renderings/preds.jpg')

pred_set()




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