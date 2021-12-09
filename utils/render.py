import torch, tqdm
import numpy as np

import sys, cv2
sys.path.append('.')

import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

import matplotlib.pyplot as plt

from os import path as osp


support_dir = './support_data/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

subject_gender = 'male'

bm_fname = osp.join(support_dir, 'body_models/smplh/{}/model.npz'.format(subject_gender))
dmpl_fname = osp.join(support_dir, 'body_models/dmpls/{}/model.npz'.format(subject_gender))


num_betas = 16 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

bm = BodyModel(bm_path=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, path_dmpl=dmpl_fname).to(device)
faces = c2c(bm.f)

def vis_body_pose_hand(mv, body_pose_hand, fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_hand.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    # show_image(body_image)
    return body_image

def render_pose_sequence(poses, transl = None, betas = None, out_dir = 'renderings/render.mp4', resolution = 600, fps = 60):
    """Renders an animation from a pose and translation sequence.

    Args:
        pose ([type]): [description]
        transl ([type]): [description]
        resolution (int, optional): [video resolution] : int 600 or tuple (768, 1024).
    """
    if type(resolution) == int:
        size = (resolution, resolution)
        mv = MeshViewer(width=resolution, height=resolution, use_offscreen=True)
    else:
        size = (resolution[0], resolution[1])
        mv = MeshViewer(width=resolution[0], height=resolution[1], use_offscreen=True)
    
    if betas is None:
        betas = torch.zeros(1, 16).float().to(device)

    print(f"Rendering animation : {poses.shape[0]} frames - {fps} frames/sec. - resolution{size}")
    
    out = cv2.VideoWriter(out_dir,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for fId in tqdm.tqdm(range(poses.shape[0])):
        anim = {'pose_body':poses[[fId], ...], 'betas' : betas}
        body_pose_hand = bm(**anim)
        frame = vis_body_pose_hand(mv, body_pose_hand)
        out.write(frame)
    out.release()

def render_pose(poses, betas = None, out_dir = 'renderings/frame.jpg', resolution = 600, show = False):
    if type(resolution) == int:
        size = (resolution, resolution)
        mv = MeshViewer(width=resolution, height=resolution, use_offscreen=True)
    else:
        size = (resolution[0], resolution[1])
        mv = MeshViewer(width=resolution[0], height=resolution[1], use_offscreen=True)
    
    if betas is None:
        bteas = torch.zeros(1, 16).float().to(device)
    
    anim = {'pose_body' : poses[[0], ...], 'betas' : betas}
    body_pose_hand = bm(**anim)
    frame = vis_body_pose_hand(mv, body_pose_hand)

    plt.imsave(out_dir, frame)

    if show:
        plt.figure()
        plt.imshow(frame)
        plt.axis("off")
        plt.show()

    return frame

    
if __name__ == '__main__':
    amass_npz_fname = osp.join(support_dir, 'github_data/amass_sample.npz') # the path to body data
    bdata = np.load(amass_npz_fname)

    time_length = len(bdata['trans'])

    body_parms = {
        'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(device), # controls the global root orientation
        'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(device), # controls the body
        'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(device), # controls the finger articulation
        'trans': torch.Tensor(bdata['trans']).to(device), # controls the global body position
        'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(device), # controls the body shape. Body shape is static
        'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(device) # controls soft tissue dynamics
    }

    # render_pose_sequence(body_parms['pose_body'], body_parms['betas'])
    

    T_pose  = torch.zeros(1, 63).float().to(device)

    # render_pose(body_parms['pose_body'][[0], ...], resolution = (2560, 2560), show = True)
    render_pose(T_pose, resolution = (2560, 2560), show = True)
