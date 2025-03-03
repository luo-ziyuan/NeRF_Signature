import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .utils_wtmk import get_rays
import copy
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from einops import rearrange

# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1.0, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


def poses_circle(size, device, radius):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.tensor([np.pi/3 for _ in range(size)]).to(device)
    phis = torch.linspace(0, 2*np.pi, steps=size + 1).to(device)
    phis = phis[:-1]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:
    def __init__(self, opt, device, downscale=1, type='train', n_test=10):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose

        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.poses = []
            self.images = []
            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = int(image.shape[0] // downscale)
                    self.W = int(image.shape[1] // downscale)

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.poses.append(pose)
                self.images.append(image)
            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])


    def collate(self, index):

        B = len(index) # a list of length 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H * self.W / self.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],    
            }

        poses = self.poses[index].to(self.device) # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]
        
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map, self.opt.patch_size)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
            
        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader


# class NeRFDataset_fixview:
#     def __init__(self, opt, keyposes_dir, keyposes_save_dir, device, downscale=1, type='train', n_test=100, pretrained_model=None):
#         super().__init__()
        
#         self.opt = opt
#         self.device = device
#         self.type = type # train, val, test
#         self.downscale = downscale
#         self.root_path = opt.path
#         self.preload = opt.preload # preload data into GPU
#         self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
#         self.offset = opt.offset # camera offset
#         self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
#         self.fp16 = opt.fp16 # if preload, load into fp16.

#         self.training = self.type in ['train', 'all', 'trainval']
#         self.num_rays = self.opt.num_rays if self.training else -1

#         self.rand_pose = opt.rand_pose
#         self.pretrained_model = pretrained_model.to(device)
#         self.n_test = n_test
#         # auto-detect transforms.json and split mode.
#         if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
#             self.mode = 'colmap' # manually split, use view-interpolation for test.
#         elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
#             self.mode = 'blender' # provided split
#         else:
#             raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

#         # load nerf-compatible format data.
#         if self.mode == 'colmap':
#             with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
#                 transform = json.load(f)
#         elif self.mode == 'blender':
#             # load all splits (train/valid/test), this is what instant-ngp in fact does...
#             if type == 'all':
#                 transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
#                 transform = None
#                 for transform_path in transform_paths:
#                     with open(transform_path, 'r') as f:
#                         tmp_transform = json.load(f)
#                         if transform is None:
#                             transform = tmp_transform
#                         else:
#                             transform['frames'].extend(tmp_transform['frames'])
#             # load train and val split
#             elif type == 'trainval':
#                 with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
#                     transform = json.load(f)
#                 with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
#                     transform_val = json.load(f)
#                 transform['frames'].extend(transform_val['frames'])
#             # only load one specified split
#             else:
#                 with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
#                     transform = json.load(f)

#         else:
#             raise NotImplementedError(f'unknown dataset mode: {self.mode}')

#         # load image size
#         if 'h' in transform and 'w' in transform:
#             self.H = int(transform['h']) // downscale
#             self.W = int(transform['w']) // downscale
#         else:
#             # we have to actually read an image to get H and W later.
#             self.H = self.W = None
        
#         # read images
#         frames = transform["frames"]
#         #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
#         # for colmap, manually interpolate a test set.
#         if self.mode == 'colmap' and type == 'test':
            
#             # choose two random poses, and interpolate between.
#             f0, f1 = np.random.choice(frames, 2, replace=False)
#             pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
#             pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
#             rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
#             slerp = Slerp([0, 1], rots)

#             self.poses = []
#             self.images = None
#             for i in range(n_test + 1):
#                 ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
#                 pose = np.eye(4, dtype=np.float32)
#                 pose[:3, :3] = slerp(ratio).as_matrix()
#                 pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
#                 self.poses.append(pose)

#         else:
#             # for colmap, manually split a valid set (the first frame).
#             if self.mode == 'colmap':
#                 if type == 'train':
#                     frames = frames[1:]
#                 elif type == 'val':
#                     frames = frames[:1]
#                 # else 'all' or 'trainval' : use all frames
            
#             self.poses = []
#             self.images = []
#             for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
#                 f_path = os.path.join(self.root_path, f['file_path'])
#                 if self.mode == 'blender' and '.' not in os.path.basename(f_path):
#                     f_path += '.png' # so silly...

#                 # there are non-exist paths in fox...
#                 if not os.path.exists(f_path):
#                     continue
                
#                 pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
#                 pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

#                 image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
#                 if self.H is None or self.W is None:
#                     self.H = int(image.shape[0] // downscale)
#                     self.W = int(image.shape[1] // downscale)

#                 # add support for the alpha channel as a mask.
#                 if image.shape[-1] == 3: 
#                     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 else:
#                     image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

#                 if image.shape[0] != self.H or image.shape[1] != self.W:
#                     image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
#                 image = image.astype(np.float32) / 255 # [H, W, 3/4]

#                 self.poses.append(pose)
#                 self.images.append(image)
            
#         self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
#         if self.images is not None:
#             self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
#         # calculate mean radius of all camera poses
#         self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
#         #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

#         # initialize error_map
#         if self.training and self.opt.error_map:
#             self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
#         else:
#             self.error_map = None

#         # [debug] uncomment to view all training poses.
#         # visualize_poses(self.poses.numpy())

#         # [debug] uncomment to view examples of randomly generated poses.
#         # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

#         if self.preload:
#             self.poses = self.poses.to(self.device)
#             if self.images is not None:
#                 # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
#                 if self.fp16 and self.opt.color_space != 'linear':
#                     dtype = torch.half
#                 else:
#                     dtype = torch.float
#                 self.images = self.images.to(dtype).to(self.device)
#             if self.error_map is not None:
#                 self.error_map = self.error_map.to(self.device)

#         # load intrinsics
#         if 'fl_x' in transform or 'fl_y' in transform:
#             fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
#             fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
#         elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
#             # blender, assert in radians. already downscaled since we use H/W
#             fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
#             fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
#             if fl_x is None: fl_x = fl_y
#             if fl_y is None: fl_y = fl_x
#         else:
#             raise RuntimeError('Failed to load focal length, please check the transforms.json!')

#         cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
#         cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
#         self.intrinsics = np.array([fl_x, fl_y, cx, cy])
#         if keyposes_dir is None:
#             poses = rand_poses(1, self.device, radius=self.radius, theta_range=[np.pi/3, np.pi/2])
#             self.poses = poses
#             if os.path.exists(keyposes_save_dir):
#                 raise ValueError("Key poses file exists!")
#             np.save(keyposes_save_dir, poses.cpu().numpy())
#         else:
#             self.poses = torch.from_numpy(np.load(keyposes_dir)).to(self.device)
#         self.rays = get_rays(self.poses, self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
#         with torch.no_grad():
#             with torch.cuda.amp.autocast(enabled=self.fp16):
#                 outputs = self.pretrained_model.render(self.rays['rays_o'], self.rays['rays_d'], None, staged=False, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
#                 images = outputs['image'].reshape(1, self.H, self.W, 3)
#         if self.training:
#             C = images.shape[-1]
#             images = torch.gather(images.view(1, -1, C), 1, torch.stack(C * [self.rays['inds']], -1)) # [B, N, 3/4]
#         self.images = images
#     def collate(self, index):

#         B = len(index) # a list of length 1

#         results = {
#             'H': self.H,
#             'W': self.W,
#             'rays_o': self.rays['rays_o'],
#             'rays_d': self.rays['rays_d'],
#         }

#         results['images'] = self.images
#         # if self.images is not None:
#         #     images = self.images[index].to(self.device) # [B, H, W, 3/4]
#         #     if self.training:
#         #         C = images.shape[-1]
#         #         images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
#         #     results['images'] = images
        
#         # need inds to update error_map
#         # if error_map is not None:
#         #     results['index'] = index
#         #     results['inds_coarse'] = self.rays['inds_coarse']
            
#         return results

#     def dataloader(self):
#         # size = len(self.poses)
#         # if self.training and self.rand_pose > 0:
#         #     size += size // self.rand_pose # index >= size means we use random pose.
#         size = self.n_test
#         loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
#         loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
#         loader.has_gt = self.images is not None
#         return loader
    

class NeRFDataset_multiview:
    def __init__(self, opt, keyposes_dir, keyposes_save_dir, device, downscale=1, type='train', n_views=8, n_test=48, pretrained_model=None):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose
        self.pretrained_model = pretrained_model.to(device)
        self.n_test = n_test
        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.poses = []
            self.images = []
            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = int(image.shape[0] // downscale)
                    self.W = int(image.shape[1] // downscale)

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.poses.append(pose)
                self.images.append(image)
            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.n_views = n_views
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        if keyposes_dir is None:
            poses = rand_poses(n_views, self.device, radius=self.radius)
            self.poses = poses
            if os.path.exists(keyposes_save_dir):
                raise ValueError("Key poses file exists!")
            np.save(keyposes_save_dir, poses.cpu().numpy())
        else:
            self.poses = torch.from_numpy(np.load(keyposes_dir)).to(self.device)
        self.images = []
        self.rays = []
        for pose in self.poses:
            rays = get_rays(pose.unsqueeze(0), self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.pretrained_model.render(rays['rays_o'], rays['rays_d'], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                    images = outputs['image'].reshape(1, self.H, self.W, 3)
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(1, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            self.images.append(images)
            self.rays.append(rays)
    def collate(self, index):
        B = len(index)
        i = index[0]
        i = i % self.n_views
            # a list of length 1

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': self.rays[i]['rays_o'],
            'rays_d': self.rays[i]['rays_d'],
            'images': self.images[i],
            'i_view': i,
        }
        # if self.images is not None:
        #     images = self.images[index].to(self.device) # [B, H, W, 3/4]
        #     if self.training:
        #         C = images.shape[-1]
        #         images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
        #     results['images'] = images
        
        # need inds to update error_map
        # if error_map is not None:
        #     results['index'] = index
        #     results['inds_coarse'] = self.rays['inds_coarse']
            
        return results

    def dataloader(self):
        # size = len(self.poses)
        # if self.training and self.rand_pose > 0:
        #     size += size // self.rand_pose # index >= size means we use random pose.
        size = self.n_test
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader
    
class NeRFDataset_multiview_gather:
    def __init__(self, opt, keyposes_dir, keyposes_save_dir, device, downscale=1, type='train', n_views=8, n_test=48, pretrained_model=None):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose
        self.pretrained_model = pretrained_model.to(device)
        self.n_test = n_test
        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.poses = []
            self.images = []
            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = int(image.shape[0] // downscale)
                    self.W = int(image.shape[1] // downscale)

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.poses.append(pose)
                self.images.append(image)
            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.n_views = n_views
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        if keyposes_dir is None:
            # poses = rand_poses(n_views, self.device, radius=self.radius, theta_range=[np.pi/3, np.pi/2])
            poses = rand_poses(n_views, self.device, radius=self.radius)
            self.poses = poses
            if os.path.exists(keyposes_save_dir):
                raise ValueError("Key poses file exists!")
            np.save(keyposes_save_dir, poses.cpu().numpy())
        else:
            self.poses = torch.from_numpy(np.load(keyposes_dir)).to(self.device)
        self.images = []
        self.rays = []
        for pose in self.poses:
            rays = get_rays(pose.unsqueeze(0), self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.pretrained_model.render(rays['rays_o'], rays['rays_d'], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                    images = outputs['image'].detach().reshape(1, self.H, self.W, 3)
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(1, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            self.images.append(images)
            self.rays.append(rays)
        self.images = torch.cat(self.images, dim=0)
        self.images = self.images
    def collate(self, index):
        # B = len(index)
        # i = index[0]
        # i = i % self.n_views
            # a list of length 1

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': [self.rays[i]['rays_o'] for i in range(self.n_views)],
            'rays_d': [self.rays[i]['rays_d'] for i in range(self.n_views)],
            'images': self.images,
        }
        # if self.images is not None:
        #     images = self.images[index].to(self.device) # [B, H, W, 3/4]
        #     if self.training:
        #         C = images.shape[-1]
        #         images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
        #     results['images'] = images
        
        # need inds to update error_map
        # if error_map is not None:
        #     results['index'] = index
        #     results['inds_coarse'] = self.rays['inds_coarse']
            
        return results

    def dataloader(self):
        # size = len(self.poses)
        # if self.training and self.rand_pose > 0:
        #     size += size // self.rand_pose # index >= size means we use random pose.
        size = self.n_test
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader

def compute_average_gray(image):
    # 将图像转换为灰度图像
    gray_image = torch.mean(image, dim=3)
    # 求取图像的平均灰度值
    average_gray = torch.mean(gray_image, dim=(3, 4))
    return average_gray

def calculate_compression_ratio(image_tensor):
    # 将图像转换为PIL图像
#     print(image_tensor.shape)
    N, w1, h1, w2, h2, c = image_tensor.shape
    image_tensor = rearrange(image_tensor, 'N w1 h1 c w2 h2 -> (N w1 h1) c w2 h2')
    to_pil = transforms.ToPILImage()
    image_pils = [to_pil(image) for image in image_tensor]
    
    compression_ratio = []
    
    # 计算每张图像的原始大小和压缩大小
    for image_pil in image_pils:
        # 保存原始图像到内存中
        original_image_buffer = BytesIO()
        image_pil.save(original_image_buffer, format='JPEG')
        original_size = original_image_buffer.tell()
        
        # 保存压缩后的图像到内存中
        compressed_image_buffer = BytesIO()
        image_pil.save(compressed_image_buffer, format='JPEG', optimize=True, quality=75)  # 根据需要选择压缩参数
        compressed_size = compressed_image_buffer.tell()
        compression_ratio.append(float(original_size) / compressed_size)
    compression_ratio = torch.from_numpy(np.stack(compression_ratio, axis=0)).to(image_tensor.device)
    compression_ratio = rearrange(compression_ratio, '(N w1 h1 v) -> N w1 h1 v', N=1, w1=w1, v=1)
    return compression_ratio

def split_image(image, num_rows, num_cols):
    # 获取图像的尺寸
    _, height, width, _ = image.size()
    # 计算每个小块的高度和宽度
    block_height = height // num_rows
    block_width = width // num_cols
    
    image = image[:, 0:block_height*num_rows, 0:block_width*num_cols]
    # 划分图像为小块
    image_blocks = image.unfold(1, block_height, block_height).unfold(2, block_width, block_width)
    # 计算每个小块的平均灰度
#     import matplotlib.pyplot as plt
#     fig, axs = plt.subplots(16, 16, figsize=(16, 16))
#     for i, ax in enumerate(axs.flat):
#         sub_image = image_blocks[0, i // 16, i % 16, :, :, :].permute(1, 2, 0)
#         ax.imshow(sub_image.cpu().numpy())
#         ax.axis('off')
            
#     plt.tight_layout()
#     plt.show()
#     plt.savefig('./logs/trial_nerf_lego_wtmk_48b_multiviews_disen_multidecoder_bit/plot.png')

#     average_grays = compute_average_gray(image_blocks)
    average_grays = calculate_compression_ratio(image_blocks)
#     print(average_grays)
    return average_grays, block_height, block_width

def select_lowest_grays(average_grays, num_selections):
    # 将平均灰度展平为一维向量
    # flattened_grays = average_grays.flatten()
    # print(flattened_grays)
    # 对平均灰度进行排序
    # average_grays_sorted = torch.sort(average_grays.view(-1))
    sorted_indices = torch.argsort(average_grays.view(-1))
#     print(sorted_indices)
    # 选择灰度最低的48块
    selected_indices = sorted_indices[:num_selections]
    
    return selected_indices

def select_random_grays(num_rows, num_cols, num_selections):
    # 将平均灰度展平为一维向量
    # flattened_grays = average_grays.flatten()
    # print(flattened_grays)
    # 对平均灰度进行排序
    # average_grays_sorted = torch.sort(average_grays.view(-1))
#     sorted_indices = torch.argsort(average_grays.view(-1))
#     print(sorted_indices)
    # 选择灰度最低的48块
    permuted_indices = torch.randperm(num_rows*num_cols)
# Select the first num_selections indices from the permuted indices
    selected_indices = permuted_indices[:num_selections]
    
    return selected_indices

def get_block_coordinates(selected_indices, num_rows, num_cols, block_height, block_width):
    row_indices = torch.div(selected_indices, num_cols, rounding_mode='floor')
    col_indices = selected_indices % num_cols
    block_coordinates = torch.stack((row_indices * block_height, col_indices * block_width, (row_indices + 1) * block_height, (col_indices + 1) * block_width), dim=1)
    return block_coordinates

def process_image(image, num_rows, num_cols, num_selections):
    average_grays, block_height, block_width = split_image(image, num_rows, num_cols)
    # block_size = block_height
    selected_indices = select_lowest_grays(average_grays, num_selections)
    
    block_coordinates = get_block_coordinates(selected_indices, num_rows, num_cols, block_height, block_width)
#     import matplotlib.pyplot as plt
#     fig, axs = plt.subplots(6, 8, figsize=(16, 16))
#     for i, ax in enumerate(axs.flat):
#         x_start, y_start, x_end, y_end = block_coordinates[i][0].item(), block_coordinates[i][1].item(), block_coordinates[i][2].item(), block_coordinates[i][3].item()
#         sub_image = image[0, x_start : x_end, y_start:y_end, :]
#         ax.imshow(sub_image.cpu().numpy())
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
#     plt.savefig('./logs/trial_nerf_lego_wtmk_48b_multiviews_disen_multidecoder_bit/plot_select.png')
    return block_coordinates, block_height, block_width

def process_image_random_patch(height, width, num_rows, num_cols, num_selections):
#     average_grays, block_height, block_width = split_image(image, num_rows, num_cols)
    block_height = height // num_rows
    block_width = width // num_cols
    selected_indices = select_random_grays(num_rows, num_cols, num_selections)
    
    block_coordinates = get_block_coordinates(selected_indices, num_rows, num_cols, block_height, block_width)
#     import matplotlib.pyplot as plt
#     fig, axs = plt.subplots(6, 8, figsize=(16, 16))
#     for i, ax in enumerate(axs.flat):
#         x_start, y_start, x_end, y_end = block_coordinates[i][0].item(), block_coordinates[i][1].item(), block_coordinates[i][2].item(), block_coordinates[i][3].item()
#         sub_image = image[0, x_start : x_end, y_start:y_end, :]
#         ax.imshow(sub_image.cpu().numpy())
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
#     plt.savefig('./logs/trial_nerf_lego_wtmk_48b_multiviews_disen_multidecoder_bit/plot_select.png')
    return block_coordinates, block_height, block_width

def test_poses_from_exist(n_views, frames, poses, device, scale, offset):
        poses_numpy = poses.cpu().numpy()
        poses_list = [poses_numpy[i] for i in range(poses_numpy.shape[0])]
        
        f0, f1 = np.random.choice(frames, 2, replace=False)
        pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=scale, offset=offset) # [4, 4]
        pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=scale, offset=offset) # [4, 4]
        rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
        slerp = Slerp([0, 1], rots)

        test_poses = []
        if n_views >= (len(poses_list) + 2):
            print('+++++++ add existing poses to test ++++++++')
            for i in range(n_views - len(poses_list)):
                ratio = np.sin(((i / (n_views-1)) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                test_poses.append(pose)
            test_poses = test_poses + poses_list
        else:
            for i in range(n_views):
                ratio = np.sin(((i / (n_views-1)) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                test_poses.append(pose)
        return torch.from_numpy(np.stack(test_poses, axis=0)).to(device)

def rand_poses_from_exist(n_views, frames, poses, device, scale, offset):
        poses_numpy = poses.cpu().numpy()
        poses_list = [poses_numpy[i] for i in range(poses_numpy.shape[0])]
        


        test_poses = []
        
        for i in range(n_views):
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=scale, offset=offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=scale, offset=offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)
            ratio = np.sin((np.random.rand() - 0.5) * np.pi) * 0.5 + 0.5
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = slerp(ratio).as_matrix()
            pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
            test_poses.append(pose)
        return torch.from_numpy(np.stack(test_poses, axis=0)).to(device)
    
class NeRFDataset_Disen:
    def __init__(self, opt, keyposes_dir, keyposes_save_dir, keyblocks_dir, keyblocks_save_dir, device, downscale=1, type='train', n_views=8, n_test=48, message_dim=48, pretrained_model=None, num_rows=16, num_cols=16, use_existset=False):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose
        self.pretrained_model = pretrained_model.to(device)
        self.n_test = n_test
        self.num_rows = num_rows
        self.num_cols = num_cols
        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

#         # load nerf-compatible format data.
#         if self.mode == 'colmap':
#             with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
#                 transform = json.load(f)
#         elif self.mode == 'blender':
#             # load all splits (train/valid/test), this is what instant-ngp in fact does...
#             if type == 'all':
#                 transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
#                 transform = None
#                 for transform_path in transform_paths:
#                     with open(transform_path, 'r') as f:
#                         tmp_transform = json.load(f)
#                         if transform is None:
#                             transform = tmp_transform
#                         else:
#                             transform['frames'].extend(tmp_transform['frames'])
#             # load train and val split
#             elif type == 'trainval':
#                 with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
#                     transform = json.load(f)
#                 with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
#                     transform_val = json.load(f)
#                 transform['frames'].extend(transform_val['frames'])
#             # only load one specified split
#             else:
#                 with open(os.path.join(self.root_path, f'transforms_test.json'), 'r') as f:
#                     transform = json.load(f)

#         else:
#             raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
            transform = json.load(f)
        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h'] // downscale)
            self.W = int(transform['w'] // downscale)
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.train_poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.train_poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.train_poses = []
            self.images = []
            for f in frames:
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = int(image.shape[0] // downscale)
                    self.W = int(image.shape[1] // downscale)

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
#                     print(self.W, self.H)
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.train_poses.append(pose)
                self.images.append(image)
            
        self.train_poses = torch.from_numpy(np.stack(self.train_poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.train_poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.train_poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.train_poses = self.train_poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.n_views = n_views
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        if type != 'test_image':
            if keyposes_dir is None:
                if use_existset:
                    poses = self.train_poses[0:1, ...]
                else:
                    # poses = rand_poses(n_views, self.device, radius=self.radius, theta_range=[np.pi/3, np.pi/2])
                    poses = rand_poses(n_views, self.device, radius=self.radius)
                self.poses = poses
                if keyposes_save_dir is not None:
                    if os.path.exists(keyposes_save_dir):
                        raise ValueError("Key poses file exists!")
                    np.save(keyposes_save_dir, poses.cpu().numpy())
            else:
                self.poses = torch.from_numpy(np.load(keyposes_dir)).to(self.device)

            rays = get_rays(self.poses, self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.pretrained_model.render(rays['rays_o'], rays['rays_d'], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                    images = outputs['image'].detach().reshape(1, self.H, self.W, 3)
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(1, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
                images = images.reshape(1, self.H, self.W, 3)
            self.images = images
            rays['rays_o'] = rays['rays_o'].reshape(1, self.H, self.W, 3)
            rays['rays_d'] = rays['rays_d'].reshape(1, self.H, self.W, 3)
            self.rays = rays

            if keyblocks_dir is None:
                block_coordinates, block_height, block_width = process_image(self.images, self.num_rows, self.num_cols, message_dim)
                self.block_coordinates = block_coordinates
                if keyblocks_save_dir is not None:
                    if os.path.exists(keyblocks_save_dir):
                        raise ValueError("Key blocks file exists!")
                    np.save(keyblocks_save_dir, block_coordinates.cpu().numpy())
            else:
                self.block_coordinates = torch.from_numpy(np.load(keyblocks_dir)).to(self.device)
                block_height = self.H // self.num_rows
                block_width = self.W // self.num_cols
            images_block = []
            rays_o_block = []
            rays_d_block = []
            for i in range(message_dim):
                x_start, y_start, x_end, y_end = self.block_coordinates[i][0].item(), self.block_coordinates[i][1].item(), self.block_coordinates[i][2].item(), self.block_coordinates[i][3].item()
                sub_image = images[0, x_start : x_end, y_start:y_end, :]
                images_block.append(sub_image)
                rays_o_i = rays['rays_o'][:, x_start:x_end, y_start:y_end, :] # [B, N, 3]
                rays_d_i = rays['rays_d'][:, x_start:x_end, y_start:y_end, :] # [B, N, 3]
                rays_o_block.append(rays_o_i)
                rays_d_block.append(rays_d_i)
            self.rays_o_block = torch.cat(rays_o_block, dim=0)
            self.rays_d_block = torch.cat(rays_d_block, dim=0)
            self.images_block = torch.stack(images_block)
            self.patch_H = block_height
            self.patch_W = block_width
        
        else:
            if use_existset:
                poses = test_poses_from_exist(n_views, frames, self.train_poses, self.device, self.scale, self.offset)
            else:
                poses = poses_circle(n_views, self.device, self.radius)
            self.poses = poses
            images = []
            rays = get_rays(self.poses, self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    for i in tqdm.tqdm(range(n_views), desc=f'Loading {type} data'):
                        outputs = self.pretrained_model.render(rays['rays_o'][i:i+1], rays['rays_d'][i:i+1], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                        images = images + [outputs['image'].detach().reshape(1, self.H, self.W, 3)]

            self.images = torch.cat(images, dim=0)
#             rays['rays_o'] = rays['rays_o'].reshape(n_views, self.H, self.W, 3)
#             rays['rays_d'] = rays['rays_d'].reshape(n_views, self.H, self.W, 3)
            self.rays = rays
            
            self.block_coordinates = None
            self.patch_H = None
            self.patch_W = None
            self.images_block = None
            self.rays_o_block = None
            self.rays_d_block = None

    def collate(self, index):
        # B = len(index)
        # i = index[0]
        # i = i % self.n_views
            # a list of length 1

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': self.rays['rays_o'],
            'rays_d': self.rays['rays_d'],
            'images': self.images,
            'block_coordinates': self.block_coordinates,
            'patch_H': self.patch_H,
            'patch_W': self.patch_W,
            'images_block': self.images_block,
            'rays_o_block': self.rays_o_block,
            'rays_d_block': self.rays_d_block,
        }
        # if self.images is not None:
        #     images = self.images[index].to(self.device) # [B, H, W, 3/4]
        #     if self.training:
        #         C = images.shape[-1]
        #         images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
        #     results['images'] = images
        
        # need inds to update error_map
        # if error_map is not None:
        #     results['index'] = index
        #     results['inds_coarse'] = self.rays['inds_coarse']
            
        return results

    def collate_image(self, index):
        # B = len(index)
        # i = index[0]
        # i = i % self.n_views
            # a list of length 1
        i = index[0]
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': self.rays['rays_o'][i:i+1],
            'rays_d': self.rays['rays_d'][i:i+1],
            'images': self.images[i:i+1],
            'block_coordinates': self.block_coordinates,
            'patch_H': self.patch_H,
            'patch_W': self.patch_W,
            'images_block': self.images_block,
            'rays_o_block': self.rays_o_block,
            'rays_d_block': self.rays_d_block,
        }
        # if self.images is not None:
        #     images = self.images[index].to(self.device) # [B, H, W, 3/4]
        #     if self.training:
        #         C = images.shape[-1]
        #         images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
        #     results['images'] = images
        
        # need inds to update error_map
        # if error_map is not None:
        #     results['index'] = index
        #     results['inds_coarse'] = self.rays['inds_coarse']
            
        return results
    
    def dataloader(self):
        # size = len(self.poses)
        # if self.training and self.rand_pose > 0:
        #     size += size // self.rand_pose # index >= size means we use random pose.
        if self.type != 'test_image':
            size = self.n_test
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        else:
            size = self.n_views
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate_image, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader
    
    

    
    
    
class NeRFDataset_Copyrnerf:
    def __init__(self, opt, keyposes_dir, keyposes_save_dir, keyblocks_dir, keyblocks_save_dir, device, downscale=1, type='train', n_views=8, n_test=48, message_dim=48, pretrained_model=None, num_rows=16, num_cols=16, use_existset=False):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose
        self.pretrained_model = pretrained_model.to(device)
        self.n_test = n_test
        self.num_rows = num_rows
        self.num_cols = num_cols
        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
            transform = json.load(f)
        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h'] // downscale)
            self.W = int(transform['w'] // downscale)
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.train_poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.train_poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.train_poses = []
            self.images = []
            for f in frames:
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = int(image.shape[0] // downscale)
                    self.W = int(image.shape[1] // downscale)

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
#                     print(self.W, self.H)
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.train_poses.append(pose)
                self.images.append(image)
            
        self.train_poses = torch.from_numpy(np.stack(self.train_poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.train_poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.train_poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.train_poses = self.train_poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.n_views = n_views
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        if type != 'test_image':
            poses = rand_poses(1, self.device, radius=self.radius)
            self.poses = poses
        else:
            if use_existset:
                poses = test_poses_from_exist(n_views, frames, self.train_poses, self.device, self.scale, self.offset)
            else:
                poses = poses_circle(n_views, self.device, self.radius)
            self.poses = poses
            images = []
            rays = get_rays(self.poses, self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    for i in tqdm.tqdm(range(n_views), desc=f'Loading {type} data'):
                        outputs = self.pretrained_model.render(rays['rays_o'][i:i+1], rays['rays_d'][i:i+1], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                        images = images + [outputs['image'].detach().reshape(1, self.H, self.W, 3)]

            self.images = torch.cat(images, dim=0)
#             rays['rays_o'] = rays['rays_o'].reshape(n_views, self.H, self.W, 3)
#             rays['rays_d'] = rays['rays_d'].reshape(n_views, self.H, self.W, 3)
            self.rays = rays
        self.use_existset = use_existset
        self.frames = frames
            

    def collate(self, index):
        # B = len(index)
        # i = index[0]
        # i = i % self.n_views
            # a list of length 1
        if self.use_existset:
            poses = rand_poses_from_exist(1, self.frames, self.train_poses, self.device, self.scale, self.offset)
        else:
            poses = rand_poses(1, self.device, radius=self.radius)
        self.poses = poses
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                outputs = self.pretrained_model.render(rays['rays_o'], rays['rays_d'], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                images = outputs['image'].detach().reshape(1, self.H, self.W, 3)
        if self.training:
            C = images.shape[-1]
            images = torch.gather(images.view(1, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            images = images.reshape(1, self.H, self.W, 3)
        
        rays['rays_o'] = rays['rays_o'].reshape(1, self.H, self.W, 3)
        rays['rays_d'] = rays['rays_d'].reshape(1, self.H, self.W, 3)
        
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'images': images,
        }
            
        return results

    def collate_image(self, index):
        # B = len(index)
        # i = index[0]
        # i = i % self.n_views
            # a list of length 1
        i = index[0]
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': self.rays['rays_o'][i:i+1],
            'rays_d': self.rays['rays_d'][i:i+1],
            'images': self.images[i:i+1],
        }
            
        return results
    
    def dataloader(self):
        if self.type != 'test_image':
            size = self.n_test
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        else:
            size = self.n_views
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate_image, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader

    

class NeRFDataset_Finetune:
    def __init__(self, opt, keyposes_dir, keyposes_save_dir, keyblocks_dir, keyblocks_save_dir, device, downscale=1, type='train', n_views=8, n_test=48, message_dim=48, pretrained_model=None, num_rows=16, num_cols=16, use_existset=False):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose
        self.pretrained_model = pretrained_model.to(device)
        self.n_test = n_test
        self.num_rows = num_rows
        self.num_cols = num_cols
        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
            transform = json.load(f)
        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h'] // downscale)
            self.W = int(transform['w'] // downscale)
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.train_poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.train_poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.train_poses = []
            self.images = []
            for f in frames:
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = int(image.shape[0] // downscale)
                    self.W = int(image.shape[1] // downscale)

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
#                     print(self.W, self.H)
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.train_poses.append(pose)
                self.images.append(image)
            
        self.train_poses = torch.from_numpy(np.stack(self.train_poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.train_poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.train_poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.train_poses = self.train_poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.n_views = n_views
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        if type != 'test_image':
            if use_existset:
                poses = rand_poses_from_exist(self.n_test, frames, self.train_poses, self.device, self.scale, self.offset)
            else:
                poses = rand_poses(self.n_test, self.device, radius=self.radius)
            self.poses = poses
            images = []
            rays = get_rays(self.poses, self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    for i in tqdm.tqdm(range(self.n_test), desc=f'Loading {type} data'):
                        outputs = self.pretrained_model.render(rays['rays_o'][i:i+1], rays['rays_d'][i:i+1], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                        images = images + [outputs['image'].detach().reshape(1, self.H, self.W, 3)]

            self.images = torch.cat(images, dim=0)
#             rays['rays_o'] = rays['rays_o'].reshape(n_views, self.H, self.W, 3)
#             rays['rays_d'] = rays['rays_d'].reshape(n_views, self.H, self.W, 3)
            rays['rays_o'] = rays['rays_o'].reshape(self.n_test, self.H, self.W, 3)
            rays['rays_d'] = rays['rays_d'].reshape(self.n_test, self.H, self.W, 3)
            self.rays = rays
        else:
            if use_existset:
                poses = test_poses_from_exist(n_views, frames, self.train_poses, self.device, self.scale, self.offset)
            else:
                poses = poses_circle(n_views, self.device, self.radius)
            self.poses = poses
            images = []
            rays = get_rays(self.poses, self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    for i in tqdm.tqdm(range(n_views), desc=f'Loading {type} data'):
                        outputs = self.pretrained_model.render(rays['rays_o'][i:i+1], rays['rays_d'][i:i+1], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                        images = images + [outputs['image'].detach().reshape(1, self.H, self.W, 3)]

            self.images = torch.cat(images, dim=0)
#             rays['rays_o'] = rays['rays_o'].reshape(n_views, self.H, self.W, 3)
#             rays['rays_d'] = rays['rays_d'].reshape(n_views, self.H, self.W, 3)
            self.rays = rays
            

    def collate(self, index):
        i = index[0]
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': self.rays['rays_o'][i:i+1],
            'rays_d': self.rays['rays_d'][i:i+1],
            'images': self.images[i:i+1],
        }
            
        return results

    def collate_image(self, index):
        # B = len(index)
        # i = index[0]
        # i = i % self.n_views
            # a list of length 1
        i = index[0]
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': self.rays['rays_o'][i:i+1],
            'rays_d': self.rays['rays_d'][i:i+1],
            'images': self.images[i:i+1],
        }
            
        return results
    
    def dataloader(self):
        if self.type != 'test_image':
            size = self.n_test
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        else:
            size = self.n_views
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate_image, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader


class NeRFDataset_Prewatermarking:
    def __init__(self, opt, device, downscale=1, type='train', n_views=8, n_test=48, pretrained_model=None, use_existset=False):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose
        self.pretrained_model = pretrained_model.to(device)
        self.n_test = n_test
        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
            transform = json.load(f)
        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h'] // downscale)
            self.W = int(transform['w'] // downscale)
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.train_poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.train_poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.train_poses = []
            self.images = []
            for f in frames:
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = int(image.shape[0] // downscale)
                    self.W = int(image.shape[1] // downscale)

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
#                     print(self.W, self.H)
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.train_poses.append(pose)
                self.images.append(image)
            
        self.train_poses = torch.from_numpy(np.stack(self.train_poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.train_poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.train_poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.train_poses = self.train_poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.n_views = n_views
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        if type != 'test_image':
            if use_existset:
                poses = rand_poses_from_exist(self.n_test, frames, self.train_poses, self.device, self.scale, self.offset)
            else:
                poses = rand_poses(self.n_test, self.device, radius=self.radius)
            self.poses = poses
            images = []
            rays = get_rays(self.poses, self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    for i in tqdm.tqdm(range(self.n_test), desc=f'Loading {type} data'):
                        outputs = self.pretrained_model.render(rays['rays_o'][i:i+1], rays['rays_d'][i:i+1], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                        images = images + [outputs['image'].detach().reshape(1, self.H, self.W, 3)]

            self.images = torch.cat(images, dim=0)
#             rays['rays_o'] = rays['rays_o'].reshape(n_views, self.H, self.W, 3)
#             rays['rays_d'] = rays['rays_d'].reshape(n_views, self.H, self.W, 3)
            rays['rays_o'] = rays['rays_o'].reshape(self.n_test, self.H, self.W, 3)
            rays['rays_d'] = rays['rays_d'].reshape(self.n_test, self.H, self.W, 3)
            self.rays = rays
        else:
            if use_existset:
                poses = test_poses_from_exist(n_views, frames, self.train_poses, self.device, self.scale, self.offset)
            else:
                poses = poses_circle(n_views, self.device, self.radius)
            self.poses = poses
            images = []
            rays = get_rays(self.poses, self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    for i in tqdm.tqdm(range(n_views), desc=f'Loading {type} data'):
                        outputs = self.pretrained_model.render(rays['rays_o'][i:i+1], rays['rays_d'][i:i+1], None, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                        images = images + [outputs['image'].detach().reshape(1, self.H, self.W, 3)]

            self.images = torch.cat(images, dim=0)
#             rays['rays_o'] = rays['rays_o'].reshape(n_views, self.H, self.W, 3)
#             rays['rays_d'] = rays['rays_d'].reshape(n_views, self.H, self.W, 3)
            self.rays = rays
            

    def collate(self, index):
        i = index[0]
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': self.rays['rays_o'][i:i+1],
            'rays_d': self.rays['rays_d'][i:i+1],
            'images': self.images[i:i+1],
        }
            
        return results

    def collate_image(self, index):
        # B = len(index)
        # i = index[0]
        # i = i % self.n_views
            # a list of length 1
        i = index[0]
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': self.rays['rays_o'][i:i+1],
            'rays_d': self.rays['rays_d'][i:i+1],
            'images': self.images[i:i+1],
        }
            
        return results
    
    def dataloader(self):
        if self.type != 'test_image':
            size = self.n_test
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        else:
            size = self.n_views
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate_image, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader

    
class NeRFDataset_Disen_finetuning_attack:
    def __init__(self, opt, keyposes_dir, keyposes_save_dir, keyblocks_dir, keyblocks_save_dir, device, downscale=1, type='train', n_views=8, n_test=48, message_dim=48, pretrained_model=None, num_rows=16, num_cols=16, use_existset=False):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose
        self.pretrained_model = pretrained_model.to(device)
        self.n_test = n_test
        self.num_rows = num_rows
        self.num_cols = num_cols
        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')
        self.message = torch.randint(1, 2, (message_dim,), dtype=torch.float32, device=self.device)
#         # load nerf-compatible format data.
#         if self.mode == 'colmap':
#             with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
#                 transform = json.load(f)
#         elif self.mode == 'blender':
#             # load all splits (train/valid/test), this is what instant-ngp in fact does...
#             if type == 'all':
#                 transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
#                 transform = None
#                 for transform_path in transform_paths:
#                     with open(transform_path, 'r') as f:
#                         tmp_transform = json.load(f)
#                         if transform is None:
#                             transform = tmp_transform
#                         else:
#                             transform['frames'].extend(tmp_transform['frames'])
#             # load train and val split
#             elif type == 'trainval':
#                 with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
#                     transform = json.load(f)
#                 with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
#                     transform_val = json.load(f)
#                 transform['frames'].extend(transform_val['frames'])
#             # only load one specified split
#             else:
#                 with open(os.path.join(self.root_path, f'transforms_test.json'), 'r') as f:
#                     transform = json.load(f)

#         else:
#             raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
            transform = json.load(f)
        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h'] // downscale)
            self.W = int(transform['w'] // downscale)
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.train_poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.train_poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.train_poses = []
            self.images = []
            for f in frames:
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = int(image.shape[0] // downscale)
                    self.W = int(image.shape[1] // downscale)

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
#                     print(self.W, self.H)
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.train_poses.append(pose)
                self.images.append(image)
            
        self.train_poses = torch.from_numpy(np.stack(self.train_poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.train_poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.train_poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.train_poses = self.train_poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.n_views = n_views
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        if type != 'test_image':
            if keyposes_dir is None:
                if use_existset:
                    poses = self.train_poses[0:1, ...]
                else:
                    # poses = rand_poses(n_views, self.device, radius=self.radius, theta_range=[np.pi/3, np.pi/2])
                    poses = rand_poses(n_views, self.device, radius=self.radius)
                self.poses = poses
#                 if os.path.exists(keyposes_save_dir):
#                     raise ValueError("Key poses file exists!")
#                 np.save(keyposes_save_dir, poses.cpu().numpy())
            else:
                self.poses = torch.from_numpy(np.load(keyposes_dir)).to(self.device)

            rays = get_rays(self.poses, self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.pretrained_model.render(rays['rays_o'], rays['rays_d'], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                    images = outputs['image'].detach().reshape(n_views, self.H, self.W, 3)
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(n_views, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
                images = images.reshape(n_views, self.H, self.W, 3)
            self.images = images
            rays['rays_o'] = rays['rays_o'].reshape(n_views, self.H, self.W, 3)
            rays['rays_d'] = rays['rays_d'].reshape(n_views, self.H, self.W, 3)
            self.rays = rays


            self.block_coordinates = None
            self.patch_H = None
            self.patch_W = None
            self.images_block = None
            self.rays_o_block = None
            self.rays_d_block = None
        
        else:
            if use_existset:
                poses = test_poses_from_exist(n_views, frames, self.train_poses, self.device, self.scale, self.offset)
            else:
                poses = poses_circle(n_views, self.device, self.radius)
            self.poses = poses
            images = []
            rays = get_rays(self.poses, self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    for i in tqdm.tqdm(range(n_views), desc=f'Loading {type} data'):
                        outputs = self.pretrained_model.render(rays['rays_o'][i:i+1], rays['rays_d'][i:i+1], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                        images = images + [outputs['image'].detach().reshape(n_views, self.H, self.W, 3)]

            self.images = torch.cat(images, dim=0)
#             rays['rays_o'] = rays['rays_o'].reshape(n_views, self.H, self.W, 3)
#             rays['rays_d'] = rays['rays_d'].reshape(n_views, self.H, self.W, 3)
            self.rays = rays
            
            self.block_coordinates = None
            self.patch_H = None
            self.patch_W = None
            self.images_block = None
            self.rays_o_block = None
            self.rays_d_block = None

    def collate(self, index):
        # B = len(index)
        # i = index[0]
        # i = i % self.n_views
            # a list of length 1
        i = index[0]
        print('-----------index-----------', i)
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': self.rays['rays_o'][i:i+1],
            'rays_d': self.rays['rays_d'][i:i+1],
            'images': self.images[i:i+1],
            'block_coordinates': self.block_coordinates,
            'patch_H': self.patch_H,
            'patch_W': self.patch_W,
            'images_block': self.images_block,
            'rays_o_block': self.rays_o_block,
            'rays_d_block': self.rays_d_block,
        }
        # if self.images is not None:
        #     images = self.images[index].to(self.device) # [B, H, W, 3/4]
        #     if self.training:
        #         C = images.shape[-1]
        #         images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
        #     results['images'] = images
        
        # need inds to update error_map
        # if error_map is not None:
        #     results['index'] = index
        #     results['inds_coarse'] = self.rays['inds_coarse']
            
        return results

    def collate_image(self, index):
        # B = len(index)
        # i = index[0]
        # i = i % self.n_views
            # a list of length 1
        i = index[0]
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': self.rays['rays_o'][i:i+1],
            'rays_d': self.rays['rays_d'][i:i+1],
            'images': self.images[i:i+1],
            'block_coordinates': self.block_coordinates,
            'patch_H': self.patch_H,
            'patch_W': self.patch_W,
            'images_block': self.images_block,
            'rays_o_block': self.rays_o_block,
            'rays_d_block': self.rays_d_block,
        }
        # if self.images is not None:
        #     images = self.images[index].to(self.device) # [B, H, W, 3/4]
        #     if self.training:
        #         C = images.shape[-1]
        #         images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
        #     results['images'] = images
        
        # need inds to update error_map
        # if error_map is not None:
        #     results['index'] = index
        #     results['inds_coarse'] = self.rays['inds_coarse']
            
        return results
    
    def dataloader(self):
        # size = len(self.poses)
        # if self.training and self.rand_pose > 0:
        #     size += size // self.rand_pose # index >= size means we use random pose.
        if self.type != 'test_image':
            size = self.n_views
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        else:
            size = self.n_views
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate_image, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader
    

    
class NeRFDataset_fixview:
    def __init__(self, opt, keyposes_dir, keyposes_save_dir, keyblocks_dir, keyblocks_save_dir, device, downscale=1, type='train', n_views=8, n_test=48, message_dim=48, pretrained_model=None, num_rows=16, num_cols=16, use_existset=False):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose
        self.pretrained_model = pretrained_model.to(device)
        self.n_test = n_test
        self.num_rows = num_rows
        self.num_cols = num_cols
        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
            transform = json.load(f)
        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h'] // downscale)
            self.W = int(transform['w'] // downscale)
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.train_poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.train_poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.train_poses = []
            self.images = []
            for f in frames:
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = int(image.shape[0] // downscale)
                    self.W = int(image.shape[1] // downscale)

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
#                     print(self.W, self.H)
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.train_poses.append(pose)
                self.images.append(image)
            
        self.train_poses = torch.from_numpy(np.stack(self.train_poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.train_poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.train_poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.train_poses = self.train_poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.n_views = n_views
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        if type != 'test_image':
            if use_existset:
                if use_existset:
                    poses = rand_poses_from_exist(1, frames, self.train_poses, self.device, self.scale, self.offset)
                else:
                    poses = rand_poses(1, self.device, radius=self.radius)
                self.poses = poses
                if keyposes_save_dir is not None:
                    if os.path.exists(keyposes_save_dir):
                        raise ValueError("Key poses file exists!")
                    np.save(keyposes_save_dir, poses.cpu().numpy())
            else:
                self.poses = torch.from_numpy(np.load(keyposes_dir)).to(self.device)
            rays = get_rays(self.poses, self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.pretrained_model.render(rays['rays_o'], rays['rays_d'], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                    images = outputs['image'].detach().reshape(1, self.H, self.W, 3)
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(1, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
                images = images.reshape(1, self.H, self.W, 3)

            rays['rays_o'] = rays['rays_o'].reshape(1, self.H, self.W, 3)
            rays['rays_d'] = rays['rays_d'].reshape(1, self.H, self.W, 3)
            
            self.rays = rays
            self.images = images
            
        else:
            if use_existset:
                poses = test_poses_from_exist(n_views, frames, self.train_poses, self.device, self.scale, self.offset)
            else:
                poses = poses_circle(n_views, self.device, self.radius)
            self.poses = poses
            images = []
            rays = get_rays(self.poses, self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    for i in tqdm.tqdm(range(n_views), desc=f'Loading {type} data'):
                        outputs = self.pretrained_model.render(rays['rays_o'][i:i+1], rays['rays_d'][i:i+1], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                        images = images + [outputs['image'].detach().reshape(1, self.H, self.W, 3)]

            self.images = torch.cat(images, dim=0)
#             rays['rays_o'] = rays['rays_o'].reshape(n_views, self.H, self.W, 3)
#             rays['rays_d'] = rays['rays_d'].reshape(n_views, self.H, self.W, 3)
            self.rays = rays
#         self.use_existset = use_existset
#         self.frames = frames
            

    def collate(self, index):
        # B = len(index)
        # i = index[0]
        # i = i % self.n_views
            # a list of length 1
#         if self.use_existset:
#             poses = rand_poses_from_exist(1, self.frames, self.train_poses, self.device, self.scale, self.offset)
#         else:
#             poses = rand_poses(1, self.device, radius=self.radius)
#         self.poses = poses
#         rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
#         with torch.no_grad():
#             with torch.cuda.amp.autocast(enabled=self.fp16):
#                 outputs = self.pretrained_model.render(rays['rays_o'], rays['rays_d'], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
#                 images = outputs['image'].detach().reshape(1, self.H, self.W, 3)
#         if self.training:
#             C = images.shape[-1]
#             images = torch.gather(images.view(1, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
#             images = images.reshape(1, self.H, self.W, 3)
        
#         rays['rays_o'] = rays['rays_o'].reshape(1, self.H, self.W, 3)
#         rays['rays_d'] = rays['rays_d'].reshape(1, self.H, self.W, 3)
        
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': self.rays['rays_o'],
            'rays_d': self.rays['rays_d'],
            'images': self.images,
        }
            
        return results

    def collate_image(self, index):
        # B = len(index)
        # i = index[0]
        # i = i % self.n_views
            # a list of length 1
        i = index[0]
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': self.rays['rays_o'][i:i+1],
            'rays_d': self.rays['rays_d'][i:i+1],
            'images': self.images[i:i+1],
        }
            
        return results
    
    def dataloader(self):
        if self.type != 'test_image':
            size = self.n_test
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        else:
            size = self.n_views
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate_image, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader

    
    
class NeRFDataset_random_patch:
    def __init__(self, opt, keyposes_dir, keyposes_save_dir, keyblocks_dir, keyblocks_save_dir, device, downscale=1, type='train', n_views=8, n_test=48, message_dim=48, pretrained_model=None, num_rows=16, num_cols=16, use_existset=False):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose
        self.pretrained_model = pretrained_model.to(device)
        self.n_test = n_test
        self.num_rows = num_rows
        self.num_cols = num_cols
        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

#         # load nerf-compatible format data.
#         if self.mode == 'colmap':
#             with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
#                 transform = json.load(f)
#         elif self.mode == 'blender':
#             # load all splits (train/valid/test), this is what instant-ngp in fact does...
#             if type == 'all':
#                 transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
#                 transform = None
#                 for transform_path in transform_paths:
#                     with open(transform_path, 'r') as f:
#                         tmp_transform = json.load(f)
#                         if transform is None:
#                             transform = tmp_transform
#                         else:
#                             transform['frames'].extend(tmp_transform['frames'])
#             # load train and val split
#             elif type == 'trainval':
#                 with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
#                     transform = json.load(f)
#                 with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
#                     transform_val = json.load(f)
#                 transform['frames'].extend(transform_val['frames'])
#             # only load one specified split
#             else:
#                 with open(os.path.join(self.root_path, f'transforms_test.json'), 'r') as f:
#                     transform = json.load(f)

#         else:
#             raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
            transform = json.load(f)
        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h'] // downscale)
            self.W = int(transform['w'] // downscale)
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.train_poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.train_poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.train_poses = []
            self.images = []
            for f in frames:
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = int(image.shape[0] // downscale)
                    self.W = int(image.shape[1] // downscale)

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
#                     print(self.W, self.H)
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.train_poses.append(pose)
                self.images.append(image)
            
        self.train_poses = torch.from_numpy(np.stack(self.train_poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.train_poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.train_poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.train_poses = self.train_poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.n_views = n_views
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        if type != 'test_image':
            if keyposes_dir is None:
                if use_existset:
                    poses = self.train_poses[0:1, ...]
                else:
                    # poses = rand_poses(n_views, self.device, radius=self.radius, theta_range=[np.pi/3, np.pi/2])
                    poses = rand_poses(n_views, self.device, radius=self.radius)
                self.poses = poses
                if keyposes_save_dir is not None:
                    if os.path.exists(keyposes_save_dir):
                        raise ValueError("Key poses file exists!")
                    np.save(keyposes_save_dir, poses.cpu().numpy())
            else:
                self.poses = torch.from_numpy(np.load(keyposes_dir)).to(self.device)

            rays = get_rays(self.poses, self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.pretrained_model.render(rays['rays_o'], rays['rays_d'], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                    images = outputs['image'].detach().reshape(1, self.H, self.W, 3)
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(1, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
                images = images.reshape(1, self.H, self.W, 3)
            self.images = images
            rays['rays_o'] = rays['rays_o'].reshape(1, self.H, self.W, 3)
            rays['rays_d'] = rays['rays_d'].reshape(1, self.H, self.W, 3)
            self.rays = rays

            if keyblocks_dir is None:
                block_coordinates, block_height, block_width = process_image_random_patch(self.H, self.W, self.num_rows, self.num_cols, message_dim)
                self.block_coordinates = block_coordinates
                if keyblocks_save_dir is not None:
                    if os.path.exists(keyblocks_save_dir):
                        raise ValueError("Key blocks file exists!")
                    np.save(keyblocks_save_dir, block_coordinates.cpu().numpy())
            else:
                self.block_coordinates = torch.from_numpy(np.load(keyblocks_dir)).to(self.device)
                block_height = self.H // self.num_rows
                block_width = self.W // self.num_cols
            images_block = []
            rays_o_block = []
            rays_d_block = []
            for i in range(message_dim):
                x_start, y_start, x_end, y_end = self.block_coordinates[i][0].item(), self.block_coordinates[i][1].item(), self.block_coordinates[i][2].item(), self.block_coordinates[i][3].item()
                sub_image = images[0, x_start : x_end, y_start:y_end, :]
                images_block.append(sub_image)
                rays_o_i = rays['rays_o'][:, x_start:x_end, y_start:y_end, :] # [B, N, 3]
                rays_d_i = rays['rays_d'][:, x_start:x_end, y_start:y_end, :] # [B, N, 3]
                rays_o_block.append(rays_o_i)
                rays_d_block.append(rays_d_i)
            self.rays_o_block = torch.cat(rays_o_block, dim=0)
            self.rays_d_block = torch.cat(rays_d_block, dim=0)
            self.images_block = torch.stack(images_block)
            self.patch_H = block_height
            self.patch_W = block_width
        
        else:
            if use_existset:
                poses = test_poses_from_exist(n_views, frames, self.train_poses, self.device, self.scale, self.offset)
            else:
                poses = poses_circle(n_views, self.device, self.radius)
            self.poses = poses
            images = []
            rays = get_rays(self.poses, self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    for i in tqdm.tqdm(range(n_views), desc=f'Loading {type} data'):
                        outputs = self.pretrained_model.render(rays['rays_o'][i:i+1], rays['rays_d'][i:i+1], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                        images = images + [outputs['image'].detach().reshape(1, self.H, self.W, 3)]

            self.images = torch.cat(images, dim=0)
#             rays['rays_o'] = rays['rays_o'].reshape(n_views, self.H, self.W, 3)
#             rays['rays_d'] = rays['rays_d'].reshape(n_views, self.H, self.W, 3)
            self.rays = rays
            
            self.block_coordinates = None
            self.patch_H = None
            self.patch_W = None
            self.images_block = None
            self.rays_o_block = None
            self.rays_d_block = None

    def collate(self, index):
        # B = len(index)
        # i = index[0]
        # i = i % self.n_views
            # a list of length 1

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': self.rays['rays_o'],
            'rays_d': self.rays['rays_d'],
            'images': self.images,
            'block_coordinates': self.block_coordinates,
            'patch_H': self.patch_H,
            'patch_W': self.patch_W,
            'images_block': self.images_block,
            'rays_o_block': self.rays_o_block,
            'rays_d_block': self.rays_d_block,
        }
        # if self.images is not None:
        #     images = self.images[index].to(self.device) # [B, H, W, 3/4]
        #     if self.training:
        #         C = images.shape[-1]
        #         images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
        #     results['images'] = images
        
        # need inds to update error_map
        # if error_map is not None:
        #     results['index'] = index
        #     results['inds_coarse'] = self.rays['inds_coarse']
            
        return results

    def collate_image(self, index):
        # B = len(index)
        # i = index[0]
        # i = i % self.n_views
            # a list of length 1
        i = index[0]
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': self.rays['rays_o'][i:i+1],
            'rays_d': self.rays['rays_d'][i:i+1],
            'images': self.images[i:i+1],
            'block_coordinates': self.block_coordinates,
            'patch_H': self.patch_H,
            'patch_W': self.patch_W,
            'images_block': self.images_block,
            'rays_o_block': self.rays_o_block,
            'rays_d_block': self.rays_d_block,
        }
        # if self.images is not None:
        #     images = self.images[index].to(self.device) # [B, H, W, 3/4]
        #     if self.training:
        #         C = images.shape[-1]
        #         images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
        #     results['images'] = images
        
        # need inds to update error_map
        # if error_map is not None:
        #     results['index'] = index
        #     results['inds_coarse'] = self.rays['inds_coarse']
            
        return results
    
    def dataloader(self):
        # size = len(self.poses)
        # if self.training and self.rand_pose > 0:
        #     size += size // self.rand_pose # index >= size means we use random pose.
        if self.type != 'test_image':
            size = self.n_test
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        else:
            size = self.n_views
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate_image, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader


class NeRFDataset_Disen_random_view:
    def __init__(self, opt, keyposes_dir, keyposes_save_dir, keyblocks_dir, keyblocks_save_dir, device, downscale=1, type='train', n_views=8, n_test=48, message_dim=48, pretrained_model=None, num_rows=16, num_cols=16, use_existset=False):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose
        self.pretrained_model = pretrained_model.to(device)
        self.n_test = n_test
        self.num_rows = num_rows
        self.num_cols = num_cols
        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

#         # load nerf-compatible format data.
#         if self.mode == 'colmap':
#             with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
#                 transform = json.load(f)
#         elif self.mode == 'blender':
#             # load all splits (train/valid/test), this is what instant-ngp in fact does...
#             if type == 'all':
#                 transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
#                 transform = None
#                 for transform_path in transform_paths:
#                     with open(transform_path, 'r') as f:
#                         tmp_transform = json.load(f)
#                         if transform is None:
#                             transform = tmp_transform
#                         else:
#                             transform['frames'].extend(tmp_transform['frames'])
#             # load train and val split
#             elif type == 'trainval':
#                 with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
#                     transform = json.load(f)
#                 with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
#                     transform_val = json.load(f)
#                 transform['frames'].extend(transform_val['frames'])
#             # only load one specified split
#             else:
#                 with open(os.path.join(self.root_path, f'transforms_test.json'), 'r') as f:
#                     transform = json.load(f)

#         else:
#             raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
            transform = json.load(f)
        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h'] // downscale)
            self.W = int(transform['w'] // downscale)
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.train_poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.train_poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.train_poses = []
            self.images = []
            for f in frames:
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = int(image.shape[0] // downscale)
                    self.W = int(image.shape[1] // downscale)

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
#                     print(self.W, self.H)
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.train_poses.append(pose)
                self.images.append(image)
            
        self.train_poses = torch.from_numpy(np.stack(self.train_poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.train_poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.train_poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.train_poses = self.train_poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.n_views = n_views
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        self.images = None
        if type != 'test_image':
            if use_existset:
                poses = self.train_poses[0:1, ...]
            else:
                # poses = rand_poses(n_views, self.device, radius=self.radius, theta_range=[np.pi/3, np.pi/2])
                poses = rand_poses(n_views, self.device, radius=self.radius)
            self.poses = poses


            rays = get_rays(self.poses, self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
#             with torch.no_grad():
#                 with torch.cuda.amp.autocast(enabled=self.fp16):
#                     outputs = self.pretrained_model.render(rays['rays_o'], rays['rays_d'], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
#                     images = outputs['image'].detach().reshape(1, self.H, self.W, 3)
#             if self.training:
#                 C = images.shape[-1]
#                 images = torch.gather(images.view(1, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
#                 images = images.reshape(1, self.H, self.W, 3)
#             self.images = images
            rays['rays_o'] = rays['rays_o'].reshape(1, self.H, self.W, 3)
            rays['rays_d'] = rays['rays_d'].reshape(1, self.H, self.W, 3)
            self.rays = rays

            block_coordinates, block_height, block_width = process_image_random_patch(self.H, self.W, self.num_rows, self.num_cols, message_dim)
            self.block_coordinates = block_coordinates

            images_block = []
            rays_o_block = []
            rays_d_block = []
            for i in range(message_dim):
                x_start, y_start, x_end, y_end = self.block_coordinates[i][0].item(), self.block_coordinates[i][1].item(), self.block_coordinates[i][2].item(), self.block_coordinates[i][3].item()
#                 sub_image = images[0, x_start : x_end, y_start:y_end, :]
#                 images_block.append(sub_image)
                rays_o_i = rays['rays_o'][:, x_start:x_end, y_start:y_end, :] # [B, N, 3]
                rays_d_i = rays['rays_d'][:, x_start:x_end, y_start:y_end, :] # [B, N, 3]
                rays_o_block.append(rays_o_i)
                rays_d_block.append(rays_d_i)
            self.rays_o_block = torch.cat(rays_o_block, dim=0)
            self.rays_d_block = torch.cat(rays_d_block, dim=0)
            self.images_block = None
            self.patch_H = block_height
            self.patch_W = block_width
        
        else:
            if use_existset:
                poses = test_poses_from_exist(n_views, frames, self.train_poses, self.device, self.scale, self.offset)
            else:
                poses = poses_circle(n_views, self.device, self.radius)
            self.poses = poses
            images = []
            rays = get_rays(self.poses, self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    for i in tqdm.tqdm(range(n_views), desc=f'Loading {type} data'):
                        outputs = self.pretrained_model.render(rays['rays_o'][i:i+1], rays['rays_d'][i:i+1], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                        images = images + [outputs['image'].detach().reshape(1, self.H, self.W, 3)]

            self.images = torch.cat(images, dim=0)
#             rays['rays_o'] = rays['rays_o'].reshape(n_views, self.H, self.W, 3)
#             rays['rays_d'] = rays['rays_d'].reshape(n_views, self.H, self.W, 3)
            self.rays = rays
            
            self.block_coordinates = None
            self.patch_H = None
            self.patch_W = None
            self.images_block = None
            self.rays_o_block = None
            self.rays_d_block = None

    def collate(self, index):
        # B = len(index)
        # i = index[0]
        # i = i % self.n_views
            # a list of length 1

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': self.rays['rays_o'],
            'rays_d': self.rays['rays_d'],
            'images': self.images,
            'block_coordinates': self.block_coordinates,
            'patch_H': self.patch_H,
            'patch_W': self.patch_W,
            'images_block': self.images_block,
            'rays_o_block': self.rays_o_block,
            'rays_d_block': self.rays_d_block,
        }
        # if self.images is not None:
        #     images = self.images[index].to(self.device) # [B, H, W, 3/4]
        #     if self.training:
        #         C = images.shape[-1]
        #         images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
        #     results['images'] = images
        
        # need inds to update error_map
        # if error_map is not None:
        #     results['index'] = index
        #     results['inds_coarse'] = self.rays['inds_coarse']
            
        return results

    def collate_image(self, index):
        # B = len(index)
        # i = index[0]
        # i = i % self.n_views
            # a list of length 1
        i = index[0]
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': self.rays['rays_o'][i:i+1],
            'rays_d': self.rays['rays_d'][i:i+1],
            'images': self.images[i:i+1],
            'block_coordinates': self.block_coordinates,
            'patch_H': self.patch_H,
            'patch_W': self.patch_W,
            'images_block': self.images_block,
            'rays_o_block': self.rays_o_block,
            'rays_d_block': self.rays_d_block,
        }
        # if self.images is not None:
        #     images = self.images[index].to(self.device) # [B, H, W, 3/4]
        #     if self.training:
        #         C = images.shape[-1]
        #         images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
        #     results['images'] = images
        
        # need inds to update error_map
        # if error_map is not None:
        #     results['index'] = index
        #     results['inds_coarse'] = self.rays['inds_coarse']
            
        return results
    
    def dataloader(self):
        # size = len(self.poses)
        # if self.training and self.rand_pose > 0:
        #     size += size // self.rand_pose # index >= size means we use random pose.
        if self.type != 'test_image':
            size = self.n_test
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        else:
            size = self.n_views
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate_image, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader