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
    return block_coordinates, block_height, block_width

def process_image_random_patch(height, width, num_rows, num_cols, num_selections):
#     average_grays, block_height, block_width = split_image(image, num_rows, num_cols)
    block_height = height // num_rows
    block_width = width // num_cols
    selected_indices = select_random_grays(num_rows, num_cols, num_selections)
    
    block_coordinates = get_block_coordinates(selected_indices, num_rows, num_cols, block_height, block_width)
    return block_coordinates, block_height, block_width

def test_poses_from_testviews(n_views, frames, poses, device, scale, offset):
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

def test_poses(n_views, frames, device, scale, offset):
        test_poses = []
        for _ in range(n_views):
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=scale, offset=offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=scale, offset=offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)
            ratio = np.random.random()
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = slerp(ratio).as_matrix()
            pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
            test_poses.append(pose)
        return torch.from_numpy(np.stack(test_poses, axis=0)).to(device)

def rand_poses_from_exist(n_views, frames, poses, device, scale, offset):    
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
        self.type = type # train, test_image, test_bitacc
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload
        self.scale = opt.scale
        self.offset = opt.offset
        self.bound = opt.bound
        self.fp16 = opt.fp16

        self.training = self.type in ['train']
        self.num_rays = self.opt.num_rays if self.training else -1
        self.rand_pose = opt.rand_pose
        self.pretrained_model = pretrained_model.to(device)
        self.n_test = n_test
        self.num_rows = num_rows
        self.num_cols = num_cols

        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap'
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender'
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        if type == 'test_image_testviews':
            with open(os.path.join(self.root_path, f'transforms_test.json'), 'r') as f:
                transform = json.load(f)
        else:
            with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                transform = json.load(f)
        
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h'] // downscale)
            self.W = int(transform['w'] // downscale)
        else:
            self.H = self.W = None
        

        self.n_views = n_views
        frames = transform["frames"]
        
        # 读取训练poses
        self.poses = []
        for f in frames:
            f_path = os.path.join(self.root_path, f['file_path'])
            if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                f_path += '.png'

            # if not os.path.exists(f_path):
            #     continue
            
            pose = np.array(f['transform_matrix'], dtype=np.float32)
            pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

            # use the first image to get H and W
            if self.H is None or self.W is None:
                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
                self.H = int(image.shape[0] // downscale)
                self.W = int(image.shape[1] // downscale)

            self.poses.append(pose)
            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0))
        if self.preload:
            self.poses = self.poses.to(self.device)
        
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()

        # 先加载intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)

        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

        if type == 'train':
            rays = get_rays(self.poses, self.intrinsics, self.H, self.W, -1, None, self.opt.patch_size)
            self.train_images = []
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    for i in tqdm.tqdm(range(len(self.poses)), desc=f'Rendering training images'):
                    # for i in tqdm.tqdm(range(2), desc=f'Rendering training images'):
                        outputs = self.pretrained_model.render(rays['rays_o'][i:i+1], rays['rays_d'][i:i+1], None, 
                                                            staged=True, bg_color=None, perturb=False, 
                                                            force_all_rays=True, **vars(self.opt))
                        image = outputs['image'].detach().reshape(1, self.H, self.W, 3)
                        self.train_images.append(image)

            self.train_images = torch.cat(self.train_images, dim=0)

            if self.preload:
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.train_images = self.train_images.to(dtype).to(self.device)

            # initialize error_map
            if self.training and self.opt.error_map:
                self.error_map = torch.ones([self.train_images.shape[0], 128 * 128], dtype=torch.float)
                if self.preload:
                    self.error_map = self.error_map.to(self.device)
            else:
                self.error_map = None
        else:
            self.train_images = None
            self.error_map = None
        # generate watermark data
        if type in ['train', 'test']:
            if keyposes_dir is None:
                if use_existset:
                    watermark_poses = self.poses[0:1, ...]
                else:
                    watermark_poses = rand_poses(n_views, self.device, radius=self.radius)
                self.watermark_poses = watermark_poses
                if keyposes_save_dir is not None:
                    if os.path.exists(keyposes_save_dir):
                        raise ValueError("Key poses file exists!")
                    np.save(keyposes_save_dir, watermark_poses.cpu().numpy())
            else:
                self.watermark_poses = torch.from_numpy(np.load(keyposes_dir)).to(self.device)

            rays = get_rays(self.watermark_poses, self.intrinsics, self.H, self.W, -1, None, self.opt.patch_size)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.pretrained_model.render(rays['rays_o'], rays['rays_d'], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                    watermark_images = outputs['image'].detach().reshape(1, self.H, self.W, 3)
            if self.training:
                C = watermark_images.shape[-1]
                watermark_images = torch.gather(watermark_images.view(1, -1, C), 1, torch.stack(C * [rays['inds']], -1))
                watermark_images = watermark_images.reshape(1, self.H, self.W, 3)
            self.watermark_images = watermark_images
            rays['rays_o'] = rays['rays_o'].reshape(1, self.H, self.W, 3)
            rays['rays_d'] = rays['rays_d'].reshape(1, self.H, self.W, 3)
            self.rays = rays

            if keyblocks_dir is None:
                block_coordinates, block_height, block_width = process_image(self.watermark_images, self.num_rows, self.num_cols, message_dim)
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
                sub_image = watermark_images[0, x_start:x_end, y_start:y_end, :]
                images_block.append(sub_image)
                rays_o_i = rays['rays_o'][:, x_start:x_end, y_start:y_end, :]
                rays_d_i = rays['rays_d'][:, x_start:x_end, y_start:y_end, :]
                rays_o_block.append(rays_o_i)
                rays_d_block.append(rays_d_i)
            self.rays_o_block = torch.cat(rays_o_block, dim=0)
            self.rays_d_block = torch.cat(rays_d_block, dim=0)
            self.images_block = torch.stack(images_block)
            self.patch_H = block_height
            self.patch_W = block_width
        
        else:
            # if use_existset:
            #     watermark_poses = test_poses_from_exist(n_views, frames, self.poses, self.device, self.scale, self.offset)
            # else:
            if type == 'test_image':
                image_poses = test_poses(n_views, frames, self.device, self.scale, self.offset)
            elif type == 'test_image_testviews':
                image_poses = self.poses
            else:
                raise NotImplementedError(f'Wrong type: {type}')
            self.image_poses = image_poses
            content_images = []
            rays = get_rays(self.image_poses, self.intrinsics, self.H, self.W, self.num_rays, None, self.opt.patch_size)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    for i in tqdm.tqdm(range(len(image_poses)), desc=f'Loading {type} data'):
                        outputs = self.pretrained_model.render(rays['rays_o'][i:i+1], rays['rays_d'][i:i+1], None, staged=True, bg_color=None, perturb=False, force_all_rays=True, **vars(self.opt))
                        content_images = content_images + [outputs['image'].detach().reshape(1, self.H, self.W, 3)]

            self.content_images = torch.cat(content_images, dim=0)
            self.rays = rays
            
            self.block_coordinates = None
            self.patch_H = None
            self.patch_W = None
            self.images_block = None
            self.rays_o_block = None
            self.rays_d_block = None

    def collate(self, index):
        B = len(index)
        poses = self.poses[index].to(self.device)
        error_map = None if self.error_map is None else self.error_map[index]
        content_rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map, self.opt.patch_size)
        
        content = {
            'H': self.H,
            'W': self.W,
            'rays_o': content_rays['rays_o'],
            'rays_d': content_rays['rays_d'],
        }

        if self.train_images is not None:
            train_images = self.train_images[index].to(self.device)
            if self.training:
                C = train_images.shape[-1]
                train_images = torch.gather(train_images.view(B, -1, C), 1, torch.stack(C * [content_rays['inds']], -1))
            content['images'] = train_images
            # print('self.num_rays', self.num_rays)

        if error_map is not None:
            content['index'] = index
            content['inds_coarse'] = content_rays['inds_coarse']

        # 处理watermark部分
        watermark = {
            'H': self.H,
            'W': self.W,
            'rays_o': self.rays['rays_o'],
            'rays_d': self.rays['rays_d'],
            'images': self.watermark_images,
            'block_coordinates': self.block_coordinates,
            'patch_H': self.patch_H,
            'patch_W': self.patch_W,
            'images_block': self.images_block,
            'rays_o_block': self.rays_o_block,
            'rays_d_block': self.rays_d_block,
        }

        results = {
            'watermark': watermark,
            'content': content
        }
        
        return results

    def collate_bitacc(self, index):
        watermark = {
            'H': self.H,
            'W': self.W,
            'rays_o': self.rays['rays_o'],
            'rays_d': self.rays['rays_d'],
            'images': self.watermark_images,
            'block_coordinates': self.block_coordinates,
            'patch_H': self.patch_H,
            'patch_W': self.patch_W,
            'images_block': self.images_block,
            'rays_o_block': self.rays_o_block,
            'rays_d_block': self.rays_d_block,
        }   
        results = {
            'watermark': watermark
        }
        return results

    def collate_image(self, index):
        i = index[0]
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': self.rays['rays_o'][i:i+1],
            'rays_d': self.rays['rays_d'][i:i+1],
            'images': self.content_images[i:i+1],
            'block_coordinates': self.block_coordinates,
            'patch_H': self.patch_H,
            'patch_W': self.patch_W,
            'images_block': self.images_block,
            'rays_o_block': self.rays_o_block,
            'rays_d_block': self.rays_d_block,
        }
        return results
    
    def dataloader(self):
        if self.type == 'test_image':
            size = self.n_views
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate_image, shuffle=False, num_workers=0)
        elif self.type == 'test_image_testviews':
            size = len(self.poses)
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate_image, shuffle=False, num_workers=0)
        elif self.type == 'train':
            size = len(self.poses)
            # size = 2
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=True, num_workers=0)
        elif self.type == 'test':
            size = self.n_test
            loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate_bitacc, shuffle=False, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = True
        return loader
