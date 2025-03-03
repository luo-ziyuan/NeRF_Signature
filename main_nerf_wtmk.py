import torch
import argparse

from nerf.provider_wtmk import NeRFDataset_Disen
# from nerf.gui import NeRFGUI
from nerf.utils_wtmk_disen import *

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=800000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")
    parser.add_argument('--wtmk_tcnn', action='store_true', help="use TCNN backend with watermarking")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    parser.add_argument('--message_dim', type=int, default=16, help="message_dim")
    parser.add_argument('--downscale', type=float, default=1, help="dataset downscale")
    parser.add_argument('--lambda_w', type=float, default=1.0, help="lambda_w")
    parser.add_argument('--lambda_i', type=float, default=1.0, help="lambda_i")
    parser.add_argument('--loss_w', type=str, default='bce', help="dataset downscale")
    parser.add_argument('--n_views', type=int, default=1, help="n_views")
    parser.add_argument('--num_rows', type=int, default=16, help="num_rows")
    parser.add_argument('--num_cols', type=int, default=16, help="num_cols")
    parser.add_argument('--use_existset', action='store_true', help="use existed set to test")
    parser.add_argument('--eval_interval', type=int, default=10, help="eval_interval")
    parser.add_argument('--save_interval', type=int, default=10, help="save_interval")
    parser.add_argument('--num_images_test', type=int, default=360, help="num_images_test")
    parser.add_argument('--distortion', type=str, default='none', choices=['none', 'noise', 'rotation', 'scaling', 'blurring', 'brightness'], help="distortion layer")
    
    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True

    if opt.wtmk_tcnn:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        from nerf.network_wtmk_tcnn import NeRFNetwork

    print(opt)
    
    seed_everything(opt.seed)

    model = NeRFNetwork(
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        message_dim=opt.message_dim,
        n_views=opt.n_views,
    )
    
    print(model)

    criterion = torch.nn.MSELoss(reduction='none')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

    keyposes_save_dir = os.path.join(opt.workspace, f'key_poses.npy')
    keyblocks_save_dir = os.path.join(opt.workspace, f'key_blocks.npy')

    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    metrics = [PSNRMeter(), LPIPSMeter(device=device), SSIMMeter()]
    metrics_message = [BIT_ACC()]
    trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics,metrics_message=metrics_message, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, save_interval=opt.save_interval, message_dim=opt.message_dim, n_views=opt.n_views)

    if opt.test:
        test_loader_bitacc = NeRFDataset_Disen(opt, keyposes_dir=keyposes_save_dir, keyposes_save_dir=None, keyblocks_dir=keyblocks_save_dir, keyblocks_save_dir=None, device=device, downscale=opt.downscale, type='test', n_views=opt.n_views, message_dim=opt.message_dim, n_test=200, pretrained_model=model, num_rows=opt.num_rows, num_cols=opt.num_cols, use_existset=opt.use_existset).dataloader()
        trainer.test_bitacc(test_loader_bitacc)
        
        test_loader_image = NeRFDataset_Disen(opt, keyposes_dir=keyposes_save_dir, keyposes_save_dir=None, keyblocks_dir=keyblocks_save_dir, keyblocks_save_dir=None, device=device, downscale=opt.downscale, type='test_image', n_views=opt.num_images_test, message_dim=opt.message_dim, n_test=4, pretrained_model=model, num_rows=opt.num_rows, num_cols=opt.num_cols, use_existset=opt.use_existset).dataloader()
        trainer.test_image(test_loader_image, name='test_image_from_randomviews')
        
        test_loader_image_testviews = NeRFDataset_Disen(opt, keyposes_dir=keyposes_save_dir, keyposes_save_dir=None, keyblocks_dir=keyblocks_save_dir, keyblocks_save_dir=None, device=device, downscale=opt.downscale, type='test_image_testviews', n_views=opt.num_images_test, message_dim=opt.message_dim, n_test=4, pretrained_model=model, num_rows=opt.num_rows, num_cols=opt.num_cols, use_existset=opt.use_existset).dataloader()
        trainer.test_image(test_loader_image_testviews, name='test_image_from_testviews')
    else:    
        if os.path.exists(keyposes_save_dir):
            train_loader = NeRFDataset_Disen(opt, keyposes_dir=keyposes_save_dir, keyposes_save_dir=None, keyblocks_dir=keyblocks_save_dir, keyblocks_save_dir=None, device=device, downscale=opt.downscale, type='train', n_views=opt.n_views, message_dim=opt.message_dim, n_test=100, pretrained_model=model, num_rows=opt.num_rows, num_cols=opt.num_cols, use_existset=opt.use_existset).dataloader()
        else:
            train_loader = NeRFDataset_Disen(opt, keyposes_dir=None, keyposes_save_dir=keyposes_save_dir, keyblocks_dir=None, keyblocks_save_dir=keyblocks_save_dir, device=device, downscale=opt.downscale, type='train', n_views=opt.n_views, message_dim=opt.message_dim, n_test=100, pretrained_model=model, num_rows=opt.num_rows, num_cols=opt.num_cols, use_existset=opt.use_existset).dataloader()
        
        
        test_loader_bitacc = NeRFDataset_Disen(opt, keyposes_dir=keyposes_save_dir, keyposes_save_dir=None, keyblocks_dir=keyblocks_save_dir, keyblocks_save_dir=None, device=device, downscale=opt.downscale, type='test', n_views=opt.n_views, message_dim=opt.message_dim, n_test=200, pretrained_model=model, num_rows=opt.num_rows, num_cols=opt.num_cols, use_existset=opt.use_existset).dataloader()
        test_loader_image = NeRFDataset_Disen(opt, keyposes_dir=keyposes_save_dir, keyposes_save_dir=None, keyblocks_dir=keyblocks_save_dir, keyblocks_save_dir=None, device=device, downscale=opt.downscale, type='test_image', n_views=opt.num_images_test, message_dim=opt.message_dim, n_test=4, pretrained_model=model, num_rows=opt.num_rows, num_cols=opt.num_cols, use_existset=opt.use_existset).dataloader()
        
        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        trainer.train(train_loader, test_loader_image, test_loader_bitacc, max_epoch)

        # save sample images
        test_loader_image_testviews = NeRFDataset_Disen(opt, keyposes_dir=keyposes_save_dir, keyposes_save_dir=None, keyblocks_dir=keyblocks_save_dir, keyblocks_save_dir=None, device=device, downscale=opt.downscale, type='test_image_testviews', n_views=opt.num_images_test, message_dim=opt.message_dim, n_test=4, pretrained_model=model, num_rows=opt.num_rows, num_cols=opt.num_cols, use_existset=opt.use_existset).dataloader()
        trainer.test_image(test_loader_image_testviews, name='test_image_from_testviews')