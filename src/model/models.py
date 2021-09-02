"""
Main model implementation
"""
from numpy.core.fromnumeric import repeat
import torch
from .encoder import ImageEncoder
from .transformer import RadianceTransformer, RadianceTransformer2, RadianceTransformer3
from .code import PositionalEncoding
from .model_util import make_encoder, make_mlp
import torch.autograd.profiler as profiler
import torch.nn.functional as F
from util import repeat_interleave, batched_index_select_nd
import os
import os.path as osp
import warnings

from model import transformer


class PixelNeRFNet(torch.nn.Module):
    def __init__(self, conf, stop_encoder_grad=False):
        """
        :param conf PyHocon config subtree 'model'
        """
        super().__init__()
        self.encoder = make_encoder(conf["encoder"])
        self.use_encoder = conf.get_bool("use_encoder", True)  # Image features?

        self.use_xyz = conf.get_bool("use_xyz", False)

        assert self.use_encoder or self.use_xyz  # Must use some feature..

        # Whether to shift z to align in canonical frame.
        # So that all objects, regardless of camera distance to center, will
        # be centered at z=0.
        # Only makes sense in ShapeNet-type setting.
        self.normalize_z = conf.get_bool("normalize_z", True)

        self.stop_encoder_grad = (
            stop_encoder_grad  # Stop ConvNet gradient (freeze weights)
        )
        self.use_code = conf.get_bool("use_code", False)  # Positional encoding
        self.use_code_viewdirs = conf.get_bool(
            "use_code_viewdirs", True
        )  # Positional encoding applies to viewdirs

        # Enable view directions
        self.use_viewdirs = conf.get_bool("use_viewdirs", False)

        d_latent = self.encoder.latent_size if self.use_encoder else 0
        d_in = 3 if self.use_xyz else 1

        if self.use_viewdirs and self.use_code_viewdirs:
            # Apply positional encoding to viewdirs
            d_in += 3
        if self.use_code and d_in > 0:
            # Positional encoding for x,y,z OR view z
            self.code = PositionalEncoding.from_conf(conf["code"], d_in=d_in)
            d_in = self.code.d_out
        if self.use_viewdirs and not self.use_code_viewdirs:
            # Don't apply positional encoding to viewdirs (concat after encoded)
            d_in += 3

        d_out = 1
        self.latent_size = self.encoder.latent_size

        # Note: this is world -> camera, and bottom row is omitted
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)
        self.register_buffer("image_shape", torch.empty(2), persistent=False)

        self.d_in = d_in
        self.d_out = d_out
        self.d_latent = d_latent
        self.register_buffer("focal", torch.empty(1, 2), persistent=False)
        # Principal point
        self.register_buffer("c", torch.empty(1, 2), persistent=False)

        # Pose_ref
        self.register_buffer("poses_ref", torch.empty(1, 3, 4), persistent=False)
        self.num_objs = 0
        self.num_views_per_obj = 1        
        
        # Radiance Transformer
        d_embed = 128
        n_head=4
        d_color=64
        iteration=4
        self.transformer_coarse = RadianceTransformer3(
            n_head=n_head,
            d_input=self.latent_size+d_in,
            d_embed=d_embed,
            d_view=self.code.d_out,
            d_color=d_color,
            iteration=iteration,)
        self.transformer_fine = RadianceTransformer3(
            n_head=n_head,
            d_input=self.latent_size+d_in,
            d_embed=d_embed,
            d_view=self.code.d_out,
            d_color=d_color,
            iteration=iteration,)

    def encode(self, images, poses, focal, z_bounds=None, c=None):
        """
        :param images (NS, 3, H, W)
        NS is number of input (aka source or reference) views
        :param poses (NS, 4, 4)
        :param focal focal length () or (2) or (NS) or (NS, 2) [fx, fy]
        :param z_bounds ignored argument (used in the past)
        :param c principal point None or () or (2) or (NS) or (NS, 2) [cx, cy],
        default is center of image
        """
        self.num_objs = images.size(0)
        if len(images.shape) == 5:
            assert len(poses.shape) == 4
            assert poses.size(1) == images.size(
                1
            )  # Be consistent with NS = num input views
            self.num_views_per_obj = images.size(1)
            images = images.reshape(-1, *images.shape[2:])
            poses = poses.reshape(-1, 4, 4)
        else:
            self.num_views_per_obj = 1

        self.encoder(images)
        rot = poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, poses[:, :3, 3:])  # (B, 3, 1)
        self.poses = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)

        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        # Handle various focal length/principal point formats
        if len(focal.shape) == 0:
            # Scalar: fx = fy = value for all views
            focal = focal[None, None].repeat((1, 2))
        elif len(focal.shape) == 1:
            # Vector f: fx = fy = f_i *for view i*
            # Length should match NS (or 1 for broadcast)
            focal = focal.unsqueeze(-1).repeat((1, 2))
        else:
            focal = focal.clone()
        self.focal = focal.float()
        #self.focal[..., 1] *= -1.0

        if c is None:
            # Default principal point is center of image
            c = (self.image_shape * 0.5).unsqueeze(0)
        elif len(c.shape) == 0:
            # Scalar: cx = cy = value for all views
            c = c[None, None].repeat((1, 2))
        elif len(c.shape) == 1:
            # Vector c: cx = cy = c_i *for view i*
            c = c.unsqueeze(-1).repeat((1, 2))
        self.c = c

    def forward(self, xyz, index_target, coarse=True, viewdirs=None, compute_target=True, far=False):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3)
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :return (SB, B, 4) r g b sigma
        """
        with profiler.record_function("model_inference"):
            SB, B, _ = xyz.shape
            NS = self.num_views_per_obj

            # Transform query points into the camera spaces of the input views            
            if compute_target:
                poses_tgt = batched_index_select_nd(self.poses_ref, index_target)

                viewdirs_tgt = viewdirs.reshape(SB, B, 3, 1)
                viewdirs_tgt = torch.matmul(
                    poses_tgt[:, :, :3, :3], viewdirs_tgt
                )
                viewdirs_tgt = viewdirs_tgt.reshape(-1, 3)
                transformer_query = self.code(viewdirs_tgt)
                transformer_query = transformer_query.reshape(SB*B, 1, -1)

            xyz = repeat_interleave(xyz, NS)  # (SB*NS, B, 3)
            xyz_rot = torch.matmul(self.poses[:, None, :3, :3], xyz.unsqueeze(-1))[
                ..., 0
            ]
            xyz = xyz_rot + self.poses[:, None, :3, 3]

            if self.d_in > 0:
                # * Encode the xyz coordinates
                if self.use_xyz:
                    if self.normalize_z:
                        z_feature = xyz_rot.reshape(-1, 3)  # (SB*B, 3)
                    else:
                        z_feature = xyz.reshape(-1, 3)  # (SB*B, 3)
                else:
                    if self.normalize_z:
                        z_feature = -xyz_rot[..., 2].reshape(-1, 1)  # (SB*B, 1)
                    else:
                        z_feature = -xyz[..., 2].reshape(-1, 1)  # (SB*B, 1)

                if self.use_code and not self.use_code_viewdirs:
                    # Positional encoding (no viewdirs)
                    z_feature = self.code(z_feature)
                if self.use_viewdirs:
                    # * Encode the view directions
                    assert viewdirs is not None
                    # Viewdirs to input view space
                    viewdirs = viewdirs.reshape(SB, B, 3, 1)
                    viewdirs = repeat_interleave(viewdirs, NS)  # (SB*NS, B, 3, 1)
                    viewdirs = torch.matmul(
                        self.poses[:, None, :3, :3], viewdirs
                    )  # (SB*NS, B, 3, 1)
                    viewdirs = viewdirs.reshape(-1, 3)  # (SB*B, 3)
                    z_feature = torch.cat(
                        (z_feature, viewdirs), dim=1
                    )  # (SB*B, 4 or 6)

                if self.use_code and self.use_code_viewdirs:
                    # Positional encoding (with viewdirs)
                    z_feature = self.code(z_feature)

                mlp_input = z_feature

            if self.use_encoder:
                # Grab encoder's latent code.
                uv = xyz[:, :, :2] / xyz[:, :, 2:]  # (SB, B, 2)
                uv *= repeat_interleave(
                    self.focal.unsqueeze(1), NS if self.focal.shape[0] > 1 else 1
                )
                uv += repeat_interleave(
                    self.c.unsqueeze(1), NS if self.c.shape[0] > 1 else 1
                )  # (SB*NS, B, 2)
                latent = self.encoder.index(
                    uv, None, self.image_shape
                )  # (SB * NS, latent, B)
                latent_src = latent
                latent_src = latent_src.transpose(1, 2).reshape(SB, NS, B, self.latent_size)
                latent_src = latent_src.transpose(1, 2).reshape(SB*B, NS, self.latent_size)
                z_feature_src = z_feature.reshape(SB, NS, B, -1)
                z_feature_src = z_feature_src.transpose(1, 2).reshape(SB*B, NS, -1)
                transformer_key = torch.cat((latent_src, z_feature_src), dim=-1)

            # Run Radiance Transformer network.
            if coarse:
                transformer_output_rgb, transformer_output_sigma = self.transformer_coarse(input=transformer_key, view_dir=transformer_query)
            else:
                transformer_output_rgb, transformer_output_sigma = self.transformer_fine(input=transformer_key, view_dir=transformer_query)
            
            rgb = transformer_output_rgb[:, 0].reshape(SB, B, 3)
            sigma = transformer_output_sigma.reshape(SB, B, 1)
            
            rgb = torch.sigmoid(rgb)
            sigma = torch.relu(sigma)

            transformer_key = transformer_key.reshape(SB, B, NS, -1)

        return rgb, sigma, transformer_key

    def forward_ref(self, xyz, viewdirs, index_batch, transformer_key, coarse):
        B = viewdirs.shape[0]
        _, NR, _, _ = self.poses_ref.shape
        
        poses_ref = self.poses_ref[index_batch]#batched_index_select_nd(all_poses ,index_batch.reshape(-1, 1))

        xyz_ref = xyz[:, None, :, None].repeat(1, NR, 1, 1)
        xyz_rot_ref = torch.matmul(poses_ref[:, :, :3, :3], xyz_ref)[..., 0]
        xyz_ref = xyz_rot_ref + poses_ref[:, :, :3, 3]

        uv_ref = xyz_ref[:, :, :2] / xyz_ref[:, :, 2:]
        uv_ref *= self.focal[index_batch].unsqueeze(1).repeat(1, NR, 1)
        uv_ref += self.c[index_batch].unsqueeze(1).repeat(1, NR, 1)

        viewdirs_ref = viewdirs[:, None, :, None].repeat(1, NR, 1, 1)
        viewdirs_ref = torch.matmul(poses_ref[:, :, :3, :3], viewdirs_ref)[..., 0]

        transformer_query = self.code(viewdirs_ref.reshape(-1, 3))
        transformer_query = transformer_query.reshape(B, NR, -1)

        if coarse:
            transformer_output_rgb, _ = self.transformer_coarse(input=transformer_key, view_dir=transformer_query)
        else:
            transformer_output_rgb, _ = self.transformer_fine(input=transformer_key, view_dir=transformer_query)

        rgb_ref = transformer_output_rgb
        rgb_ref = torch.sigmoid(rgb_ref)

        return rgb_ref, uv_ref

    def load_weights(self, args, opt_init=False, strict=True, device=None):
        """
        Helper for loading weights according to argparse arguments.
        Your can put a checkpoint at checkpoints/<exp>/pixel_nerf_init to use as initialization.
        :param opt_init if true, loads from init checkpoint instead of usual even when resuming
        """
        # TODO: make backups
        if opt_init and not args.resume:
            return
        """
        ckpt_name = (
            "pixel_nerf_init" if opt_init or not args.resume else "pixel_nerf_latest"
        )
        """
        ckpt_name = (
            "pixel_nerf_init" if opt_init or not args.resume else "pixel_nerf_{}".format(args.eval_epoch)
        )
        model_path = "%s/%s/%s" % (args.checkpoints_path, args.name, ckpt_name)

        if device is None:
            device = self.poses.device

        if os.path.exists(model_path):
            print("Load", model_path)
            self.load_state_dict(
                torch.load(model_path, map_location=device), strict=strict
            )
        elif not opt_init:
            warnings.warn(
                (
                    "WARNING: {} does not exist, not loaded!! Model will be re-initialized.\n"
                    + "If you are trying to load a pretrained model, STOP since it's not in the right place. "
                    + "If training, unless you are startin a new experiment, please remember to pass --resume."
                ).format(model_path)
            )
        return self

    def save_weights(self, args, epoch, opt_init=False):
        """
        Helper for saving weights according to argparse arguments
        :param opt_init if true, saves from init checkpoint instead of usual
        """
        from shutil import copyfile

        ckpt_name = "pixel_nerf_init" if opt_init else "pixel_nerf_{:04}".format(epoch)
        backup_name = "pixel_nerf_init_backup" if opt_init else "pixel_nerf_backup"
        latest_name = "pixel_nerf_init" if opt_init else "pixel_nerf_latest"

        ckpt_path = osp.join(args.checkpoints_path, args.name, ckpt_name)
        ckpt_backup_path = osp.join(args.checkpoints_path, args.name, backup_name)

        if osp.exists(ckpt_path):
            copyfile(ckpt_path, ckpt_backup_path)
        torch.save(self.state_dict(), ckpt_path)
        torch.save(self.state_dict(), latest_name)
        return self

    
    def encode_all_poses(self, poses):
        SB, NV, _ , _ = poses.shape
        poses = poses.reshape(-1, 4, 4)
        
        rot = poses[:, :3, :3].transpose(1, 2)  # (NV, 3, 3)
        trans = -torch.bmm(rot, poses[:, :3, 3:])  # (NV, 3, 1)
        poses_ref = torch.cat((rot, trans), dim=-1)  # (NV, 3, 4)
        self.poses_ref = poses_ref.reshape(SB, NV, 3, 4)

    def encode_pointcloud(self, points):
        self.pcloud = points