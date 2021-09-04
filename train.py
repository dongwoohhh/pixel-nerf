# Training to a set of multiple objects (e.g. ShapeNet or DTU)
# tensorboard logs available in logs/<expname>

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "train"))
)

import warnings
import trainlib
from model import make_model, loss
from render import NeRFRenderer
from data import get_split_dataset
import util
import numpy as np
import torch.nn.functional as F
import torch
from dotmap import DotMap


def extra_args(parser):
    parser.add_argument(
        "--batch_size", "-B", type=int, default=4, help="Object batch size ('SB')"
    )
    parser.add_argument(
        "--nviews",
        "-V",
        type=str,
        default="1",
        help="Number of source views (multiview); put multiple (space delim) to pick randomly per batch ('NV')",
    )
    parser.add_argument(
        "--freeze_enc",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )

    parser.add_argument(
        "--no_bbox_step",
        type=int,
        default=100000,
        help="Step to stop using bbox sampling",
    )
    parser.add_argument(
        "--fixed_test",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )
    parser.add_argument(
        "--use_color_ref",
        action="store_true",
        default=None,
        help="Use color ref loss",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=400000,
        help="Maximum training step",
    )
    parser.add_argument(
        "--no_save_model_step",
        type=int,
        default=200000,
        help="Step to stop using bbox sampling",
    )
    parser.add_argument(
        "--vis_source",
        "-VS",
        type=str,
        default=None,
        help="Source view(s) for each object. Alternatively, specify -L to viewlist file and leave this blank.",
    )
    return parser
    


args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=128)
device = util.get_cuda(args.gpu_id[0])

dset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir, args.split_name, pointcloud=True)
print(
    "dset z_near {}, z_far {}, lindisp {}".format(dset.z_near, dset.z_far, dset.lindisp)
)

net = make_model(conf["model"]).to(device=device)
net.stop_encoder_grad = args.freeze_enc
if args.freeze_enc:
    print("Encoder frozen")
    net.encoder.eval()

renderer = NeRFRenderer.from_conf(conf["renderer"], lindisp=dset.lindisp,).to(
    device=device
)

# Parallize
render_par = renderer.bind_parallel(net, args.gpu_id).eval()

nviews = list(map(int, args.nviews.split()))


class PixelNeRFTrainer(trainlib.Trainer):
    def __init__(self):
        super().__init__(net, dset, val_dset, args, conf["train"], device=device)
        self.renderer_state_path = "%s/%s/_renderer" % (
            self.args.checkpoints_path,
            self.args.name,
        )

        self.lambda_coarse = conf.get_float("loss.lambda_coarse")
        self.lambda_fine = conf.get_float("loss.lambda_fine", 1.0)
        print(
            "lambda coarse {} and fine {}".format(self.lambda_coarse, self.lambda_fine)
        )
        self.rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True)
        fine_loss_conf = conf["loss.rgb"]
        if "rgb_fine" in conf["loss"]:
            print("using fine loss")
            fine_loss_conf = conf["loss.rgb_fine"]
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        self.rgb_ref_crit = loss.RGBRefLoss()


        if args.resume:
            if os.path.exists(self.renderer_state_path):
                renderer.load_state_dict(
                    torch.load(self.renderer_state_path, map_location=device)
                )

        self.z_near = dset.z_near
        self.z_far = dset.z_far

        self.use_bbox = args.no_bbox_step > 0

    def post_batch(self, epoch, batch):
        renderer.sched_step(args.batch_size)

    def extra_save_state(self):
        torch.save(renderer.state_dict(), self.renderer_state_path)

    def calc_losses(self, data, is_train=True, global_step=0):
        if "images" not in data:
            return {}
        all_images = data["images"].to(device=device)  # (SB, NV, 3, H, W)

        SB, NV, _, H, W = all_images.shape
        all_poses = data["poses"].to(device=device)  # (SB, NV, 4, 4)
        all_bboxes = data.get("bbox")  # (SB, NV, 4)  cmin rmin cmax rmax
        all_focals = data["focal"]  # (SB)
        all_c = data.get("c")  # (SB)
        all_points = data.get("points").to(device=device)
        all_pcd_mask = data.get("pcd_mask").to(device=device)

        if self.use_bbox and global_step >= args.no_bbox_step:
            self.use_bbox = False
            print(">>> Stopped using bbox sampling @ iter", global_step)

        if not is_train or not self.use_bbox:
            all_bboxes = None

        all_rgb_gt = []
        all_rays = []
        all_index_target = []

        curr_nviews = nviews[torch.randint(0, len(nviews), ()).item()]
        if curr_nviews == 1:
            image_ord = torch.randint(0, NV, (SB, 1))
        else:
            image_ord = torch.empty((SB, curr_nviews), dtype=torch.long)
        for obj_idx in range(SB):
            if all_bboxes is not None:
                bboxes = all_bboxes[obj_idx]
            images = all_images[obj_idx]  # (NV, 3, H, W)
            poses = all_poses[obj_idx]  # (NV, 4, 4)
            focal = all_focals[obj_idx]
            c = None
            if "c" in data:
                c = data["c"][obj_idx]
            if curr_nviews > 1:
                # Somewhat inefficient, don't know better way
                image_ord[obj_idx] = torch.from_numpy(
                    np.random.choice(NV, curr_nviews, replace=False)
                )
            images_0to1 = images * 0.5 + 0.5

            cam_rays = util.gen_rays(
                poses, W, H, focal, self.z_near, self.z_far, c=c
            )  # (NV, H, W, 8)
            rgb_gt_all = images_0to1
            rgb_gt_all = (
                rgb_gt_all.permute(0, 2, 3, 1).contiguous().reshape(-1, 3)
            )  # (NV, H, W, 3)

            if all_bboxes is not None:
                pix = util.bbox_sample(bboxes, args.ray_batch_size)
                pix_inds = pix[..., 0] * H * W + pix[..., 1] * W + pix[..., 2]
            else:
                pix_inds = torch.randint(0, NV * H * W, (args.ray_batch_size,))
            
            idx_val = torch.stack([pix_inds//(H*W), pix_inds%(H*W)//(W), pix_inds%W]).T

            rgb_gt = rgb_gt_all[pix_inds]  # (ray_batch_size, 3)
            rays = cam_rays.view(-1, cam_rays.shape[-1])[pix_inds].to(
                device=device
            )  # (ray_batch_size, 8)
            index_target = torch.arange(0, NV).reshape(NV, 1, 1).repeat(1, H, W)
            index_target = index_target.contiguous().reshape(-1)
            index_target = index_target[pix_inds].to(device=device)  # (ray_batch_size, 1)

            all_rgb_gt.append(rgb_gt)
            all_rays.append(rays)
            all_index_target.append(index_target)

        all_rgb_gt = torch.stack(all_rgb_gt)  # (SB, ray_batch_size, 3)
        all_rgb_gt_multi = all_rgb_gt.unsqueeze(-2).repeat(1, 1, net.n_iteration, 1)
        all_rays = torch.stack(all_rays)  # (SB, ray_batch_size, 8)
        all_index_target = torch.stack(all_index_target)

        image_ord = image_ord.to(device)

        src_images = util.batched_index_select_nd(
            all_images, image_ord
        )  # (SB, NS, 3, H, W)
        src_poses = util.batched_index_select_nd(all_poses, image_ord)  # (SB, NS, 4, 4)

        net.encode(
            src_images,
            src_poses,
            all_focals.to(device=device),
            c=all_c.to(device=device) if all_c is not None else None,
        )
        net.encode_all_poses(all_poses)
        
        all_bboxes = all_poses = None

        render_dict = DotMap(render_par(all_rays, all_index_target, training=True, want_weights=True,))

        coarse = render_dict.coarse
        fine = render_dict.fine

        using_fine = len(fine) > 0
        loss_dict = {}
        rgb_loss = self.rgb_coarse_crit(coarse.rgb, all_rgb_gt)
        loss_dict["rc"] = rgb_loss.item() * self.lambda_coarse

        all_images_0to1 = all_images * 0.5 + 0.5
        #rgb_ref_loss = self.rgb_ref_crit(coarse.rgb_ref, coarse.uv_ref, coarse.points_ref, all_images_0to1, all_points, all_pcd_mask)
        #loss_dict["rc_ref"] = rgb_ref_loss.item() * self.lambda_coarse  

        if using_fine:
            fine_loss = self.rgb_fine_crit(fine.rgb, all_rgb_gt)
            rgb_loss = rgb_loss * self.lambda_coarse + fine_loss * self.lambda_fine
            loss_dict["rf"] = fine_loss.item() * self.lambda_fine

            #fine_ref_loss = self.rgb_ref_crit(fine.rgb_ref, fine.uv_ref, fine.points_ref, all_images_0to1, all_points, all_pcd_mask)
            #rgb_ref_loss = rgb_ref_loss * self.lambda_coarse + fine_ref_loss * self.lambda_fine
            #loss_dict["rf_ref"] = fine_ref_loss.item() * self.lambda_fine          

        #if args.use_color_ref:
        #    loss = rgb_loss + rgb_ref_loss
        #else:
        #    loss = rgb_loss
        loss = rgb_loss

        if is_train:
            loss.backward()
        loss_dict["t"] = loss.item()

        all_index_target = all_points = all_pcd_mask = all_images = None

        return loss_dict

    def train_step(self, data, global_step):
        return self.calc_losses(data, is_train=True, global_step=global_step)

    def eval_step(self, data, global_step):
        renderer.eval()
        losses = self.calc_losses(data, is_train=False, global_step=global_step)
        renderer.train()
        return losses

    def vis_step(self, data, global_step, idx=None):
        if "images" not in data:
            return {}
        if idx is None:
            batch_idx = np.random.randint(0, data["images"].shape[0])
        else:
            print(idx)
            batch_idx = idx
        images = data["images"][batch_idx].to(device=device)  # (NV, 3, H, W)
        poses = data["poses"][batch_idx].to(device=device)  # (NV, 4, 4)
        focal = data["focal"][batch_idx : batch_idx + 1]  # (1)
        c = data.get("c")
        if c is not None:
            c = c[batch_idx : batch_idx + 1]  # (1)
        NV, _, H, W = images.shape
        cam_rays = util.gen_rays(
            poses, W, H, focal, self.z_near, self.z_far, c=c
        )  # (NV, H, W, 8)
        images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)

        
        curr_nviews = nviews[torch.randint(0, len(nviews), (1,)).item()]

        if self.args.vis_source is None:
            views_src = np.sort(np.random.choice(NV, curr_nviews, replace=False))
        else:
            views_src = np.array(sorted(list(map(int, args.vis_source.split()))))
        view_dest = np.random.randint(0, NV - curr_nviews)

        for vs in range(curr_nviews):
            view_dest += view_dest >= views_src[vs]
        views_src = torch.from_numpy(views_src)

        # set renderer net to eval mode
        renderer.eval()
        source_views = (
            images_0to1[views_src]
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .reshape(-1, H, W, 3)
        )

        gt = images_0to1[view_dest].permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)
        with torch.no_grad():
            test_rays = cam_rays[view_dest]  # (H, W, 8)
            test_images = images[views_src]  # (NS, 3, H, W)
            test_poses = poses[view_dest]
            net.encode(
                test_images.unsqueeze(0),
                poses[views_src].unsqueeze(0),
                focal.to(device=device),
                c=c.to(device=device) if c is not None else None,
            )
            #index_dest =  view_dest*torch.ones((H, W), dtype=torch.long).to(device=device)
            index_dest =  torch.zeros((H, W), dtype=torch.long).to(device=device)
            index_dest = index_dest.reshape(1, -1, 1)
            
            test_rays = test_rays.reshape(1, H * W, -1)
            test_poses = test_poses.reshape(1, 1, 4, 4)
            net.encode_all_poses(test_poses)
            render_dict = DotMap(render_par(test_rays, index_dest, training=False, want_weights=True))
            coarse = render_dict.coarse
            fine = render_dict.fine

            using_fine = len(fine) > 0

            alpha_coarse_np = coarse.weights[0].sum(dim=-1).cpu().numpy().reshape(H, W)
            rgb_coarse_np = coarse.rgb[0].cpu().numpy().reshape(H, W, 3)
            depth_coarse_np = coarse.depth[0].cpu().numpy().reshape(H, W)

            rgb_coarse_multi_np = coarse.rgb_multi[0].cpu().numpy().reshape(H, W, net.n_iteration, 3)
            rgb_coarse_multi_np = rgb_coarse_multi_np.transpose(0, 2, 1, 3).reshape(H, W*net.n_iteration, 3)

            if using_fine:
                alpha_fine_np = fine.weights[0].sum(dim=1).cpu().numpy().reshape(H, W)
                depth_fine_np = fine.depth[0].cpu().numpy().reshape(H, W)
                rgb_fine_np = fine.rgb[0].cpu().numpy().reshape(H, W, 3)

                rgb_fine_multi_np = fine.rgb_multi[0].cpu().numpy().reshape(H, W, net.n_iteration, 3)
                rgb_fine_multi_np = rgb_fine_multi_np.transpose(0, 2, 1, 3).reshape(H, W*net.n_iteration, 3)

        print("c rgb min {} max {}".format(rgb_coarse_np.min(), rgb_coarse_np.max()))
        print(
            "c alpha min {}, max {}".format(
                alpha_coarse_np.min(), alpha_coarse_np.max()
            )
        )
        alpha_coarse_cmap = util.cmap(alpha_coarse_np) / 255
        depth_coarse_cmap = util.cmap(depth_coarse_np) / 255
        vis_list = [
            *source_views,
            gt,
            depth_coarse_cmap,
            rgb_coarse_np,
            alpha_coarse_cmap,
        ]

        vis_coarse = np.hstack(vis_list)
        vis = vis_coarse

        if using_fine:
            print("f rgb min {} max {}".format(rgb_fine_np.min(), rgb_fine_np.max()))
            print(
                "f alpha min {}, max {}".format(
                    alpha_fine_np.min(), alpha_fine_np.max()
                )
            )
            depth_fine_cmap = util.cmap(depth_fine_np) / 255
            alpha_fine_cmap = util.cmap(alpha_fine_np) / 255
            vis_list = [
                *source_views,
                gt,
                depth_fine_cmap,
                rgb_fine_np,
                alpha_fine_cmap,
            ]

            vis_fine = np.hstack(vis_list)
            vis = np.vstack((vis_coarse, vis_fine))
            rgb_psnr = rgb_fine_np
        else:
            rgb_psnr = rgb_coarse_np

        vis2 = np.vstack([np.hstack([rgb_coarse_multi_np, gt]), np.hstack([rgb_fine_multi_np, gt])])

        psnr = util.psnr(rgb_psnr, gt)
        vals = {"psnr": psnr}
        print("psnr", psnr)

        #cam_rays = images = index_dest = poses = render_dict = coarse = fine = None
        #test_rays = test_poses = test_images = None

        # set the renderer network back to train mode
        torch.cuda.empty_cache()
        renderer.train()
        return vis, vis2, vals


trainer = PixelNeRFTrainer()
trainer.start()
