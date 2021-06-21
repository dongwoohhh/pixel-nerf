"""
NeRF differentiable renderer.
References:
https://github.com/bmild/nerf
https://github.com/kwea123/nerf_pl
"""
import torch
import torch.nn.functional as F
import util
import torch.autograd.profiler as profiler
from torch.nn import DataParallel
from dotmap import DotMap


class _RenderWrapper(torch.nn.Module):
    def __init__(self, net, renderer, simple_output):
        super().__init__()
        self.net = net
        self.renderer = renderer
        self.simple_output = simple_output

    def forward(self, rays, poses, index_target,training, want_weights=False):
        if rays.shape[0] == 0:
            return (
                torch.zeros(0, 3, device=rays.device),
                torch.zeros(0, device=rays.device),
            )

        outputs = self.renderer(
            self.net, rays, poses, index_target, training, want_weights=want_weights and not self.simple_output
        )
        if self.simple_output:
            if self.renderer.using_fine:
                rgb = outputs.fine.rgb
                depth = outputs.fine.depth
            else:
                rgb = outputs.coarse.rgb
                depth = outputs.coarse.depth
            return rgb, depth
        else:
            # Make DotMap to dict to support DataParallel
            return outputs.toDict()


class NeRFRenderer(torch.nn.Module):
    """
    NeRF differentiable renderer
    :param n_coarse number of coarse (binned uniform) samples
    :param n_fine number of fine (importance) samples
    :param n_fine_depth number of expected depth samples
    :param noise_std noise to add to sigma. We do not use it
    :param depth_std noise for depth samples
    :param eval_batch_size ray batch size for evaluation
    :param white_bkgd if true, background color is white; else black
    :param lindisp if to use samples linear in disparity instead of distance
    :param sched ray sampling schedule. list containing 3 lists of equal length.
    sched[0] is list of iteration numbers,
    sched[1] is list of coarse sample numbers,
    sched[2] is list of fine sample numbers
    """

    def __init__(
        self,
        n_coarse=128,
        n_fine=0,
        n_fine_depth=0,
        noise_std=0.0,
        depth_std=0.01,
        weights_threshold=0.95,
        eval_batch_size=100000,
        white_bkgd=False,
        lindisp=False,
        sched=None,  # ray sampling schedule for coarse and fine rays
    ):
        super().__init__()
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.n_fine_depth = n_fine_depth

        self.noise_std = noise_std
        self.depth_std = depth_std
        self.weights_threshold = weights_threshold

        self.eval_batch_size = eval_batch_size
        self.white_bkgd = white_bkgd
        self.lindisp = lindisp
        if lindisp:
            print("Using linear displacement rays")
        self.using_fine = n_fine > 0
        self.sched = sched
        if sched is not None and len(sched) == 0:
            self.sched = None
        self.register_buffer(
            "iter_idx", torch.tensor(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "last_sched", torch.tensor(0, dtype=torch.long), persistent=True
        )

    def sample_coarse(self, rays):
        """
        Stratified sampling. Note this is different from original NeRF slightly.
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :return (B, Kc)
        """
        device = rays.device
        near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)

        step = 1.0 / self.n_coarse
        B = rays.shape[0]
        z_steps = torch.linspace(0, 1 - step, self.n_coarse, device=device)  # (Kc)
        z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # (B, Kc)
        z_steps += torch.rand_like(z_steps) * step
        if not self.lindisp:  # Use linear sampling in depth space
            return near * (1 - z_steps) + far * z_steps  # (B, Kf)
        else:  # Use linear sampling in disparity space
            return 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)

        # Use linear sampling in depth space
        return near * (1 - z_steps) + far * z_steps  # (B, Kc)

    def sample_fine(self, rays, weights):
        """
        Weighted stratified (importance) sample
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param weights (B, Kc)
        :return (B, Kf-Kfd)
        """
        device = rays.device
        B = rays.shape[0]

        weights = weights.detach() + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, -1, keepdim=True)  # (B, Kc)
        cdf = torch.cumsum(pdf, -1)  # (B, Kc)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (B, Kc+1)

        u = torch.rand(
            B, self.n_fine - self.n_fine_depth, dtype=torch.float32, device=device
        )  # (B, Kf)
        inds = torch.searchsorted(cdf, u, right=True).float() - 1.0  # (B, Kf)
        inds = torch.clamp_min(inds, 0.0)

        z_steps = (inds + torch.rand_like(inds)) / self.n_coarse  # (B, Kf)

        near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)
        if not self.lindisp:  # Use linear sampling in depth space
            z_samp = near * (1 - z_steps) + far * z_steps  # (B, Kf)
        else:  # Use linear sampling in disparity space
            z_samp = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)
        return z_samp

    def sample_fine_depth(self, rays, depth):
        """
        Sample around specified depth
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param depth (B)
        :return (B, Kfd)
        """
        z_samp = depth.unsqueeze(1).repeat((1, self.n_fine_depth))
        z_samp += torch.randn_like(z_samp) * self.depth_std
        # Clamp does not support tensor bounds
        z_samp = torch.max(torch.min(z_samp, rays[:, -1:]), rays[:, -2:-1])
        return z_samp

    def composite(self, model, rays, z_samp, poses, index_target, coarse=True, sb=0):
        """
        Render RGB and depth for each ray using NeRF alpha-compositing formula,
        given sampled positions along each ray (see sample_*)
        :param model should return (B, (r, g, b, sigma)) when called with (B, (x, y, z))
        should also support 'coarse' boolean argument
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param z_samp z positions sampled for each ray (B, K)
        :param coarse whether to evaluate using coarse NeRF
        :param sb super-batch dimension; 0 = disable
        :return weights (B, K), rgb (B, 3), depth (B)
        """
        with profiler.record_function("renderer_composite"):
            B, K = z_samp.shape

            deltas = z_samp[:, 1:] - z_samp[:, :-1]  # (B, K-1)
            #  if far:
            #      delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # infty (B, 1)
            delta_inf = rays[:, -1:] - z_samp[:, -1:]
            deltas = torch.cat([deltas, delta_inf], -1)  # (B, K)

            # (B, K, 3)
            points = rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]
            points = points.reshape(-1, 3)  # (B*K, 3)

            use_viewdirs = hasattr(model, "use_viewdirs") and model.use_viewdirs
            index_target = index_target[:, None].repeat(1, K)  # (B, K)

            val_all = []
            rgb_ray_all = []
            sigma_ray_all = []
            rgb_ref_all = []
            uv_ref_all = []
            transformer_key_all = []
            if sb > 0:
                points = points.reshape(
                    sb, -1, 3
                )  # (SB, B'*K, 3) B' is real ray batch size
                index_target = index_target.reshape(sb, -1)
                eval_batch_size = (self.eval_batch_size - 1) // sb + 1
                eval_batch_dim = 1
            else:
                eval_batch_size = self.eval_batch_size
                eval_batch_dim = 0
            #print(points.shape)
            split_points = torch.split(points, eval_batch_size, dim=eval_batch_dim)
            split_index = torch.split(index_target, eval_batch_size, dim=eval_batch_dim)   
            #raise NotImplementedError
            if use_viewdirs:
                dim1 = K
                viewdirs = rays[:, None, 3:6].expand(-1, dim1, -1)  # (B, K, 3)
                if sb > 0:
                    viewdirs = viewdirs.reshape(sb, -1, 3)  # (SB, B'*K, 3)
                else:
                    viewdirs = viewdirs.reshape(-1, 3)  # (B*K, 3)
                split_viewdirs = torch.split(
                    viewdirs, eval_batch_size, dim=eval_batch_dim
                )
                for pnts, dirs, indices in zip(split_points, split_viewdirs, split_index):
                    #output_ray, rgb_ref = model(pnts, indices, coarse=coarse, viewdirs=dirs)
                    #val_all.append(output_ray)
                    
                    rgb_ray, sigma_ray, transformer_key = model(pnts, indices, coarse=coarse, viewdirs=dirs, all_poses=poses)
                    rgb_ray_all.append(rgb_ray)
                    sigma_ray_all.append(sigma_ray)
                    if self.training:
                        transformer_key_all.append(transformer_key)
                    else:
                        del transformer_key
                    """
                    rgb_ray, sigma_ray, rgb_ref, uv_ref = model(pnts, indices, coarse=coarse, viewdirs=dirs)
                    rgb_ray_all.append(rgb_ray)
                    sigma_ray_all.append(sigma_ray)
                    rgb_ref_all.append(rgb_ref)
                    uv_ref_all.append(uv_ref)
                    """
            else:
                for pnts in split_points:
                    val_all.append(model(pnts, index_target, coarse=coarse))
            if self.training:
                transformer_keys = torch.cat(transformer_key_all, dim=eval_batch_dim)

            #print(transformer_keys.shape)
            
            # (B*K, 4) OR (SB, B'*K, 4)
            rgbs = torch.cat(rgb_ray_all, dim=eval_batch_dim)
            sigmas = torch.cat(sigma_ray_all, dim=eval_batch_dim)
            
            rgbs = rgbs.reshape(B, K, -1)
            sigmas = sigmas.reshape(B, K)
            """
            rgb_ref = torch.cat(rgb_ref_all, dim=eval_batch_dim)
            uv_ref = torch.cat(uv_ref_all, dim=eval_batch_dim)
            
            rgb_ref = rgb_ref.reshape(B, K, -1, 3)
            uv_ref = uv_ref.reshape(B, K, -1, 2)
            """

            if self.training and self.noise_std > 0.0:
                sigmas = sigmas + torch.randn_like(sigmas) * self.noise_std

            alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))  # (B, K)
            deltas = None
            sigmas = None
            alphas_shifted = torch.cat(
                [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
            )  # (B, K+1) = [1, a1, a2, ...]
            T = torch.cumprod(alphas_shifted, -1)  # (B)
            weights = alphas * T[:, :-1]  # (B, K)
            alphas = None
            alphas_shifted = None

            rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (B, 3)
            depth_final = torch.sum(weights * z_samp, -1)  # (B)

            NR = poses.shape[1]
            rgb_ref_all = torch.zeros((sb,B*K//sb, NR, 3), device=rgb_final.device)
            uv_ref_all = -1 * torch.ones((sb,B*K//sb, NR, 2), device=rgb_final.device)
            n_ref_all = torch.zeros((sb))

            if self.training:
                weights_ref = weights.reshape(-1)
                mask_ref = torch.where(weights_ref > self.weights_threshold, # 0.01,#
                                       torch.ones_like(weights_ref, device=rgb_final.device),
                                       torch.zeros_like(weights_ref, device=rgb_final.device))

                cond_ref = torch.sum(mask_ref)
                mask_ref = mask_ref.bool()[:, None]
                points_ref = torch.masked_select(points.reshape(-1, 3), mask_ref).reshape(-1, 3)

                viewdirs_ref = torch.masked_select(viewdirs.reshape(-1, 3), mask_ref).reshape(-1, 3)
                index_batch = torch.arange(sb, device=rgb_final.device)[:, None].repeat(1, B*K//sb).reshape(-1, 1)
                index_batch_ref = torch.masked_select(index_batch, mask_ref)
                _, _, NV, NC =transformer_key.shape
                """
                print(points.shape)
                print(mask_ref.shape)
                print(B*K, NV, NC)
                print(transformer_keys.reshape(B*K, -1).shape)
                """
                
                transformer_key_ref = torch.masked_select(transformer_keys.reshape(B*K, -1), mask_ref).reshape(-1, NV, NC)
                #raise NotImplementedError
                if cond_ref:
                    rgb_ref, uv_ref = model.forward_ref(points_ref, viewdirs_ref, index_batch_ref, poses, transformer_key_ref, coarse)                    

                    for i in range(sb):
                        mask_i = torch.eq(index_batch_ref, i)
                        n_batch_i = torch.sum(mask_i.int())
                        rgb_ref_all[i, :n_batch_i] = torch.masked_select(rgb_ref.reshape(-1, NR*3), mask_i.unsqueeze(-1)).reshape(n_batch_i, NR, 3)
                        uv_ref_all[i, :n_batch_i] = torch.masked_select(uv_ref.reshape(-1, NR*2), mask_i.unsqueeze(-1)).reshape(n_batch_i, NR, 2)
                        n_ref_all[i] = n_batch_i

            points = None
            viewdirs = None
            index_target = None
            if self.white_bkgd:
                # White background
                pix_alpha = weights.sum(dim=1)  # (B), pixel alpha
                rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)  # (B, 3)
            return (
                weights,
                rgb_final,
                depth_final,
                rgb_ref_all,
                uv_ref_all,
                n_ref_all
            )

    def forward(
        self, model, rays, poses, index_target, training, want_weights=False,
    ):
        """
        :model nerf model, should return (SB, B, (r, g, b, sigma))
        when called with (SB, B, (x, y, z)), for multi-object:
        SB = 'super-batch' = size of object batch,
        B  = size of per-object ray batch.
        Should also support 'coarse' boolean argument for coarse NeRF.
        :param rays ray spec [origins (3), directions (3), near (1), far (1)] (SB, B, 8)
        :param want_weights if true, returns compositing weights (SB, B, K)
        :return render dict
        """
        with profiler.record_function("renderer_forward"):
            self.training = training
            if self.sched is not None and self.last_sched.item() > 0:
                self.n_coarse = self.sched[1][self.last_sched.item() - 1]
                self.n_fine = self.sched[2][self.last_sched.item() - 1]

            assert len(rays.shape) == 3
            superbatch_size = rays.shape[0]
            rays = rays.reshape(-1, 8)  # (SB * B, 8)
            index_target = index_target.reshape(-1)
            z_coarse = self.sample_coarse(rays)  # (B, Kc)
            coarse_composite = self.composite(
                model, rays, z_coarse, poses, index_target, coarse=True, sb=superbatch_size,
            )

            outputs = DotMap(
                coarse=self._format_outputs(
                    coarse_composite, superbatch_size, want_weights=want_weights,
                ),
            )

            if self.using_fine:
                all_samps = [z_coarse]
                if self.n_fine - self.n_fine_depth > 0:
                    all_samps.append(
                        self.sample_fine(rays, coarse_composite[0].detach())
                    )  # (B, Kf - Kfd)
                if self.n_fine_depth > 0:
                    all_samps.append(
                        self.sample_fine_depth(rays, coarse_composite[2])
                    )  # (B, Kfd)
                z_combine = torch.cat(all_samps, dim=-1)  # (B, Kc + Kf)
                z_combine_sorted, argsort = torch.sort(z_combine, dim=-1)
                
                fine_composite = self.composite(
                    model, rays, z_combine_sorted, poses, index_target, coarse=False, sb=superbatch_size,
                )

                outputs.fine = self._format_outputs(
                    fine_composite, superbatch_size, want_weights=want_weights,
                )

            return outputs

    def _format_outputs(
        self, rendered_outputs, superbatch_size, want_weights=False,
    ):
        weights, rgb, depth, rgb_ref, uv_ref, n_ref = rendered_outputs #mask_ref
        if superbatch_size > 0:
            rgb = rgb.reshape(superbatch_size, -1, 3)
            depth = depth.reshape(superbatch_size, -1)
            weights = weights.reshape(superbatch_size, -1, weights.shape[-1])
            rgb_ref = rgb_ref.reshape(superbatch_size, -1, rgb_ref.shape[-2] , 3)
            uv_ref = uv_ref.reshape(superbatch_size, -1, rgb_ref.shape[-2], 2)
            n_ref = n_ref.reshape(superbatch_size)
            #mask_ref = mask_ref.reshape(superbatch_size,)
        ret_dict = DotMap(rgb=rgb, depth=depth, rgb_ref=rgb_ref, uv_ref=uv_ref, n_ref=n_ref)
        if want_weights:
            ret_dict.weights = weights
        return ret_dict

    def sched_step(self, steps=1):
        """
        Called each training iteration to update sample numbers
        according to schedule
        """
        if self.sched is None:
            return
        self.iter_idx += steps
        while (
            self.last_sched.item() < len(self.sched[0])
            and self.iter_idx.item() >= self.sched[0][self.last_sched.item()]
        ):
            self.n_coarse = self.sched[1][self.last_sched.item()]
            self.n_fine = self.sched[2][self.last_sched.item()]
            print(
                "INFO: NeRF sampling resolution changed on schedule ==> c",
                self.n_coarse,
                "f",
                self.n_fine,
            )
            self.last_sched += 1

    @classmethod
    def from_conf(cls, conf, white_bkgd=False, lindisp=False, eval_batch_size=100000):
        return cls(
            conf.get_int("n_coarse", 128),
            conf.get_int("n_fine", 0),
            n_fine_depth=conf.get_int("n_fine_depth", 0),
            noise_std=conf.get_float("noise_std", 0.0),
            depth_std=conf.get_float("depth_std", 0.01),
            white_bkgd=conf.get_float("white_bkgd", white_bkgd),
            lindisp=lindisp,
            eval_batch_size=conf.get_int("eval_batch_size", eval_batch_size),
            sched=conf.get_list("sched", None),
        )

    def bind_parallel(self, net, gpus=None, simple_output=False):
        """
        Returns a wrapper module compatible with DataParallel.
        Specifically, it renders rays with this renderer
        but always using the given network instance.
        Specify a list of GPU ids in 'gpus' to apply DataParallel automatically.
        :param net A PixelNeRF network
        :param gpus list of GPU ids to parallize to. If length is 1,
        does not parallelize
        :param simple_output only returns rendered (rgb, depth) instead of the 
        full render output map. Saves data tranfer cost.
        :return torch module
        """
        wrapped = _RenderWrapper(net, self, simple_output=simple_output)
        if gpus is not None and len(gpus) > 1:
            print("Using multi-GPU", gpus)
            wrapped = torch.nn.DataParallel(wrapped, gpus, dim=1)
        return wrapped
