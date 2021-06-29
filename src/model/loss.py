import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import random

class AlphaLossNV2(torch.nn.Module):
    """
    Implement Neural Volumes alpha loss 2
    """

    def __init__(self, lambda_alpha, clamp_alpha, init_epoch, force_opaque=False):
        super().__init__()
        self.lambda_alpha = lambda_alpha
        self.clamp_alpha = clamp_alpha
        self.init_epoch = init_epoch
        self.force_opaque = force_opaque
        if force_opaque:
            self.bceloss = torch.nn.BCELoss()
        self.register_buffer(
            "epoch", torch.tensor(0, dtype=torch.long), persistent=True
        )

    def sched_step(self, num=1):
        self.epoch += num

    def forward(self, alpha_fine):
        if self.lambda_alpha > 0.0 and self.epoch.item() >= self.init_epoch:
            alpha_fine = torch.clamp(alpha_fine, 0.01, 0.99)
            if self.force_opaque:
                alpha_loss = self.lambda_alpha * self.bceloss(
                    alpha_fine, torch.ones_like(alpha_fine)
                )
            else:
                alpha_loss = torch.log(alpha_fine) + torch.log(1.0 - alpha_fine)
                alpha_loss = torch.clamp_min(alpha_loss, -self.clamp_alpha)
                alpha_loss = self.lambda_alpha * alpha_loss.mean()
        else:
            alpha_loss = torch.zeros(1, device=alpha_fine.device)
        return alpha_loss


def get_alpha_loss(conf):
    lambda_alpha = conf.get_float("lambda_alpha")
    clamp_alpha = conf.get_float("clamp_alpha")
    init_epoch = conf.get_int("init_epoch")
    force_opaque = conf.get_bool("force_opaque", False)

    return AlphaLossNV2(
        lambda_alpha, clamp_alpha, init_epoch, force_opaque=force_opaque
    )


class RGBWithUncertainty(torch.nn.Module):
    """Implement the uncertainty loss from Kendall '17"""

    def __init__(self, conf):
        super().__init__()
        self.element_loss = (
            torch.nn.L1Loss(reduction="none")
            if conf.get_bool("use_l1")
            else torch.nn.MSELoss(reduction="none")
        )

    def forward(self, outputs, targets, betas):
        """computes the error per output, weights each element by the log variance
        outputs is B x 3, targets is B x 3, betas is B"""
        weighted_element_err = (
            torch.mean(self.element_loss(outputs, targets), -1) / betas
        )
        return torch.mean(weighted_element_err) + torch.mean(torch.log(betas))


class RGBWithBackground(torch.nn.Module):
    """Implement the uncertainty loss from Kendall '17"""

    def __init__(self, conf):
        super().__init__()
        self.element_loss = (
            torch.nn.L1Loss(reduction="none")
            if conf.get_bool("use_l1")
            else torch.nn.MSELoss(reduction="none")
        )

    def forward(self, outputs, targets, lambda_bg):
        """If we're using background, then the color is color_fg + lambda_bg * color_bg.
        We want to weight the background rays less, while not putting all alpha on bg"""
        weighted_element_err = torch.mean(self.element_loss(outputs, targets), -1) / (
            1 + lambda_bg
        )
        return torch.mean(weighted_element_err) + torch.mean(torch.log(lambda_bg))


def get_rgb_loss(conf, coarse=True, using_bg=False, reduction="mean"):
    if conf.get_bool("use_uncertainty", False) and not coarse:
        print("using loss with uncertainty")
        return RGBWithUncertainty(conf)
    #     if using_bg:
    #         print("using loss with background")
    #         return RGBWithBackground(conf)
    print("using vanilla rgb loss")
    return (
        torch.nn.L1Loss(reduction=reduction)
        if conf.get_bool("use_l1")
        else torch.nn.MSELoss(reduction=reduction)
    )

class RGBRefLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss(reduction="none")
        
        self.register_buffer("scale", torch.empty(2, dtype=torch.float32), persistent=False)
    def forward(self, rgb_ref, idx_pcloud, color_pcloud, mask_pcloud):
        SB, B, NR, _ = rgb_ref.shape

        loss = []
        for i in range(SB):
            idx_pcloud_i = idx_pcloud[i]

            n_samples = idx_pcloud_i[idx_pcloud_i!=-1].shape[0]

            rgb_ref_i = rgb_ref[i, :n_samples]
            idx_pcloud_i = idx_pcloud[i, :n_samples].long()
            
            rgb_gt_i = color_pcloud[i, :, idx_pcloud_i, :].squeeze(2).transpose(1, 0)
            mask_gt_i = mask_pcloud[i, :, idx_pcloud_i, :].squeeze(2).transpose(1, 0)
            
            mask_sum = torch.sum(mask_gt_i)
            if mask_sum > 0:
                loss_i = self.l1_loss(rgb_ref_i, rgb_gt_i)
                loss_i = torch.mul(loss_i, mask_gt_i)
                loss_i = torch.sum(loss_i) / (mask_sum * 3)
            else:
                loss_i = mask_sum

            loss.append(loss_i)

        loss = torch.stack(loss)
        loss = torch.mean(loss)

        return loss




        image_size = (float(W), float(H))
        uv_ref = uv_ref.transpose(1, 2).reshape(SB*NR, B, 2)
        images = images.reshape(SB*NR, 3, H, W)
        #weights = weights.reshape(SB, -1)
        rgb_ref_gt, mask_uv = self.index_images(uv_ref, images, image_size)

        rgb_ref_gt = rgb_ref_gt.reshape(SB, NR, 3, B).permute(0, 3, 1, 2)
        mask_uv = mask_uv.reshape(SB, NR, B).transpose(1, 2)
        """
        # For visualize.
        i = random.randint(1, 100)
        if torch.sum(mask_uv[0, 0]) > 0:
            print(torch.sum(mask_uv[0, 0]))
            print(rgb_ref_gt[0, 0, :, :].shape)
            save_image(images[24], 'image_{}.png'.format(i))
            tmp = rgb_ref_gt[0, 0, :, :]
            #tmp = tmp.repeat(30,1)
            #tmp = tmp[:, None, :].repeat(1, 30, 1).reshape(7*30, 7*30, 3)
            
            tmp = tmp[None, :, None, :].repeat(30, 1, 30, 1).reshape(30, -1, 3)
            print(tmp.shape)
            tmp = tmp.reshape(30, 30*49 , 3)
            save_image(tmp.permute(2, 0, 1), 'pick_{}.png'.format(i))
            print(uv_ref[24, 0, :])
        """


        n_uv = torch.sum(mask_uv, dim=2)
        loss = self.l1_loss(rgb_ref, rgb_ref_gt)

        if torch.sum(n_uv) > 0:
            loss = torch.sum(torch.mean(mask_uv.unsqueeze(-1) * loss, dim=3)) / torch.sum(n_uv) #.mean((2,3))

            return loss / 10.
        else:
            loss = torch.sum(mask_uv.unsqueeze(-1) * loss)

            return loss / 10.

    def index_images(self, uv, images, image_size):
        self.scale[0] = 1 / image_size[0]
        self.scale[1] = 1 / image_size[1]
        self.scale = self.scale.to(device=uv.device)
        epsilon = [1*self.scale[0]*2, 1*self.scale[1]*2]
        
        uv = uv * self.scale * 2 - 1.0
        mask = torch.where((uv[:, :, 0]> -1.0-epsilon[0]) & (uv[:, :, 0] <=1.0+epsilon[0]) & (uv[:, :, 1]>=-1.0-epsilon[1]) & (uv[:, :, 1]<=1.0+epsilon[1]),
                            torch.ones_like(uv[:,:, 0]), torch.zeros_like(uv[:,:, 0]))
        
        uv = uv.unsqueeze(2)
        samples = F.grid_sample(
            images,
            uv,
            align_corners=True, mode='bilinear',
            padding_mode='zeros'
        )
        return samples[:, :, :, 0], mask
