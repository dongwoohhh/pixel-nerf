import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
import open3d as o3d
import cv2
from util import get_image_to_tensor_balanced, get_mask_to_tensor


class DVRDataset(torch.utils.data.Dataset):
    """
    Dataset from DVR (Niemeyer et al. 2020)
    Provides 3D-R2N2 and NMR renderings
    """

    def __init__(
        self,
        path,
        stage="train",
        list_prefix="softras_",
        image_size=None,
        sub_format="shapenet",
        scale_focal=True,
        max_imgs=100000,
        z_near=1.2,
        z_far=4.0,
        skip_step=None,
        pointcloud=False,
        n_points=0,
    ):
        """
        :param path dataset root path, contains metadata.yml
        :param stage train | val | test
        :param list_prefix prefix for split lists: <list_prefix>[train, val, test].lst
        :param image_size result image size (resizes if different); None to keep original size
        :param sub_format shapenet | dtu dataset sub-type.
        :param scale_focal if true, assume focal length is specified for
        image of side length 2 instead of actual image size. This is used
        where image coordinates are placed in [-1, 1].
        """
        super().__init__()
        self.base_path = path
        assert os.path.exists(self.base_path)

        cats = [x for x in glob.glob(os.path.join(path, "*")) if os.path.isdir(x)]

        if stage == "train":
            file_lists = [os.path.join(x, list_prefix + "train.lst") for x in cats]
        elif stage == "val":
            file_lists = [os.path.join(x, list_prefix + "val.lst") for x in cats]
        elif stage == "test":
            file_lists = [os.path.join(x, list_prefix + "test.lst") for x in cats]

        all_objs = []
        for file_list in file_lists:
            if not os.path.exists(file_list):
                continue
            base_dir = os.path.dirname(file_list)
            cat = os.path.basename(base_dir)
            with open(file_list, "r") as f:
                objs = [(cat, os.path.join(base_dir, x.strip())) for x in f.readlines()]
            all_objs.extend(objs)

        self.all_objs = all_objs
        self.stage = stage

        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()
        print(
            "Loading DVR dataset",
            self.base_path,
            "stage",
            stage,
            len(self.all_objs),
            "objs",
            "type:",
            sub_format,
        )

        self.image_size = image_size
        if sub_format == "dtu":
            self._coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            self._coord_trans_cam = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
        else:
            self._coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            self._coord_trans_cam = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
        self.sub_format = sub_format
        self.scale_focal = scale_focal
        self.max_imgs = max_imgs

        self.z_near = z_near
        self.z_far = z_far
        self.lindisp = False

        self.pointcloud = pointcloud
        self.n_points = n_points

    def __len__(self):
        return len(self.all_objs)

    def __getitem__(self, index):
        cat, root_dir = self.all_objs[index]

        rgb_paths = [
            x
            for x in glob.glob(os.path.join(root_dir, "image", "*"))
            if (x.endswith(".jpg") or x.endswith(".png"))
        ]
        rgb_paths = sorted(rgb_paths)
        mask_paths = sorted(glob.glob(os.path.join(root_dir, "mask", "*.png")))
        if len(mask_paths) == 0:
            mask_paths = [None] * len(rgb_paths)

        if len(rgb_paths) <= self.max_imgs:
            sel_indices = np.arange(len(rgb_paths))
        else:
            sel_indices = np.random.choice(len(rgb_paths), self.max_imgs, replace=False)
            rgb_paths = [rgb_paths[i] for i in sel_indices]
            mask_paths = [mask_paths[i] for i in sel_indices]

        cam_path = os.path.join(root_dir, "cameras.npz")
        all_cam = np.load(cam_path)

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        focal = None
        if self.sub_format != "shapenet":
            # Prepare to average intrinsics over images
            fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0

        for idx, (rgb_path, mask_path) in enumerate(zip(rgb_paths, mask_paths)):
            i = sel_indices[idx]
            img = imageio.imread(rgb_path)[..., :3]
            if self.scale_focal:
                x_scale = img.shape[1] / 2.0
                y_scale = img.shape[0] / 2.0
                xy_delta = 1.0
            else:
                x_scale = y_scale = 1.0
                xy_delta = 0.0

            if mask_path is not None:
                mask = imageio.imread(mask_path)
                if len(mask.shape) == 2:
                    mask = mask[..., None]
                mask = mask[..., :1]
            if self.sub_format == "dtu":
                # Decompose projection matrix
                # DVR uses slightly different format for DTU set
                P = all_cam["world_mat_" + str(i)]
                P = P[:3]

                K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
                K = K / K[2, 2]

                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = R.transpose()
                pose[:3, 3] = (t[:3] / t[3])[:, 0]

                scale_mtx = all_cam.get("scale_mat_" + str(i))
                if scale_mtx is not None:
                    norm_trans = scale_mtx[:3, 3:]
                    norm_scale = np.diagonal(scale_mtx[:3, :3])[..., None]

                    pose[:3, 3:] -= norm_trans
                    pose[:3, 3:] /= norm_scale

                fx += torch.tensor(K[0, 0]) * x_scale
                fy += torch.tensor(K[1, 1]) * y_scale
                cx += (torch.tensor(K[0, 2]) + xy_delta) * x_scale
                cy += (torch.tensor(K[1, 2]) + xy_delta) * y_scale
            else:
                # ShapeNet
                wmat_inv_key = "world_mat_inv_" + str(i)
                wmat_key = "world_mat_" + str(i)
                if wmat_inv_key in all_cam:
                    extr_inv_mtx = all_cam[wmat_inv_key]
                else:
                    extr_inv_mtx = all_cam[wmat_key]
                    if extr_inv_mtx.shape[0] == 3:
                        extr_inv_mtx = np.vstack((extr_inv_mtx, np.array([0, 0, 0, 1])))
                    extr_inv_mtx = np.linalg.inv(extr_inv_mtx)

                intr_mtx = all_cam["camera_mat_" + str(i)]
                fx, fy = intr_mtx[0, 0], intr_mtx[1, 1]
                assert abs(fx - fy) < 1e-9
                fx = fx * x_scale
                if focal is None:
                    focal = fx
                else:
                    assert abs(fx - focal) < 1e-5
                pose = extr_inv_mtx

            pose = (
                self._coord_trans_world
                @ torch.tensor(pose, dtype=torch.float32)
                @ self._coord_trans_cam
            )

            img_tensor = self.image_to_tensor(img)
            if mask_path is not None:
                mask_tensor = self.mask_to_tensor(mask)

                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rnz = np.where(rows)[0]
                cnz = np.where(cols)[0]
                if len(rnz) == 0:
                    raise RuntimeError(
                        "ERROR: Bad image at", rgb_path, "please investigate!"
                    )
                rmin, rmax = rnz[[0, -1]]
                cmin, cmax = cnz[[0, -1]]
                bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
                all_masks.append(mask_tensor)
                all_bboxes.append(bbox)

            all_imgs.append(img_tensor)
            all_poses.append(pose)

        if self.sub_format != "shapenet":
            fx /= len(rgb_paths)
            fy /= len(rgb_paths)
            cx /= len(rgb_paths)
            cy /= len(rgb_paths)
            focal = torch.tensor((fx, fy), dtype=torch.float32)
            c = torch.tensor((cx, cy), dtype=torch.float32)
            all_bboxes = None
        elif mask_path is not None:
            all_bboxes = torch.stack(all_bboxes)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        if len(all_masks) > 0:
            all_masks = torch.stack(all_masks)
        else:
            all_masks = None

        if self.image_size is not None and all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            if self.sub_format != "shapenet":
                c *= scale
            elif mask_path is not None:
                all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            if all_masks is not None:
                all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")


        # Read pointcloud.        
        if self.pointcloud:
            n_points_batch = 500000

            pcd_data_dir = os.path.join(root_dir, 'pcd_data.pt')
            points_dir = os.path.join(root_dir, 'points.pt')
            n_views, _, height, width = all_imgs.shape
            if os.path.isfile(pcd_data_dir):
                pcd_data = torch.load(pcd_data_dir)
                points = torch.load(points_dir)
            else:
                pcd_load = o3d.io.read_point_cloud(os.path.join(root_dir, 'points.ply'))
                pcd_load = pcd_load.voxel_down_sample(voxel_size=1.0)
                
                points = np.asarray(pcd_load.points)
                colors_pcd = np.asarray(pcd_load.colors)
                
                n_points = points.shape[0]

                points -= norm_trans.T
                points /= norm_scale.T
            
                points = torch.tensor(points, dtype=torch.float32)
                points = torch.inverse(self._coord_trans_cam[:3, :3]) @ points.T
                points = points.T

                # Get color of points from multi-view images.
                focal[..., 1] *= -1.0

                rot = all_poses[:, :3, :3].transpose(2, 1)
                trans = - torch.matmul(rot, all_poses[:, :3, 3:])
                poses = torch.cat((rot, trans), dim=-1)

                xyz = points[None].repeat(n_views, 1, 1)
                xyz = torch.matmul(rot, xyz.transpose(2, 1))
                xyz = xyz + poses[:, :3, 3, None]
                xyz = xyz.transpose(2, 1)
                depth = - xyz[:, :, 2:3]
                
                uv = - xyz[:, :, :2] / xyz[:, :, 2:3]        
                uv *= focal[None, None, :]
                uv += c[None, None, :]

                scale = np.array([1 / width, 1 / height], dtype=np.float32)

                uv_normalized = uv*scale * 2 - 1.0

                color = F.grid_sample(
                    all_imgs,
                    uv_normalized.unsqueeze(2),
                    align_corners=True, mode='bilinear',
                    padding_mode='zeros'
                )[:, :, :, 0].transpose(2, 1)

                uv_round = torch.round(uv)
                mask_list = []
                for i in range(n_views):
                    print('{} / {}'.format(i, n_views))
                    
                    # To use stable sort convert to numpy tensor.
                    uv_i = uv_round[i].numpy()
                    color_i = color[i].numpy()
                    depth_i = depth[i].numpy()
                    
                    data_i = np.concatenate([uv_i, color_i, depth_i], axis=1)
                    
                    data_i.dtype = [('u', 'float32'), ('v','float32'), ('c1', 'float32'), ('c2', 'float32'), ('c3', 'float32'), ('d', 'float32')]

                    idx_sorted = np.argsort(data_i, axis=0, order=('u', 'v', 'd'))[:, 0]
                    idx_original = np.argsort(idx_sorted)

                    data_sorted = data_i[idx_sorted]

                    data_sorted_prev = np.roll(data_sorted, shift=1, axis=0)
                    mask_i = (data_sorted['u']>=0) & (data_sorted['u']<width) \
                            & (data_sorted['v']>=0) & (data_sorted['v']<height) \
                            & ((data_sorted['u'] != data_sorted_prev['u']) \
                            | (data_sorted['v'] != data_sorted_prev['v']))

                    mask_i = mask_i[idx_original]
                    mask_list.append(mask_i)
                    #pixel_id = data_sor
                    uv_i = color_i = depth_i = data_i = data_sorted = data_sorted_prev = None
                mask = np.stack(mask_list)
                mask = torch.tensor(mask)

                pcd_data = torch.cat([color, mask], dim=-1)
                torch.save(pcd_data, pcd_data_dir)    
                torch.save(points, points_dir)


            # Make same number of points in tensor
            
            n_points = points.shape[0]
            if n_points > n_points_batch:
                indices = np.random.permutation(n_points)
                
                pcd_data = pcd_data[indices[:n_points_batch]]
                points = points[indices[:n_points_batch]]
            else:
                pcd_zeros = torch.zeros((n_views, n_points_batch-n_points, 4))
                points_zeros = torch.zeros((n_points_batch-n_points, 3))
                pcd_data = torch.cat([pcd_data, pcd_zeros], dim=1)
                points = torch.cat([points, points_zeros], dim=0)

        """
        for i in range(n_views):
            uv_i = uv[i]
            xy = np.floor(uv_i.numpy()).astype(np.int32)

            mask = np.where((xy[:, 0]>0) & (xy[:, 0]<400) & (xy[:, 1]>0) & (xy[:, 1]<300))
            xy = xy[mask]
            colors_i = colors_pcd[mask]
            image_w = np.zeros((300, 400, 3), dtype=np.float32)
            image_w[xy[:, 1], xy[:, 0], :] = colors_i * 255
            cv2.imwrite('debug_pcd/warped_{}.png'.format(i), cv2.cvtColor(image_w, cv2.COLOR_RGB2BGR))
        """
        
        result = {
            "path": root_dir,
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            "poses": all_poses,
            "points": points,
            "pcd_data": pcd_data
        }

        if all_masks is not None:
            result["masks"] = all_masks
        if self.sub_format != "shapenet":
            result["c"] = c
        else:
            result["bbox"] = all_bboxes
        return result
