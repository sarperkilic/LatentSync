# Adapted from https://github.com/guanjz20/StyleSync/blob/main/utils.py

import numpy as np
import cv2
import torch
from einops import rearrange
import kornia


class AlignRestore(object):
    def __init__(self, align_points=3, resolution=256, device="cpu", dtype=torch.float16):
        if align_points == 3:
            self.upscale_factor = 1
            ratio = resolution / 256 * 2.8
            self.crop_ratio = (ratio, ratio)
            self.face_template = np.array([[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]])
            self.face_template = self.face_template * ratio
            self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
            self.p_bias = None
            self.device = device
            self.dtype = dtype
            self.fill_value = torch.tensor([127, 127, 127], device=device, dtype=dtype)
            self.mask = torch.ones((1, 1, self.face_size[1], self.face_size[0]), device=device, dtype=dtype)

    def align_warp_face_gpu(self, img, landmarks3, smooth=True):
        affine_matrix, self.p_bias = self.transformation_from_points(
            landmarks3, self.face_template, smooth, self.p_bias
        )

        # Convert affine_matrix to GPU tensor
        affine_matrix = torch.from_numpy(affine_matrix).to(device=self.device, dtype=self.dtype).unsqueeze(0)

        # Ensure img is in correct format (C, H, W) and on GPU
        if img.dim() == 3 and img.shape[2] == 3:  # H, W, C
            img = rearrange(img, "h w c -> c h w")
        img = img.to(device=self.device, dtype=self.dtype).unsqueeze(0)

        cropped_face = kornia.geometry.transform.warp_affine(
            img,
            affine_matrix,
            (self.face_size[1], self.face_size[0]),
            mode="bilinear",
            padding_mode="fill",
            fill_value=self.fill_value,
        )
        
        # Return GPU tensor directly (C, H, W format)
        return cropped_face.squeeze(0), affine_matrix.squeeze(0)

    def align_warp_face(self, img, landmarks3, smooth=True):
        affine_matrix, self.p_bias = self.transformation_from_points(
            landmarks3, self.face_template, smooth, self.p_bias
        )

        img = rearrange(torch.from_numpy(img).to(device=self.device, dtype=self.dtype), "h w c -> c h w").unsqueeze(0)
        affine_matrix = torch.from_numpy(affine_matrix).to(device=self.device, dtype=self.dtype).unsqueeze(0)

        cropped_face = kornia.geometry.transform.warp_affine(
            img,
            affine_matrix,
            (self.face_size[1], self.face_size[0]),
            mode="bilinear",
            padding_mode="fill",
            fill_value=self.fill_value,
        )
        cropped_face = rearrange(cropped_face.squeeze(0), "c h w -> h w c").cpu().numpy().astype(np.uint8)
        return cropped_face, affine_matrix

    def restore_img_gpu(self, input_img, face, affine_matrix):
        """
        GPU-optimized version that keeps data on GPU and returns GPU tensor
        Optimized to eliminate CPU-GPU transfers and improve performance
        """
        h, w, _ = input_img.shape

        # Ensure affine_matrix has correct shape [B, 2, 3] for kornia
        if isinstance(affine_matrix, np.ndarray):
            affine_matrix = torch.from_numpy(affine_matrix).to(device=self.device, dtype=self.dtype)
        
        # Add batch dimension if missing
        if affine_matrix.dim() == 2:  # [2, 3]
            affine_matrix = affine_matrix.unsqueeze(0)  # [1, 2, 3]
        elif affine_matrix.dim() == 3 and affine_matrix.shape[0] != 1:  # [2, 3, ?] or other
            affine_matrix = affine_matrix.unsqueeze(0)  # [1, 2, 3, ?]

        inv_affine_matrix = kornia.geometry.transform.invert_affine_transform(affine_matrix)
        face = face.to(dtype=self.dtype).unsqueeze(0)

        inv_face = kornia.geometry.transform.warp_affine(
            face, inv_affine_matrix, (h, w), mode="bilinear", padding_mode="fill", fill_value=self.fill_value
        ).squeeze(0)
        inv_face = (inv_face / 2 + 0.5).clamp(0, 1) * 255

        input_img = rearrange(input_img.to(dtype=self.dtype), "h w c -> c h w")
        inv_mask = kornia.geometry.transform.warp_affine(
            self.mask, inv_affine_matrix, (h, w), padding_mode="zeros"
        )  # (1, 1, h_up, w_up)

        inv_mask_erosion = kornia.morphology.erosion(
            inv_mask,
            torch.ones(
                (int(2 * self.upscale_factor), int(2 * self.upscale_factor)), device=self.device, dtype=self.dtype
            ),
        )

        inv_mask_erosion_t = inv_mask_erosion.squeeze(0).expand_as(inv_face)
        pasted_face = inv_mask_erosion_t * inv_face
        total_face_area = torch.sum(inv_mask_erosion.float())
        w_edge = int(total_face_area**0.5) // 20
        erosion_radius = w_edge * 2

        erosion_kernel = torch.ones(
            (erosion_radius, erosion_radius), device=self.device, dtype=self.dtype
        )
        inv_mask_center = kornia.morphology.erosion(inv_mask_erosion, erosion_kernel)

        blur_size = w_edge * 2 + 1
        sigma = 0.3 * ((blur_size - 1) * 0.5 - 1) + 0.8
        inv_soft_mask = kornia.filters.gaussian_blur2d(
            inv_mask_center, (blur_size, blur_size), (sigma, sigma)
        ).squeeze(0)
        inv_soft_mask_3d = inv_soft_mask.expand_as(inv_face)
        img_back = inv_soft_mask_3d * pasted_face + (1 - inv_soft_mask_3d) * input_img

        # Return GPU tensor directly (H, W, C format)
        img_back = rearrange(img_back, "c h w -> h w c").contiguous().to(dtype=torch.uint8)
        return img_back

    def restore_img(self, input_img, face, affine_matrix):
        h, w, _ = input_img.shape

        # Ensure affine_matrix has correct shape [B, 2, 3] for kornia
        if isinstance(affine_matrix, np.ndarray):
            affine_matrix = torch.from_numpy(affine_matrix).to(device=self.device, dtype=self.dtype)
        
        # Add batch dimension if missing
        if affine_matrix.dim() == 2:  # [2, 3]
            affine_matrix = affine_matrix.unsqueeze(0)  # [1, 2, 3]
        elif affine_matrix.dim() == 3 and affine_matrix.shape[0] != 1:  # [2, 3, ?] or other
            affine_matrix = affine_matrix.unsqueeze(0)  # [1, 2, 3, ?]

        inv_affine_matrix = kornia.geometry.transform.invert_affine_transform(affine_matrix)
        face = face.to(dtype=self.dtype).unsqueeze(0)

        inv_face = kornia.geometry.transform.warp_affine(
            face, inv_affine_matrix, (h, w), mode="bilinear", padding_mode="fill", fill_value=self.fill_value
        ).squeeze(0)
        inv_face = (inv_face / 2 + 0.5).clamp(0, 1) * 255

        input_img = rearrange(torch.from_numpy(input_img).to(device=self.device, dtype=self.dtype), "h w c -> c h w")
        inv_mask = kornia.geometry.transform.warp_affine(
            self.mask, inv_affine_matrix, (h, w), padding_mode="zeros"
        )  # (1, 1, h_up, w_up)

        inv_mask_erosion = kornia.morphology.erosion(
            inv_mask,
            torch.ones(
                (int(2 * self.upscale_factor), int(2 * self.upscale_factor)), device=self.device, dtype=self.dtype
            ),
        )

        inv_mask_erosion_t = inv_mask_erosion.squeeze(0).expand_as(inv_face)
        pasted_face = inv_mask_erosion_t * inv_face
        total_face_area = torch.sum(inv_mask_erosion.float())
        w_edge = int(total_face_area**0.5) // 20
        erosion_radius = w_edge * 2

        # This step will consume a large amount of GPU memory.
        # inv_mask_center = kornia.morphology.erosion(
        #     inv_mask_erosion, torch.ones((erosion_radius, erosion_radius), device=self.device, dtype=self.dtype)
        # )

        # Run on CPU to avoid consuming a large amount of GPU memory.
        inv_mask_erosion = inv_mask_erosion.squeeze().cpu().numpy().astype(np.float32)
        inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
        inv_mask_center = torch.from_numpy(inv_mask_center).to(device=self.device, dtype=self.dtype)[None, None, ...]

        blur_size = w_edge * 2 + 1
        sigma = 0.3 * ((blur_size - 1) * 0.5 - 1) + 0.8
        inv_soft_mask = kornia.filters.gaussian_blur2d(
            inv_mask_center, (blur_size, blur_size), (sigma, sigma)
        ).squeeze(0)
        inv_soft_mask_3d = inv_soft_mask.expand_as(inv_face)
        img_back = inv_soft_mask_3d * pasted_face + (1 - inv_soft_mask_3d) * input_img

        img_back = rearrange(img_back, "c h w -> h w c").contiguous().to(dtype=torch.uint8)
        img_back = img_back.cpu().numpy()
        return img_back

    def restore_img_batch_gpu(self, input_imgs, faces, affine_matrices):
        """
        Batch version of restore_img_gpu for processing multiple faces simultaneously
        Args:
            input_imgs: torch.Tensor of shape (N, H, W, C) - batch of input images
            faces: torch.Tensor of shape (N, C, H, W) - batch of faces
            affine_matrices: list of affine matrices for each face
        Returns:
            torch.Tensor of shape (N, H, W, C) - batch of restored images
        """
        batch_size = len(input_imgs)
        device = self.device
        dtype = self.dtype
        
        if input_imgs.dtype != dtype:
            input_imgs = input_imgs.to(dtype=dtype)
        if faces.dtype != dtype:
            faces = faces.to(dtype=dtype)
        
        # Convert affine matrices to tensor batch
        affine_batch = []
        for affine_matrix in affine_matrices:
            if isinstance(affine_matrix, np.ndarray):
                affine_tensor = torch.from_numpy(affine_matrix).to(device=device, dtype=dtype)
            else:
                affine_tensor = affine_matrix.to(device=device, dtype=dtype)
            
            # Ensure correct shape [2, 3]
            if affine_tensor.dim() == 3:
                affine_tensor = affine_tensor.squeeze(0)
            affine_batch.append(affine_tensor)
        
        affine_batch = torch.stack(affine_batch)  # (N, 2, 3)
        
        # Get image dimensions
        h, w = input_imgs.shape[1], input_imgs.shape[2]
        
        # Invert affine matrices
        inv_affine_matrices = kornia.geometry.transform.invert_affine_transform(affine_batch)
        
        # Warp faces back to original image space
        inv_faces = kornia.geometry.transform.warp_affine(
            faces, inv_affine_matrices, (h, w), mode="bilinear", padding_mode="fill", fill_value=self.fill_value
        )
        inv_faces = (inv_faces / 2 + 0.5).clamp(0, 1) * 255
        
        # Rearrange input images to (N, C, H, W)
        input_imgs_chw = rearrange(input_imgs, "n h w c -> n c h w")
        
        # Warp masks
        mask_batch = self.mask.expand(batch_size, -1, -1, -1)  # (N, 1, H_mask, W_mask)
        inv_masks = kornia.geometry.transform.warp_affine(
            mask_batch, inv_affine_matrices, (h, w), padding_mode="zeros"
        )  # (N, 1, H, W)
        
        # Apply erosion to masks
        erosion_kernel = torch.ones(
            (int(2 * self.upscale_factor), int(2 * self.upscale_factor)), device=device, dtype=dtype
        )
        inv_mask_erosion = kornia.morphology.erosion(inv_masks, erosion_kernel)
        
        # Calculate erosion radius for each mask
        total_face_areas = torch.sum(inv_mask_erosion.float(), dim=(2, 3))  # (N,)
        w_edges = (total_face_areas**0.5 // 20).long()
        erosion_radii = w_edges * 2
        
        # Apply face pasting
        inv_mask_erosion_expanded = inv_mask_erosion.expand_as(inv_faces)
        pasted_faces = inv_mask_erosion_expanded * inv_faces
        
        # Create soft masks for blending
        inv_soft_masks = []
        for i in range(batch_size):
            erosion_radius = erosion_radii[i].item()
            if erosion_radius > 0:
                # Create erosion kernel for this specific mask
                erosion_kernel_specific = torch.ones(
                    (erosion_radius, erosion_radius), device=device, dtype=dtype
                )
                inv_mask_center = kornia.morphology.erosion(inv_mask_erosion[i:i+1], erosion_kernel_specific)
                
                # Apply Gaussian blur
                blur_size = w_edges[i].item() * 2 + 1
                sigma = 0.3 * ((blur_size - 1) * 0.5 - 1) + 0.8
                inv_soft_mask = kornia.filters.gaussian_blur2d(
                    inv_mask_center, (blur_size, blur_size), (sigma, sigma)
                ).squeeze(0)
                inv_soft_mask_3d = inv_soft_mask.expand_as(inv_faces[i])
            else:
                # Fallback for very small faces
                inv_soft_mask_3d = inv_mask_erosion[i].expand_as(inv_faces[i])
            
            inv_soft_masks.append(inv_soft_mask_3d)
        
        inv_soft_masks = torch.stack(inv_soft_masks)
        
        # Blend faces with original images
        img_back = inv_soft_masks * pasted_faces + (1 - inv_soft_masks) * input_imgs_chw
        
        # Return in original format (N, H, W, C)
        img_back = rearrange(img_back, "n c h w -> n h w c").contiguous().to(dtype=torch.uint8)
        return img_back

    def transformation_from_points(self, points1, points0, smooth=True, p_bias=None):
        # Handle both numpy arrays and tensors for points0
        if isinstance(points0, np.ndarray):
            points2 = torch.tensor(points0, device=self.device, dtype=torch.float32)
        else:
            points2 = points0.clone()

        # Handle both numpy arrays and tensors for points1
        if isinstance(points1, np.ndarray):
            points1_tensor = torch.tensor(points1, device=self.device, dtype=torch.float32)
        else:
            points1_tensor = points1.clone()

        c1 = torch.mean(points1_tensor, dim=0)
        c2 = torch.mean(points2, dim=0)

        points1_centered = points1_tensor - c1
        points2_centered = points2 - c2

        s1 = torch.std(points1_centered)
        s2 = torch.std(points2_centered)

        points1_normalized = points1_centered / s1
        points2_normalized = points2_centered / s2

        covariance = torch.matmul(points1_normalized.T, points2_normalized)
        U, S, V = torch.svd(covariance.float())

        R = torch.matmul(V, U.T)

        det = torch.det(R.float())
        if det < 0:
            V[:, -1] = -V[:, -1]
            R = torch.matmul(V, U.T)

        sR = (s2 / s1) * R
        T = c2.reshape(2, 1) - (s2 / s1) * torch.matmul(R, c1.reshape(2, 1))

        M = torch.cat((sR, T), dim=1)

        if smooth:
            bias = points2_normalized[2] - points1_normalized[2]
            if p_bias is None:
                p_bias = bias
            else:
                bias = p_bias * 0.2 + bias * 0.8
            p_bias = bias
            M[:, 2] = M[:, 2] + bias

        return M.cpu().numpy(), p_bias
