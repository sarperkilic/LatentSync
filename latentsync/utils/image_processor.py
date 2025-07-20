# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from latentsync.utils.util import read_video, write_video
from torchvision import transforms
import cv2
from einops import rearrange
import torch
import numpy as np
from typing import Union
from .affine_transform import AlignRestore
from .face_detector import FaceDetector


def load_fixed_mask(resolution: int, mask_image_path="latentsync/utils/mask.png") -> torch.Tensor:
    mask_image = cv2.imread(mask_image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
    mask_image = rearrange(torch.from_numpy(mask_image), "h w c -> c h w")
    return mask_image


class ImageProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu", mask_image=None):
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)

        self.restorer = AlignRestore(resolution=resolution, device=device)

        if mask_image is None:
            self.mask_image = load_fixed_mask(resolution)
        else:
            self.mask_image = mask_image

        self.mask_image = self.mask_image.to(device='cuda', dtype=torch.float64)

        if device == "cpu":
            self.face_detector = None
        else:
            self.face_detector = FaceDetector(device=device)

    def preload_video_to_gpu(self, video_frames: np.ndarray) -> torch.Tensor:
        """
        Preload video frames to GPU to avoid repeated CPU-GPU transfers
        Args:
            video_frames: numpy array of shape (N, H, W, C)
        Returns:
            GPU tensor of shape (N, H, W, C)
        """
        video_tensor = torch.from_numpy(video_frames).to(device=self.restorer.device)
        return video_tensor

    def affine_transform_batch(self, images: torch.Tensor) -> tuple[torch.Tensor, list, list]:
        """
        Batch version of affine_transform to process multiple frames at once
        Args:
            images: torch.Tensor of shape (N, C, H, W) or (N, H, W, C)
        Returns:
            faces: torch.Tensor of shape (N, C, H, W)
            boxes: list of bounding boxes
            affine_matrices: list of affine matrices
        """
        if self.face_detector is None:
            raise NotImplementedError("Using the CPU for face detection is not supported")
        
        # Ensure correct format (N, H, W, C) for face detection
        if images.dim() == 4 and images.shape[1] == 3:  # N, C, H, W
            images_np = rearrange(images, "n c h w -> n h w c").cpu().numpy()
        else:  # N, H, W, C
            images_np = images.cpu().numpy()
        
        faces_list = []
        boxes_list = []
        affine_matrices_list = []
        
        # Process each frame for face detection (insightface doesn't support batch processing)
        for i, image_np in enumerate(images_np):
            bbox, landmark_2d_106 = self.face_detector(image_np)
            if bbox is None or landmark_2d_106 is None:
                raise RuntimeError(f"Face not detected in frame {i}")

            pt_left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)
            pt_right_eye = np.mean(landmark_2d_106[101:106], axis=0)
            pt_nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)

            landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])

            # Get corresponding GPU tensor
            if images.dim() == 4 and images.shape[1] == 3:  # N, C, H, W
                image_gpu = images[i]
            else:  # N, H, W, C
                image_gpu = rearrange(images[i], "h w c -> c h w")

            face, affine_matrix = self.restorer.align_warp_face_gpu(image_gpu, landmarks3=landmarks3, smooth=True)
            box = [0, 0, face.shape[2], face.shape[1]]
            
            # Resize on GPU
            face = torch.nn.functional.interpolate(
                face.unsqueeze(0), 
                size=(self.resolution, self.resolution), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            faces_list.append(face)
            boxes_list.append(box)
            affine_matrices_list.append(affine_matrix)
        
        return torch.stack(faces_list), boxes_list, affine_matrices_list

    def affine_transform(self, image: torch.Tensor) -> tuple[torch.Tensor, list, torch.Tensor]:
        if self.face_detector is None:
            raise NotImplementedError("Using the CPU for face detection is not supported")
        
        # Convert torch.Tensor to numpy for face detection (required by insightface)
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:  # C, H, W
                image_np = rearrange(image, "c h w -> h w c").cpu().numpy()
            else:  # H, W, C
                image_np = image.cpu().numpy()
        else:
            image_np = image
            
        bbox, landmark_2d_106 = self.face_detector(image_np)
        if bbox is None:
            raise RuntimeError("Face not detected")

        pt_left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)  # left eyebrow center
        pt_right_eye = np.mean(landmark_2d_106[101:106], axis=0)  # right eyebrow center
        pt_nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)  # nose center

        landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])

        # Keep image on GPU if it's already a tensor, otherwise convert
        if isinstance(image, torch.Tensor):
            image_gpu = image
        else:
            image_gpu = torch.from_numpy(image).to(device=self.restorer.device, dtype=self.restorer.dtype)
            if image_gpu.dim() == 3 and image_gpu.shape[2] == 3:  # H, W, C
                image_gpu = rearrange(image_gpu, "h w c -> c h w")

        face, affine_matrix = self.restorer.align_warp_face_gpu(image_gpu, landmarks3=landmarks3, smooth=True)
        box = [0, 0, face.shape[2], face.shape[1]]  # x1, y1, x2, y2 (C, H, W format)
        
        # Resize on GPU using torchvision
        face = torch.nn.functional.interpolate(
            face.unsqueeze(0), 
            size=(self.resolution, self.resolution), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        return face, box, affine_matrix

    def preprocess_fixed_mask_image(self, image: torch.Tensor, affine_transform=False):
        if affine_transform:
            image, _, _ = self.affine_transform(image)
        else:
            image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)  # shape become 3,512,512
        masked_pixel_values = pixel_values * self.mask_image
        return pixel_values, masked_pixel_values, self.mask_image[0:1]

    def prepare_masks_and_masked_images_batch(self, images: Union[torch.Tensor, np.ndarray], affine_transform=False):
        """
        Optimized batch version that processes all images at once instead of one by one
        Args:
            images: torch.Tensor of shape (N, C, H, W) or (N, H, W, C)
            affine_transform: whether to apply affine transform
        Returns:
            pixel_values: torch.Tensor of shape (N, C, H, W)
            masked_pixel_values: torch.Tensor of shape (N, C, H, W)
            masks: torch.Tensor of shape (N, 1, H, W)
        """
        # Ensure images are torch.Tensor and in correct format (N, C, H, W)
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        
        if images.dim() == 4 and images.shape[3] == 3:  # N, H, W, C
            images = rearrange(images, "n h w c -> n c h w")
        
        # Move to GPU if not already there
        if not images.is_cuda:
            images = images.to(device=self.restorer.device)
        
        # Apply affine transform if needed
        if affine_transform:
            # For affine transform, we need to process individually due to face detection
            # But we can optimize the post-processing
            processed_images = []
            for i in range(len(images)):
                image, _, _ = self.affine_transform(images[i])
                processed_images.append(image)
            images = torch.stack(processed_images)
        else:
            # Batch resize all images at once
            images = self.resize(images)
        
        # Batch normalize all images at once
        pixel_values = self.normalize(images / 255.0)  # shape: (N, 3, 512, 512)
        
        # Batch mask multiplication - much faster than individual processing
        # Expand mask to match batch size
        batch_size = pixel_values.shape[0]
        mask_expanded = self.mask_image.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Apply mask to all images at once
        masked_pixel_values = pixel_values * mask_expanded
        
        # Create mask tensor for all images
        masks = mask_expanded[:, 0:1, :, :]  # shape: (N, 1, H, W)
        
        return pixel_values, masked_pixel_values, masks

    def prepare_masks_and_masked_images(self, images: Union[torch.Tensor, np.ndarray], affine_transform=False):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")

        results = [self.preprocess_fixed_mask_image(image, affine_transform=affine_transform) for image in images]

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values


class VideoProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu"):
        self.image_processor = ImageProcessor(resolution, device)

    def affine_transform_video(self, video_path):
        video_frames = read_video(video_path, change_fps=False)
        results = []
        for frame in video_frames:
            frame, _, _ = self.image_processor.affine_transform(frame)
            results.append(frame)
        results = torch.stack(results)

        results = rearrange(results, "f c h w -> f h w c").numpy()
        return results


if __name__ == "__main__":
    video_processor = VideoProcessor(256, "cuda")
    video_frames = video_processor.affine_transform_video("assets/demo2_video.mp4")
    write_video("output.mp4", video_frames, fps=25)
