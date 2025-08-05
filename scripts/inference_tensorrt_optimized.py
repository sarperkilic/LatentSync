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

import sys
import argparse
import os
import time
import logging
import numpy as np
from omegaconf import OmegaConf
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from diffusers import AutoencoderKL, DDIMScheduler
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latentsync.models.unet import UNet3DConditionModel


from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
from DeepCache import DeepCacheSDHelper
from PIL import Image
from einops import rearrange

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorRTUNet3DEngine:
    """TensorRT Engine wrapper specifically for UNet3D with multiple inputs"""
    
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()
        
        self._load_engine()
        self._allocate_buffers()
    
    def _load_engine(self):
        """Load TensorRT engine from file"""
        try:
            logger.info(f"Loading UNet3D TensorRT engine from {self.engine_path}")
            
            if not os.path.exists(self.engine_path):
                raise RuntimeError(f"TensorRT engine file not found: {self.engine_path}")
            
            file_size = os.path.getsize(self.engine_path)
            logger.info(f"Engine file size: {file_size / (1024*1024):.1f} MB")
            
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
            
            logger.info(f"Read {len(engine_data)} bytes from engine file")
            
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            
            if self.engine is None:
                logger.error("TensorRT engine deserialization failed!")
                logger.error(f"TensorRT version: {trt.__version__}")
                logger.error(f"CUDA device: {torch.cuda.get_device_name()}")
                logger.error(f"CUDA version: {torch.version.cuda}")
                raise RuntimeError(f"Failed to deserialize TensorRT engine from {self.engine_path}")
            
            logger.info("UNet3D TensorRT engine deserialized successfully")
            self.context = self.engine.create_execution_context()
            
            if self.context is None:
                raise RuntimeError("Failed to create TensorRT execution context")
            
            logger.info("UNet3D TensorRT execution context created successfully")
            
        except Exception as e:
            logger.error(f"Error loading UNet3D TensorRT engine: {e}")
            raise
    
    def _allocate_buffers(self):
        """Allocate GPU buffers for inputs and outputs"""
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            
            # Handle dynamic shapes by setting concrete shapes
            if is_input:
                if "latent_sample" in tensor_name:
                    concrete_shape = (2, 13, 16, 64, 64)  # Expected shape for latent_sample
                elif "timestep" in tensor_name:
                    concrete_shape = ()  # Scalar for timestep
                elif "encoder_hidden_states" in tensor_name:
                    concrete_shape = (32, 50, 384)  # Expected shape for encoder_hidden_states
                else:
                    concrete_shape = tuple([2 if d == -1 else d for d in tensor_shape])
                
                logger.info(f"Setting input shape for {tensor_name}: {concrete_shape}")
                self.context.set_input_shape(tensor_name, concrete_shape)
                tensor_shape = concrete_shape
            else:
                # For outputs, get the shape after input shapes are set
                tensor_shape = self.context.get_tensor_shape(tensor_name)
                logger.info(f"Output shape for {tensor_name}: {tensor_shape}")
            
            tensor_size = trt.volume(tensor_shape)
            dtype = trt.nptype(tensor_dtype)
            
            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(tensor_size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            if is_input:
                self.inputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'shape': tensor_shape,
                    'dtype': dtype,
                    'name': tensor_name
                })
            else:
                self.outputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'shape': tensor_shape,
                    'dtype': dtype,
                    'name': tensor_name
                })
            
            # Set tensor address for execution context
            self.context.set_tensor_address(tensor_name, int(device_mem))
    
    def infer(self, latent_sample, timestep, encoder_hidden_states):
        """Run inference with UNet3D TensorRT engine"""
        # Find the correct input buffers
        latent_input = None
        timestep_input = None
        encoder_input = None
        
        for input_buf in self.inputs:
            if "latent_sample" in input_buf['name']:
                latent_input = input_buf
            elif "timestep" in input_buf['name']:
                timestep_input = input_buf
            elif "encoder_hidden_states" in input_buf['name']:
                encoder_input = input_buf
        
        if not all([latent_input, timestep_input, encoder_input]):
            raise ValueError("Could not find all required input tensors")
        
        # Prepare and copy latent_sample
        latent_np = latent_sample.cpu().numpy().astype(latent_input['dtype'])
        logger.debug(f"Original latent_sample shape: {latent_np.shape}")
        logger.debug(f"Expected latent_input shape: {latent_input['shape']}")
        
        if latent_np.shape != latent_input['shape']:
            logger.info(f"Reshaping latent_sample from {latent_np.shape} to {latent_input['shape']}")
            latent_np = latent_np.reshape(latent_input['shape'])
        
        np.copyto(latent_input['host'], latent_np.ravel())
        cuda.memcpy_htod_async(latent_input['device'], latent_input['host'], self.stream)
        
        # Prepare and copy timestep
        timestep_np = timestep.cpu().numpy().astype(timestep_input['dtype'])
        if timestep_np.shape != timestep_input['shape']:
            logger.info(f"Reshaping timestep from {timestep_np.shape} to {timestep_input['shape']}")
            timestep_np = timestep_np.reshape(timestep_input['shape'])
        
        np.copyto(timestep_input['host'], timestep_np.ravel())
        cuda.memcpy_htod_async(timestep_input['device'], timestep_input['host'], self.stream)
        
        # Prepare and copy encoder_hidden_states
        encoder_np = encoder_hidden_states.cpu().numpy().astype(encoder_input['dtype'])
        if encoder_np.shape != encoder_input['shape']:
            logger.info(f"Reshaping encoder_hidden_states from {encoder_np.shape} to {encoder_input['shape']}")
            encoder_np = encoder_np.reshape(encoder_input['shape'])
        
        np.copyto(encoder_input['host'], encoder_np.ravel())
        cuda.memcpy_htod_async(encoder_input['device'], encoder_input['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Copy output data from GPU
        outputs = []
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
        
        self.stream.synchronize()
        
        # Convert outputs to tensors
        for output in self.outputs:
            output_array = output['host'].reshape(output['shape'])
            output_tensor = torch.from_numpy(output_array.copy()).cuda()
            outputs.append(output_tensor)
        
        return outputs[0] if len(outputs) == 1 else outputs

class TensorRTEngine:
    """TensorRT Engine wrapper for inference"""
    
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()
        
        self._load_engine()
        self._allocate_buffers()
    
    def _load_engine(self):
        """Load TensorRT engine from file"""
        try:
            logger.info(f"Loading TensorRT engine from {self.engine_path}")
            
            # Check if file exists and is readable
            if not os.path.exists(self.engine_path):
                raise RuntimeError(f"TensorRT engine file not found: {self.engine_path}")
            
            # Check file size
            file_size = os.path.getsize(self.engine_path)
            logger.info(f"Engine file size: {file_size / (1024*1024):.1f} MB")
            
            # Read engine data
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
            
            logger.info(f"Read {len(engine_data)} bytes from engine file")
            
            # Try to deserialize
            logger.info("Attempting to deserialize TensorRT engine...")
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            
            if self.engine is None:
                # Print TensorRT version info for debugging
                logger.error("TensorRT engine deserialization failed!")
                logger.error(f"TensorRT version: {trt.__version__}")
                logger.error(f"CUDA device: {torch.cuda.get_device_name()}")
                logger.error(f"CUDA version: {torch.version.cuda}")
                
                raise RuntimeError(
                    f"Failed to deserialize TensorRT engine from {self.engine_path}. "
                    "This usually indicates a TensorRT version mismatch. "
                    "Please rebuild the TensorRT engines with the current TensorRT version."
                )
            
            logger.info("TensorRT engine deserialized successfully")
            self.context = self.engine.create_execution_context()
            
            if self.context is None:
                raise RuntimeError("Failed to create TensorRT execution context")
            
            logger.info("TensorRT execution context created successfully")
            
        except Exception as e:
            logger.error(f"Error loading TensorRT engine: {e}")
            raise
    
    def _allocate_buffers(self):
        """Allocate GPU buffers for inputs and outputs"""
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            
            # Handle dynamic shapes by setting a concrete shape
            if is_input and tensor_shape[0] == -1:
                # Set batch size to 16 for inputs with dynamic batch dimension (updated from 1 to 16)
                if "image" in tensor_name:
                    concrete_shape = (16, 3, 512, 512)  # Updated to support batch size 16
                elif "latent" in tensor_name:
                    concrete_shape = (16, 4, 64, 64)    # Updated to support batch size 16
                else:
                    concrete_shape = tuple([16 if d == -1 else d for d in tensor_shape])  # Updated to 16
                
                logger.info(f"Setting input shape for {tensor_name}: {concrete_shape}")
                self.context.set_input_shape(tensor_name, concrete_shape)
                tensor_shape = concrete_shape
            elif not is_input and tensor_shape[0] == -1:
                # For outputs, get the shape after input shapes are set
                tensor_shape = self.context.get_tensor_shape(tensor_name)
                logger.info(f"Dynamic output shape for {tensor_name}: {tensor_shape}")
            
            tensor_size = trt.volume(tensor_shape)
            dtype = trt.nptype(tensor_dtype)
            
            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(tensor_size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            if is_input:
                self.inputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'shape': tensor_shape,
                    'dtype': dtype,
                    'name': tensor_name
                })
            else:
                self.outputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'shape': tensor_shape,
                    'dtype': dtype,
                    'name': tensor_name
                })
            
            # Set tensor address for execution context
            self.context.set_tensor_address(tensor_name, int(device_mem))
    
    def infer(self, input_data):
        """Run inference with TensorRT engine"""
        # Copy input data to GPU
        if len(self.inputs) != 1:
            raise ValueError(f"Expected 1 input, got {len(self.inputs)}")
        
        input_tensor = input_data.cpu().numpy().astype(self.inputs[0]['dtype'])
        
        # Ensure input matches expected shape
        expected_shape = self.inputs[0]['shape']
        if input_tensor.shape != expected_shape:
            logger.info(f"Reshaping input from {input_tensor.shape} to {expected_shape}")
            # If batch size differs, we need to set the input shape
            if input_tensor.shape[0] != expected_shape[0]:
                new_shape = (input_tensor.shape[0],) + expected_shape[1:]
                self.context.set_input_shape(self.inputs[0]['name'], new_shape)
                # Reallocate buffers for new shape
                self._reallocate_buffers_for_shape(new_shape)
        
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Run inference using new TensorRT 10.x API
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Copy output data from GPU
        outputs = []
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
        
        self.stream.synchronize()
        
        # Convert outputs to tensors
        for output in self.outputs:
            output_array = output['host'].reshape(output['shape'])
            output_tensor = torch.from_numpy(output_array.copy()).cuda()
            outputs.append(output_tensor)
        
        return outputs[0] if len(outputs) == 1 else outputs
    
    def _reallocate_buffers_for_shape(self, new_shape):
        """Reallocate buffers when input shape changes"""
        logger.info(f"Reallocating buffers for new shape: {new_shape}")
        
        # Clear existing buffers
        self.inputs = []
        self.outputs = []
        
        # Reallocate with new shape
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            
            if is_input:
                tensor_shape = new_shape
            else:
                # Get output shape after input shape is set
                tensor_shape = self.context.get_tensor_shape(tensor_name)
            
            tensor_size = trt.volume(tensor_shape)
            dtype = trt.nptype(tensor_dtype)
            
            # Allocate new memory
            host_mem = cuda.pagelocked_empty(tensor_size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            if is_input:
                self.inputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'shape': tensor_shape,
                    'dtype': dtype,
                    'name': tensor_name
                })
            else:
                self.outputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'shape': tensor_shape,
                    'dtype': dtype,
                    'name': tensor_name
                })
            
            # Set tensor address
            self.context.set_tensor_address(tensor_name, int(device_mem))

class TensorRTUNetWrapper:
    """
    Wrapper for TensorRT UNet3D that provides the same interface as the original UNet
    """
    def __init__(self, trt_engine):
        self.trt_engine = trt_engine
        # Add required attributes that the pipeline expects
        self.config = type('MockConfig', (), {
            'sample_size': 64,  # Default value, adjust if needed
            'in_channels': 4,
            'out_channels': 4,
            'down_block_types': ['CrossAttnDownBlock2D'] * 4,
            'up_block_types': ['CrossAttnUpBlock2D'] * 4,
            'block_out_channels': [320, 640, 1280, 1280],
            'layers_per_block': 2,
            'cross_attention_dim': 768,
            'attention_head_dim': 8,
            'use_linear_projection': True,
        })()
        self.add_audio_layer = False  # Default value
    
    def forward(self, sample, timestep, encoder_hidden_states=None, return_dict=True):
        """
        Forward pass that calls the TensorRT UNet3D engine
        """
        # Handle variable frame counts - the TensorRT engine expects 16 frames
        # but we might have fewer frames at the end of the sequence
        batch_size, channels, num_frames, height, width = sample.shape
        
        if num_frames < 16:
            padding_frames = 16 - num_frames
            last_frame = sample[:, :, -1:, :, :]  # Take the last frame
            padding = last_frame.repeat(1, 1, padding_frames, 1, 1)  # Repeat it
            sample = torch.cat([sample, padding], dim=2)  # Concatenate
            
            if encoder_hidden_states.shape[0] < 32:

                remaining = 32 - encoder_hidden_states.shape[0]
                last_embedding = encoder_hidden_states[-1:].repeat(remaining, 1, 1)
                encoder_hidden_states = torch.cat([encoder_hidden_states, last_embedding], dim=0)
   
        elif num_frames > 16:
            # Truncate to 16 frames
            sample = sample[:, :, :16, :, :]
            
            # Also adjust encoder_hidden_states if needed
            if encoder_hidden_states is not None:
                # Ensure the batch size matches
                if encoder_hidden_states.shape[0] < batch_size:
                    repeat_factor = batch_size // encoder_hidden_states.shape[0]
                    encoder_hidden_states = encoder_hidden_states.repeat(repeat_factor, 1, 1)
                    if encoder_hidden_states.shape[0] < batch_size:
                        remaining = batch_size - encoder_hidden_states.shape[0]
                        last_embedding = encoder_hidden_states[-1:].repeat(remaining, 1, 1)
                        encoder_hidden_states = torch.cat([encoder_hidden_states, last_embedding], dim=0)
                
                # Ensure the sequence length is 50
                if encoder_hidden_states.shape[1] != 50:
                    if encoder_hidden_states.shape[1] < 50:
                        pad_size = 50 - encoder_hidden_states.shape[1]
                        last_embedding = encoder_hidden_states[:, -1:, :]
                        padding = last_embedding.repeat(1, pad_size, 1)
                        encoder_hidden_states = torch.cat([encoder_hidden_states, padding], dim=1)
                    else:
                        encoder_hidden_states = encoder_hidden_states[:, :50, :]
                
                # Ensure the embedding dimension is 384
                if encoder_hidden_states.shape[2] != 384:
                    if encoder_hidden_states.shape[2] < 384:
                        pad_size = 384 - encoder_hidden_states.shape[2]
                        padding = torch.zeros(encoder_hidden_states.shape[0], encoder_hidden_states.shape[1], pad_size, 
                                           device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
                        encoder_hidden_states = torch.cat([encoder_hidden_states, padding], dim=2)
                    else:
                        encoder_hidden_states = encoder_hidden_states[:, :, :384]
    
        # Call TensorRT engine with all three inputs
        try:
            output = self.trt_engine.infer(sample, timestep, encoder_hidden_states) # output torch.Size([2, 4, 16, 64, 64])
            
        except Exception as e:
            logger.error(f"TensorRT UNet3D inference failed: {e}")
            logger.error(f"Input shapes - sample: {sample.shape}, timestep: {timestep.shape}, encoder_hidden_states: {encoder_hidden_states.shape if encoder_hidden_states is not None else 'None'}")
            raise
        
        # Return a mock object with .sample attribute to match expected interface
        return type('MockReturn', (), {'sample': output})()
    
    def __call__(self, sample, timestep, encoder_hidden_states=None, return_dict=True):
        """
        Call method that matches the original UNet interface
        """
        return self.forward(sample, timestep, encoder_hidden_states, return_dict)
    
    def to(self, device):
        """Mock device transfer - TensorRT models are already on the correct device"""
        return self
    
    def eval(self):
        """Mock eval mode - TensorRT models are already in eval mode"""
        return self

class TensorRTLipsyncPipeline(LipsyncPipeline):
    """
    TensorRT-optimized version of LipsyncPipeline
    Uses TensorRT engines for VAE encoder/decoder and UNet3D
    """
    
    def __init__(self, vae, audio_encoder, scheduler, config, 
                 vae_encoder_trt_path, vae_decoder_trt_path, unet_trt_path, unet_raw,
                 mask_image_path="latentsync/utils/mask.png"):
        
        # Load TensorRT UNet3D engine
        logger.info(f"Loading TensorRT UNet3D engine from: {unet_trt_path}")
        if not os.path.exists(unet_trt_path):
            raise FileNotFoundError(f"TensorRT UNet3D engine not found at: {unet_trt_path}")
        
        unet_trt_engine = TensorRTUNet3DEngine(unet_trt_path)
        
        # Create wrapper for TensorRT UNet3D
        unet_wrapper = TensorRTUNetWrapper(unet_trt_engine)
        
        # Load TensorRT engines
        logger.info(f"🔧 Loading VAE encoder TensorRT engine from: {vae_encoder_trt_path}")
        self.vae_encoder_engine = TensorRTEngine(vae_encoder_trt_path)
        
        logger.info(f"🔧 Loading VAE decoder TensorRT engine from: {vae_decoder_trt_path}")
        self.vae_decoder_engine = TensorRTEngine(vae_decoder_trt_path)
        
        # Initialize parent pipeline with wrapped UNet
        super().__init__(
            vae=vae,
            audio_encoder=audio_encoder,
            unet= unet_wrapper,  # Use the TensorRT wrapper instead of TorchScript
            scheduler=scheduler,
            mask_image_path=mask_image_path,
            config=config
        )
        
        # Load mask image
        if os.path.exists(mask_image_path):
            self.mask_image = Image.open(mask_image_path).convert("RGB")
        else:
            logger.warning(f"⚠️  Mask image not found at {mask_image_path}, using default")
            self.mask_image = None

    def to(self, device):
        """Move pipeline to device"""
        super().to(device)
        return self

    def decode_latents(self, latents):
        """
        Decode latents using TensorRT VAE decoder
        """
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        
        # Use TensorRT decoder
        decoded_latents = self.vae_decoder_engine.infer(latents)
        return decoded_latents

    def prepare_mask_latents(self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance):
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        
        if masked_image.dtype != dtype:
            masked_image = masked_image.to(dtype=dtype)
        
        # encode the mask image into latents space using TensorRT
        with torch.no_grad():
            masked_image_latents = self.vae_encoder_engine.infer(masked_image)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        if mask.dtype != dtype:
            mask = mask.to(dtype=dtype)

        # assume batch size = 1
        from einops import rearrange
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        if images.dtype != dtype or images.device != device:
            images = images.to(device=device, dtype=dtype)

        with torch.no_grad():
            image_latents = self.vae_encoder_engine.infer(images)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        
        from einops import rearrange
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        """
        Run inference with TensorRT optimizations
        """
        logger.info("Running inference with: TensorRT VAE + TensorRT UNet3D")
        
        start_time = time.time()
        
        # Call the original pipeline (now using the TensorRT UNet wrapper)
        result = super().__call__(*args, **kwargs)
        
        total_time = time.time() - start_time
        logger.info(f"Total inference time: {total_time:.3f}s")
        
        return result

def main(config, args):
    if not os.path.exists(args.video_path):
        raise RuntimeError(f"Video path '{args.video_path}' not found")
    if not os.path.exists(args.audio_path):
        raise RuntimeError(f"Audio path '{args.audio_path}' not found")

    # Check if the GPU supports float16
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")

    # Check TensorRT engine paths - REQUIRED
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vae_encoder_path = os.path.join(base_dir, "tensorrt_conversion/tensorrt_engines/vae_encoder.trt")
    vae_decoder_path = os.path.join(base_dir, "tensorrt_conversion/tensorrt_engines/vae_decoder.trt")
    unet_trt_path = os.path.join(base_dir, "tensorrt_conversion/tensorrt_engines/unet3d_fp16.trt")
    
    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,
        device="cpu",
    )

    unet = unet.to(dtype=dtype, device='cuda')
    vae = vae.to(dtype=dtype, device='cuda')

    # Create TensorRT-optimized pipeline
    pipeline = TensorRTLipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        scheduler=scheduler,
        config=config,
        vae_encoder_trt_path=vae_encoder_path,
        vae_decoder_trt_path=vae_decoder_path,
        unet_trt_path=unet_trt_path,
        unet_raw = unet
    )

    # use DeepCache (if enabled)
    #if args.enable_deepcache:
    #    helper = DeepCacheSDHelper(pipe=pipeline)
    #    helper.set_params(cache_interval=3, cache_branch_id=0)
    #    helper.enable()
    #    logger.info(" DeepCache enabled")

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")

    t0 = time.perf_counter()

    total_generated_frames = pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=None,  # args.video_out_path,
        num_frames=config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
        mask_image_path=config.data.mask_image_path,
        temp_dir=args.temp_dir,
    )

    t1 = time.perf_counter()
    elapsed = t1 - t0
    fps = total_generated_frames / elapsed
    print(f"\n✓ elapsed: {elapsed:0.2f}s  |  FPS: {fps:0.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorRT-optimized LatentSync inference")
    parser.add_argument("--unet_config_path", type=str, default="/home/codeway/LatentSync/configs/unet/stage2_512.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, default="/home/codeway/LatentSync/checkpoints/latentsync_unet.pt")
    parser.add_argument("--video_path", type=str, default="/home/codeway/LatentSync/assets/demo1_video.mp4")
    parser.add_argument("--audio_path", type=str, default="/home/codeway/LatentSync/assets/demo1_audio.wav")
    parser.add_argument("--video_out_path", type=str, default="video_out_tensorrt.mp4")
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_false")
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)

    main(config, args) 