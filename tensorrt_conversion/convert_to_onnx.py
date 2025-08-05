#!/usr/bin/env python3
"""
LatentSync TensorRT Conversion - ONNX Export Script
Converts VAE, UNet3D, and Audio2Feature models to ONNX format
"""

import os
os.environ["ONNX_DISABLE_EXTERNAL_DATA"] = "1"
import sys
import torch
import numpy as np
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
import argparse
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsync.models.unet import UNet3DConditionModel
from latentsync.whisper.audio2feature import Audio2Feature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UNetOnnxWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
    def forward(self, sample, timestep, encoder_hidden_states):
        # Call the UNet and return only the predicted sample tensor
        output = self.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states)
        # If the UNet returns a ModelOutput, get the 'sample' tensor:
        if hasattr(output, "sample"):
            return output.sample
        else:
            # If return_dict=False was set, output might be a tuple
            return output[0]

class VAEEncoderWithSampling(torch.nn.Module):
    """VAE Encoder wrapper that includes sampling from latent distribution"""
    
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        
    def forward(self, image):
        # Encode to get latent distribution
        vae_output = self.vae.encode(image)
        # Access the latent_dist and sample from it
        latents = vae_output.latent_dist.sample()
        return latents

class LatentSyncONNXExporter:
    def __init__(self, config_path, checkpoint_path, output_dir="tensorrt_models"):
        self.config = OmegaConf.load(config_path)
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check GPU capability for FP16
        self.is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
        self.dtype = torch.float16 if self.is_fp16_supported else torch.float32
        
        logger.info(f"Using dtype: {self.dtype}")
        
    def export_vae_encoder(self):
        """Export VAE Encoder to ONNX with sampling included"""
        logger.info("Exporting VAE Encoder with sampling...")
        
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=self.dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        vae = vae.to("cuda")
        vae.eval()
        
        # Wrap VAE encoder with sampling
        vae_encoder_with_sampling = VAEEncoderWithSampling(vae)
        vae_encoder_with_sampling.eval()
        
        # Sample input (batch_size, channels, height, width)
        resolution = self.config.data.resolution
        sample_input = torch.randn(1, 3, resolution, resolution, dtype=self.dtype, device="cuda")
        
        # Test the wrapper first
        logger.info("Testing VAE encoder with sampling...")
        with torch.no_grad():
            test_output = vae_encoder_with_sampling(sample_input)
            logger.info(f"Test output shape: {test_output.shape}")
            logger.info(f"Expected shape: [1, 4, {resolution//8}, {resolution//8}]")
        
        # Export encoder with sampling
        with torch.no_grad():
            torch.onnx.export(
                vae_encoder_with_sampling,
                sample_input,
                str(self.output_dir / "vae_encoder.onnx"),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=["image"],
                output_names=["latent"],
                dynamic_axes={
                    "image": {0: "batch_size"},
                    "latent": {0: "batch_size"}
                }
            )
        
        logger.info("VAE Encoder with sampling exported successfully")
        return str(self.output_dir / "vae_encoder.onnx")
    
    def export_vae_decoder(self):
        """Export VAE Decoder to ONNX"""
        logger.info("Exporting VAE Decoder...")
        
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=self.dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        vae = vae.to("cuda")
        vae.eval()
        
        # Sample input for decoder (batch_size, latent_channels, latent_height, latent_width)
        resolution = self.config.data.resolution
        latent_resolution = resolution // 8  # VAE downsamples by 8x
        sample_input = torch.randn(1, 4, latent_resolution, latent_resolution, dtype=self.dtype, device="cuda")
        
        # Export decoder
        with torch.no_grad():
            torch.onnx.export(
                vae.decoder,
                sample_input,
                str(self.output_dir / "vae_decoder.onnx"),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=["latent"],
                output_names=["image"],
                dynamic_axes={
                    "latent": {0: "batch_size"},
                    "image": {0: "batch_size"}
                }
            )
        
        logger.info("VAE Decoder exported successfully")
        return str(self.output_dir / "vae_decoder.onnx")
    
    def export_unet3d_old(self):
        """Export UNet3D to ONNX"""
        logger.info("Exporting UNet3D...")
        
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(self.config.model),
            self.checkpoint_path,
            device="cpu",
        )
        unet = unet.to(dtype=torch.float16, device="cuda")
        unet.eval()

        # recommended 
        # unet = unet.to("cpu")  # move to CPU for export (optional but recommended)
        # unet = unet.to(torch.float32)

        
        dummy_sample = torch.randn(2, 13, 16, 64, 64, dtype=torch.float16, device="cuda")
        dummy_timestep = torch.tensor(951, dtype=torch.int64, device="cuda")  #  using an example timestep 951 as scalar
        dummy_condition = torch.randn(32, 50, 384, dtype=torch.float16, device="cuda") # (audio embeds)

        model_for_onnx = UNetOnnxWrapper(unet)
        model_for_onnx.eval() 

        torch.onnx.export(
            model_for_onnx,                           # model to export (the wrapper)
            (dummy_sample, dummy_timestep, dummy_condition),  # inputs as a tuple
            str(self.output_dir / "unet3d.onnx"),                            # output file name
            opset_version=17,                        
            input_names=["latent_sample", "timestep", "encoder_hidden_states"],
            output_names=["predicted_noise"],
            external_data=False,
        )
        
        logger.info("UNet3D exported successfully")
        return str(self.output_dir / "unet3d.onnx")


    def export_audio_encoder(self):
        """Export Audio2Feature to ONNX"""
        logger.info("Exporting Audio Encoder...")
        
        # Determine whisper model path based on cross_attention_dim
        if self.config.model.cross_attention_dim == 768:
            whisper_model_path = "checkpoints/whisper/small.pt"
        elif self.config.model.cross_attention_dim == 384:
            whisper_model_path = "checkpoints/whisper/tiny.pt"
        else:
            raise NotImplementedError("cross_attention_dim must be 768 or 384")
        
        audio_encoder = Audio2Feature(
            model_path=whisper_model_path,
            device="cuda",
            num_frames=self.config.data.num_frames,
            audio_feat_length=self.config.data.audio_feat_length,
        )
        
        # Create a wrapper for the whisper model to export just the encoder part
        class WhisperEncoderWrapper(torch.nn.Module):
            def __init__(self, whisper_model):
                super().__init__()
                self.encoder = whisper_model.encoder
                
            def forward(self, mel_spectrogram):
                return self.encoder(mel_spectrogram)
        
        wrapper = WhisperEncoderWrapper(audio_encoder.model)
        wrapper.eval()
        
        # Sample input: mel spectrogram (batch_size, n_mels, time_steps)
        sample_input = torch.randn(1, 80, 3000, dtype=self.dtype, device="cuda")
        
        # Export audio encoder
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                sample_input,
                str(self.output_dir / "audio_encoder.onnx"),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=["mel_spectrogram"],
                output_names=["audio_features"],
                dynamic_axes={
                    "mel_spectrogram": {0: "batch_size", 2: "time_steps"},
                    "audio_features": {0: "batch_size", 2: "time_steps"}
                }
            )
        
        logger.info("Audio Encoder exported successfully")
        return str(self.output_dir / "audio_encoder.onnx")
    
    def export_all(self):
        """Export all models to ONNX"""
        logger.info("Starting ONNX export for all models...")
        
        exported_models = {}
        
        try:
            exported_models["vae_encoder"] = self.export_vae_encoder()
            exported_models["vae_decoder"] = self.export_vae_decoder()
            exported_models["unet3d"] = self.export_unet3d()
            exported_models["audio_encoder"] = self.export_audio_encoder()
            
            logger.info("All models exported successfully!")
            logger.info(f"Exported models saved to: {self.output_dir}")
            
            for name, path in exported_models.items():
                logger.info(f"  {name}: {path}")
                
            return exported_models
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Export LatentSync models to ONNX")
    parser.add_argument("--config", type=str, default="/home/codeway/LatentSync/configs/unet/stage2_512.yaml",
                       help="Path to model config")
    parser.add_argument("--checkpoint", type=str, default="/home/codeway/LatentSync/checkpoints/latentsync_unet.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="tensorrt_models",
                       help="Output directory for ONNX models")
    parser.add_argument("--model", type=str, choices=["all", "vae_encoder", "vae_decoder", "unet3d", "audio_encoder"],
                       default="unet3d", help="Which model to export")
    
    args = parser.parse_args()
    
    exporter = LatentSyncONNXExporter(args.config, args.checkpoint, args.output_dir)
    
    if args.model == "all":
        exporter.export_all()
    elif args.model == "vae_encoder":
        exporter.export_vae_encoder()
    elif args.model == "vae_decoder":
        exporter.export_vae_decoder()
    elif args.model == "unet3d":
        exporter.export_unet3d_old()
    elif args.model == "audio_encoder":
        exporter.export_audio_encoder()


if __name__ == "__main__":
    main() 