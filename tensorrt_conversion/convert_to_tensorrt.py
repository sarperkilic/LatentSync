#!/usr/bin/env python3
"""
LatentSync TensorRT Conversion - TensorRT Engine Builder
Converts ONNX models to optimized TensorRT engines
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
import tensorrt as trt
import numpy as np
import torch

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import torch_tensorrt
try:
    import torch_tensorrt
    TORCH_TENSORRT_AVAILABLE = True
except ImportError:
    TORCH_TENSORRT_AVAILABLE = False
    logger.warning("torch_tensorrt not available. Install with: pip install torch-tensorrt")


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 Calibrator for UNet3D model"""
    
    def __init__(self, calibration_data, cache_file="calibration.cache"):
        super().__init__()
        self.calibration_data = calibration_data
        self.cache_file = cache_file
        self.current_index = 0
        
    def get_batch_size(self):
        return 1
    
    def get_batch(self, names):
        if self.current_index >= len(self.calibration_data):
            return None
        
        batch_data = self.calibration_data[self.current_index]
        self.current_index += 1
        
        # Return calibration data as numpy arrays
        return [batch_data[name] for name in names]
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


class LatentSyncTensorRTConverter:
    def __init__(self, tensorrt_path="/home/codeway/srper/TensorRT-10.13.0.35", 
                 onnx_dir="tensorrt_models", output_dir="tensorrt_engines"):
        self.tensorrt_path = Path(tensorrt_path)
        self.onnx_dir = Path(onnx_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add TensorRT to Python path
        tensorrt_python_path = self.tensorrt_path / "python"
        if tensorrt_python_path.exists():
            sys.path.insert(0, str(tensorrt_python_path))
        
        # Set CUDA device
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            logger.info(f"Using CUDA device: {self.device}")
        else:
            raise RuntimeError("CUDA not available")
    
    def create_optimization_profile(self, builder, network, input_specs):
        """Create optimization profile for dynamic shapes"""
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        
        for input_name, (min_shape, opt_shape, max_shape) in input_specs.items():
            # Skip optimization profile for inputs with fixed dimensions (all same)
            if min_shape == opt_shape == max_shape:
                logger.info(f"Input {input_name}: fixed shape {min_shape} (no optimization profile)")
                continue
                
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            logger.info(f"Input {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
        
        config.add_optimization_profile(profile)
        return config
    
    def build_engine(self, onnx_path, engine_path, input_specs, 
                    max_workspace_size=4 << 30,  # 4GB
                    fp16_mode=True,
                    int8_mode=False,
                    calibration_data=None):
        """Build TensorRT engine from ONNX model with INT8/FP16 support"""
        logger.info(f"Building engine for {onnx_path}")
        
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX model with external data support
        onnx_path = Path(onnx_path)
        
        # Change to the ONNX directory so TensorRT can find external data files
        original_cwd = os.getcwd()
        try:
            os.chdir(onnx_path.parent)
            logger.info(f"Changed working directory to: {onnx_path.parent}")
            
            with open(onnx_path.name, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return None
        except Exception as e:
            logger.error(f"Error parsing ONNX model: {e}")
            return None
        finally:
            os.chdir(original_cwd)
        
        # Create builder config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
        
        # Set up optimization profile only if input_specs is not empty (for dynamic inputs)
        if input_specs:
            profile = builder.create_optimization_profile()
            for input_name, (min_shape, opt_shape, max_shape) in input_specs.items():
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                logger.info(f"Input {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
            config.add_optimization_profile(profile)
        else:
            logger.info("No optimization profiles needed for static input shapes")
        
        # Enable optimizations
        if fp16_mode and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 mode enabled")
        
        if int8_mode and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            logger.info("INT8 mode enabled")
            
            # Set up INT8 calibration if calibration data provided
            if calibration_data is not None:
                config.int8_calibrator = calibration_data
                logger.info("INT8 calibration enabled")
        
        # Build engine
        logger.info("Building TensorRT engine... This may take a while.")
        start_time = time.time()
        
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            logger.error("Failed to build engine")
            return None
        
        build_time = time.time() - start_time
        logger.info(f"Engine built successfully in {build_time:.2f} seconds")
        
        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        logger.info(f"Engine saved to {engine_path}")
        
        # Deserialize to return engine object
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine
    
    def convert_vae_encoder(self):
        """Convert VAE Encoder ONNX to TensorRT"""
        logger.info("Converting VAE Encoder to TensorRT...")
        
        onnx_path = self.onnx_dir / "vae_encoder.onnx"
        engine_path = self.output_dir / "vae_encoder.trt"
        
        # Define input specifications for dynamic batching (resolution is fixed in ONNX)
        input_specs = {
            "image": (
                (1, 3, 512, 512),    # min: single batch
                (1, 3, 512, 512),    # opt: target resolution  
                (16, 3, 512, 512)     # max: larger batch size
            )
        }
        
        return self.build_engine(onnx_path, engine_path, input_specs)
    
    def convert_vae_decoder(self):
        """Convert VAE Decoder ONNX to TensorRT"""
        logger.info("Converting VAE Decoder to TensorRT...")
        
        onnx_path = self.onnx_dir / "vae_decoder.onnx"
        engine_path = self.output_dir / "vae_decoder.trt"
        
        # Define input specifications for latent space (fixed resolution in ONNX)
        input_specs = {
            "latent": (
                (1, 4, 64, 64),      # min: single batch
                (1, 4, 64, 64),      # opt: 512x512 image -> 64x64 latent
                (16, 4, 64, 64)       # max: larger batch size
            )
        }
        
        return self.build_engine(onnx_path, engine_path, input_specs)
    
    def convert_unet3d(self):
        """Convert UNet3D ONNX to TensorRT with correct input specifications"""
        logger.info("Converting UNet3D to TensorRT...")
        
        onnx_path = self.onnx_dir / "unet3d.onnx"
        engine_path = self.output_dir / "unet3d.trt"
        
        if not onnx_path.exists():
            logger.error(f"ONNX file not found: {onnx_path}")
            logger.info("Please run ONNX export first: python convert_to_onnx.py --model unet3d")
            return None
        
        # Define input specifications based on the actual ONNX model
        # The ONNX model was exported with fixed dimensions, so we need to match them
        input_specs = {
            "latent_sample": (
                (1, 13, 16, 64, 64),     # min: small batch
                (2, 13, 16, 64, 64),     # opt: target configuration from pipeline
                (4, 13, 16, 64, 64)      # max: larger batch
            ),
            "timestep": (
                (1,),   # min: scalar timestep
                (2,),   # opt: batch of timesteps  
                (4,)    # max: larger batch
            ),
            "encoder_hidden_states": (
                (1, 50, 384),     # min: small batch
                (32, 50, 384),    # opt: target from pipeline
                (64, 50, 384)     # max: larger batch
            )
        }
        
        # UNet needs more workspace due to complexity
        return self.build_engine(onnx_path, engine_path, input_specs, 
                               max_workspace_size=8 << 30)  # 8GB
    
    def convert_torchscript_to_onnx(self, torchscript_path, onnx_path, input_shapes):
        """Convert TorchScript model to ONNX"""
        logger.info(f"Converting TorchScript {torchscript_path} to ONNX {onnx_path}")
        
        # Load TorchScript model
        model = torch.jit.load(torchscript_path, map_location='cuda')
        model.eval()
        
        # Create dummy inputs based on input shapes
        dummy_inputs = []
        input_names = []
        
        for i, (name, shape) in enumerate(input_shapes.items()):
            if name == "timestep":
                # Use scalar timestep, but make it a tensor of shape () for ONNX
                if shape == ():
                    dummy_input = torch.tensor(100, dtype=torch.long, device='cuda')
                else:
                    dummy_input = torch.tensor([100], dtype=torch.long, device='cuda')
            else:
                dummy_input = torch.randn(shape, dtype=torch.float32, device='cuda')
            dummy_inputs.append(dummy_input)
            input_names.append(name)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            tuple(dummy_inputs),
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=['output'],
            dynamic_axes={name: {0: 'batch_size'} for name in input_names}
        )
        
        logger.info(f"Successfully converted to ONNX: {onnx_path}")
        return onnx_path
    
    def convert_unet3d_torchscript(self):
        """Convert UNet3D TorchScript to TensorRT via ONNX"""
        logger.info("Converting UNet3D TorchScript to TensorRT...")
        
        torchscript_path = self.onnx_dir / "unet3d_torchscript.pt"
        onnx_path = self.onnx_dir / "unet3d.onnx"
        engine_path = self.output_dir / "unet3d.trt"
        
        if not torchscript_path.exists():
            logger.error(f"TorchScript file not found: {torchscript_path}")
            return None
        
        # Test the model first to understand the expected input format
        model = torch.jit.load(str(torchscript_path), map_location='cuda')
        model.eval()
        
        # Define input shapes for ONNX conversion (use optimal shapes)
        input_shapes = {
            "sample": (1, 13, 16, 64, 64),
            "timestep": (1,),  # scalar timestep
            "encoder_hidden_states": (1, 16, 768)
        }
        
        # Test with sample inputs first
        try:
            with torch.no_grad():
                sample_input = torch.randn(1, 13, 16, 64, 64, device='cuda')
                timestep_input = torch.tensor(100, device='cuda', dtype=torch.long)  # Scalar timestep
                encoder_states = torch.randn(1, 16, 768, device='cuda')
                
                # Test the model
                output = model(sample_input, timestep_input, encoder_states)
                logger.info("Model test successful")
                
                # Update input shapes based on working inputs
                input_shapes = {
                    "sample": sample_input.shape,
                    "timestep": (),  # scalar has empty shape
                    "encoder_hidden_states": encoder_states.shape
                }
                
        except Exception as e:
            logger.warning(f"Model test failed: {e}")
            # Fall back to original shapes if test fails
        
        # Convert TorchScript to ONNX first
        self.convert_torchscript_to_onnx(torchscript_path, onnx_path, input_shapes)
        
        # Define input specifications for TensorRT
        input_specs = {
            "sample": (
                (1, 13, 8, 32, 32),     # min: small batch, few frames
                (1, 13, 16, 64, 64),    # opt: target configuration
                (2, 13, 32, 128, 128)   # max: large batch, many frames
            ),
            "timestep": (
                (),   # min: scalar
                (),   # opt: scalar
                ()    # max: scalar
            ),
            "encoder_hidden_states": (
                (1, 8, 384),     # min: few frames, small cross_attention_dim
                (1, 16, 768),    # opt: target frames and cross_attention_dim
                (2, 32, 768)     # max: large batch, many frames
            )
        }
        
        # Build TensorRT engine from ONNX
        return self.build_engine(onnx_path, engine_path, input_specs, 
                               max_workspace_size=8 << 30)  # 8GB
    
    def convert_unet3d_torchrt(self):
        """Convert UNet3D TorchScript to TensorRT directly using Torch-TensorRT"""
        logger.info("Converting UNet3D TorchScript to TensorRT using Torch-TensorRT...")
        
        if not TORCH_TENSORRT_AVAILABLE:
            logger.error("torch_tensorrt not available. Install with: pip install torch-tensorrt")
            return None
        
        torchscript_path = self.onnx_dir / "unet3d_torchscript.pt"
        engine_path = self.output_dir / "unet3d_torchrt.trt"
        
        if not torchscript_path.exists():
            logger.error(f"TorchScript file not found: {torchscript_path}")
            return None
        
        # Load TorchScript model
        model = torch.jit.load(str(torchscript_path), map_location='cuda')
        model.eval()
        
        # Define input specifications for Torch-TensorRT
        inputs = [
            torch_tensorrt.Input(
                min_shape=(1, 13, 8, 32, 32),
                opt_shape=(1, 13, 16, 64, 64),
                max_shape=(2, 13, 32, 128, 128),
                dtype=torch.float32
            ),
            torch_tensorrt.Input(
                min_shape=(1,),
                opt_shape=(1,),
                max_shape=(2,),
                dtype=torch.float32
            ),
            torch_tensorrt.Input(
                min_shape=(1, 8, 384),
                opt_shape=(1, 16, 768),
                max_shape=(2, 32, 768),
                dtype=torch.float32
            )
        ]
        
        # Compile with Torch-TensorRT
        try:
            compiled_model = torch_tensorrt.compile(
                model,
                inputs=inputs,
                enabled_precisions={torch.float16},  # Use FP16
                workspace_size=8 << 30,  # 8GB
                min_block_size=1,
                torch_executed_ops={"aten::softmax"}  # Keep softmax in PyTorch
            )
            
            # Save the compiled model
            torch.jit.save(compiled_model, str(engine_path))
            logger.info(f"Torch-TensorRT engine saved to {engine_path}")
            return compiled_model
            
        except Exception as e:
            logger.error(f"Torch-TensorRT compilation failed: {e}")
            return None
    
    def generate_calibration_data(self, num_samples=100):
        """Generate calibration data for INT8 quantization"""
        logger.info(f"Generating {num_samples} calibration samples...")
        
        calibration_data = []
        
        for i in range(num_samples):
            # Generate sample data matching the pipeline inputs
            latent_sample = torch.randn(2, 13, 16, 64, 64, dtype=torch.float32)
            timestep = torch.randint(0, 1000, (2,), dtype=torch.int64)
            encoder_hidden_states = torch.randn(32, 50, 384, dtype=torch.float32)
            
            # Convert to numpy for TensorRT
            sample_data = {
                "latent_sample": latent_sample.numpy(),
                "timestep": timestep.numpy(),
                "encoder_hidden_states": encoder_hidden_states.numpy()
            }
            
            calibration_data.append(sample_data)
        
        logger.info(f"Generated {len(calibration_data)} calibration samples")
        return calibration_data
    
    def convert_unet3d_int8(self):
        """Convert UNet3D ONNX to TensorRT with INT8 quantization"""
        logger.info("Converting UNet3D to TensorRT with INT8 quantization...")
        
        onnx_path = self.onnx_dir / "unet3d.onnx"
        engine_path = self.output_dir / "unet3d_int8.trt"
        
        if not onnx_path.exists():
            logger.error(f"ONNX file not found: {onnx_path}")
            logger.info("Please run ONNX export first: python convert_to_onnx.py --model unet3d")
            return None
        
        # Generate calibration data
        calibration_data = self.generate_calibration_data(num_samples=100)
        calibrator = Int8Calibrator(calibration_data, cache_file="unet3d_calibration.cache")
        
        # Define input specifications based on lipsync_pipeline.py usage
        input_specs = {
            "latent_sample": (
                (1, 13, 8, 32, 32),     # min: small batch, few frames
                (2, 13, 16, 64, 64),    # opt: target configuration from pipeline
                (4, 13, 32, 128, 128)   # max: large batch, many frames
            ),
            "timestep": (
                (1,),   # min: scalar timestep
                (2,),   # opt: batch of timesteps  
                (4,)    # max: larger batch
            ),
            "encoder_hidden_states": (
                (1, 8, 384),     # min: few frames, small cross_attention_dim
                (32, 50, 384),   # opt: target from pipeline
                (64, 100, 384)   # max: large batch, many frames
            )
        }
        
        # Build INT8 engine
        return self.build_engine(onnx_path, engine_path, input_specs, 
                               max_workspace_size=8 << 30,  # 8GB
                               fp16_mode=False,  # Disable FP16 for INT8
                               int8_mode=True,
                               calibration_data=calibrator)
    
    def convert_unet3d_fp16(self):
        """Convert UNet3D ONNX to TensorRT with FP16 precision"""
        logger.info("Converting UNet3D to TensorRT with FP16 precision...")
        
        onnx_path = self.onnx_dir / "unet3d.onnx"
        engine_path = self.output_dir / "unet3d_fp16.trt"
        
        if not onnx_path.exists():
            logger.error(f"ONNX file not found: {onnx_path}")
            logger.info("Please run ONNX export first: python convert_to_onnx.py --model unet3d")
            return None
        
        # Since the ONNX model has static dimensions, we don't need optimization profiles
        # The model has fixed shapes: latent_sample[2,13,16,64,64], timestep[], encoder_hidden_states[32,50,384]
        input_specs = {}  # Empty dict means no optimization profiles needed
        
        # Build FP16 engine
        return self.build_engine(onnx_path, engine_path, input_specs, 
                               max_workspace_size=8 << 30,  # 8GB
                               fp16_mode=True,
                               int8_mode=False)
    
    def convert_all(self):
        """Convert all ONNX models to TensorRT engines"""
        logger.info("Starting TensorRT conversion for all models...")
        
        engines = {}
        
        try:
            engines["vae_encoder"] = self.convert_vae_encoder()
            engines["vae_decoder"] = self.convert_vae_decoder()
            
            # Try TorchScript conversion first, fallback to ONNX if available
            torchscript_path = self.onnx_dir / "unet3d_torchscript.pt"
            if torchscript_path.exists():
                engines["unet3d"] = self.convert_unet3d_torchscript()
            else:
                engines["unet3d"] = self.convert_unet3d()
            
            logger.info("All engines converted successfully!")
            logger.info(f"TensorRT engines saved to: {self.output_dir}")
            
            # Log engine files
            for engine_file in self.output_dir.glob("*.trt"):
                size_mb = engine_file.stat().st_size / (1024 * 1024)
                logger.info(f"  {engine_file.name}: {size_mb:.1f} MB")
            
            return engines
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            raise
    
    def validate_engines(self):
        """Validate that all TensorRT engines are properly built"""
        logger.info("Validating TensorRT engines...")
        
        engine_files = [
            "vae_encoder.trt",
            "vae_decoder.trt", 
            "unet3d.trt",
            "audio_encoder.trt"
        ]
        
        all_valid = True
        for engine_file in engine_files:
            engine_path = self.output_dir / engine_file
            if not engine_path.exists():
                logger.error(f"Engine not found: {engine_path}")
                all_valid = False
                continue
            
            # Try to load engine
            try:
                with open(engine_path, 'rb') as f:
                    runtime = trt.Runtime(TRT_LOGGER)
                    engine = runtime.deserialize_cuda_engine(f.read())
                    if engine is None:
                        logger.error(f"Failed to deserialize engine: {engine_file}")
                        all_valid = False
                    else:
                        logger.info(f"✓ {engine_file} is valid")
            except Exception as e:
                logger.error(f"Error validating {engine_file}: {e}")
                all_valid = False
        
        return all_valid


def main():
    parser = argparse.ArgumentParser(description="Convert LatentSync ONNX models to TensorRT engines")
    parser.add_argument("--tensorrt_path", type=str, 
                       default="/home/codeway/srper/TensorRT-10.13.0.35",
                       help="Path to TensorRT installation")
    parser.add_argument("--onnx_dir", type=str, default="tensorrt_models",
                       help="Directory containing ONNX models")
    parser.add_argument("--output_dir", type=str, default="tensorrt_engines",
                       help="Output directory for TensorRT engines")
    parser.add_argument("--model", type=str, 
                       choices=["all", "vae_encoder", "vae_decoder", "unet3d", "unet3d_fp16", "unet3d_int8", "unet3d_torchscript", "unet3d_torchrt"],
                       default="unet3d_fp16", help="Which model to convert")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Enable FP16 precision")
    parser.add_argument("--int8", action="store_true", default=False,
                       help="Enable INT8 precision (requires calibration)")
    parser.add_argument("--workspace_size", type=int, default=8,
                       help="Max workspace size in GB")
    parser.add_argument("--validate", action="store_true",
                       help="Validate engines after conversion")
    parser.add_argument("--calibration_samples", type=int, default=100,
                       help="Number of calibration samples for INT8")
    
    args = parser.parse_args()
    
    converter = LatentSyncTensorRTConverter(
        tensorrt_path=args.tensorrt_path,
        onnx_dir=args.onnx_dir,
        output_dir=args.output_dir
    )
    
    # Convert models
    if args.model == "all":
        converter.convert_all()
    elif args.model == "vae_encoder":
        converter.convert_vae_encoder()
    elif args.model == "vae_decoder":
        converter.convert_vae_decoder()
    elif args.model == "unet3d":
        converter.convert_unet3d()
    elif args.model == "unet3d_fp16":
        converter.convert_unet3d_fp16()
    elif args.model == "unet3d_int8":
        converter.convert_unet3d_int8()
    elif args.model == "unet3d_torchscript":
        converter.convert_unet3d_torchscript()
    elif args.model == "unet3d_torchrt":
        converter.convert_unet3d_torchrt()
    
    # Validate if requested
    if args.validate:
        if converter.validate_engines():
            logger.info("All engines validated successfully!")
        else:
            logger.error("Engine validation failed!")
            sys.exit(1)


if __name__ == "__main__":
    main() 