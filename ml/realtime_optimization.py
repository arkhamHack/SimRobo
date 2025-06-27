import torch
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Union, Callable
from .vjepa2 import VJEPA2Handler
import onnx
import onnxruntime as ort
import os
from tqdm import tqdm


class RealtimeOptimizer:
    """
    Tools for optimizing V-JEPA 2 model for real-time control
    Includes model compression, quantization, and optimized inference
    """
    
    def __init__(self, vjepa_handler: VJEPA2Handler):
        """
        Initialize optimizer with a V-JEPA 2 handler
        
        Args:
            vjepa_handler: Initialized V-JEPA 2 handler
        """
        self.vjepa_handler = vjepa_handler
        self.onnx_path = None
        self.onnx_session = None
        self.inference_stats = {
            'original_avg_ms': None,
            'optimized_avg_ms': None,
            'speedup_factor': None
        }
        
    def export_to_onnx(self, output_path: str, dynamic_batch: bool = True) -> str:
        """
        Export V-JEPA 2 model to ONNX format for optimized inference
        
        Args:
            output_path: Path to save the ONNX model
            dynamic_batch: Whether to use dynamic batch size
            
        Returns:
            Path to exported ONNX model
        """
        print("Exporting V-JEPA 2 model to ONNX format...")
        
        # Create dummy input for tracing
        dummy_input = torch.randn(1, 3, 256, 256, device=self.vjepa_handler.device)
        
        # Set model to evaluation mode
        self.vjepa_handler.model.eval()
        
        # Define dynamic axes if needed
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}
        
        # Export model
        with torch.no_grad():
            torch.onnx.export(
                self.vjepa_handler.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes
            )
        
        # Verify the model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model exported and verified. Saved at: {output_path}")
        
        self.onnx_path = output_path
        return output_path
    
    def optimize_onnx_model(self, output_path: Optional[str] = None) -> str:
        """
        Apply optimizations to ONNX model for better runtime performance
        
        Args:
            output_path: Path to save the optimized model (default: original path with '_optimized' suffix)
            
        Returns:
            Path to optimized ONNX model
        """
        if self.onnx_path is None:
            raise ValueError("No ONNX model has been exported. Call export_to_onnx first.")
            
        if output_path is None:
            base, ext = os.path.splitext(self.onnx_path)
            output_path = f"{base}_optimized{ext}"
        
        print(f"Optimizing ONNX model...")
        
        # Load the ONNX model
        model = onnx.load(self.onnx_path)
        
        # Apply optimization passes (constant folding, node elimination, etc.)
        from onnxoptimizer import optimize
        optimized_model = optimize(model, ['eliminate_unused_initializer', 'eliminate_identity',
                                          'eliminate_nop_transpose', 'fuse_consecutive_transposes',
                                          'fuse_bn_into_conv'])
        
        # Save the optimized model
        onnx.save(optimized_model, output_path)
        print(f"Optimized ONNX model saved at: {output_path}")
        
        self.onnx_path = output_path
        return output_path
    
    def quantize_onnx_model(self, bit_width: int = 8, output_path: Optional[str] = None) -> str:
        """
        Quantize model to reduce size and improve inference speed
        
        Args:
            bit_width: Quantization bit width (8 or 16)
            output_path: Path to save quantized model (default: original path with '_quantized' suffix)
            
        Returns:
            Path to quantized ONNX model
        """
        if self.onnx_path is None:
            raise ValueError("No ONNX model has been exported. Call export_to_onnx first.")
            
        if output_path is None:
            base, ext = os.path.splitext(self.onnx_path)
            output_path = f"{base}_quantized_{bit_width}bit{ext}"
            
        print(f"Quantizing ONNX model to {bit_width}-bit precision...")
        
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        if bit_width == 8:
            quant_type = QuantType.QInt8
        elif bit_width == 16:
            quant_type = QuantType.QInt16
        else:
            raise ValueError("Bit width must be 8 or 16")
            
        # Quantize model
        quantize_dynamic(
            model_input=self.onnx_path,
            model_output=output_path,
            weight_type=quant_type
        )
        
        print(f"Quantized {bit_width}-bit ONNX model saved at: {output_path}")
        
        self.onnx_path = output_path
        return output_path
    
    def initialize_inference_session(self, num_threads: int = 4):
        """
        Initialize ONNX Runtime inference session
        
        Args:
            num_threads: Number of threads for inference
        """
        if self.onnx_path is None:
            raise ValueError("No ONNX model available. Export model first.")
            
        print(f"Initializing ONNX Runtime session with {num_threads} threads...")
        
        # Set execution providers based on available hardware
        providers = []
        
        # Check device compatibility
        if self.vjepa_handler.device == 'cuda':
            # Use CUDA if available
            providers.append('CUDAExecutionProvider')
        elif self.vjepa_handler.device == 'mps':
            # CoreML execution provider for Mac M-series
            providers.append('CoreMLExecutionProvider')
            
        # Always add CPU as fallback
        providers.append('CPUExecutionProvider')
        
        # Create session options
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = num_threads
        
        # Initialize session
        self.onnx_session = ort.InferenceSession(
            self.onnx_path,
            sess_options=session_options,
            providers=providers
        )
        
        print(f"ONNX Runtime session initialized with providers: {self.onnx_session.get_providers()}")
    
    def optimize_for_edge_device(self, target_device: str = 'cpu', output_path: Optional[str] = None) -> str:
        """
        Optimize model specifically for edge deployment
        
        Args:
            target_device: Target device ('cpu', 'gpu', 'npu', etc.)
            output_path: Path to save edge-optimized model
            
        Returns:
            Path to edge-optimized model
        """
        if self.onnx_path is None:
            raise ValueError("No ONNX model has been exported. Call export_to_onnx first.")
            
        if output_path is None:
            base, ext = os.path.splitext(self.onnx_path)
            output_path = f"{base}_edge_{target_device}{ext}"
            
        print(f"Optimizing model for edge deployment on {target_device}...")
        
        # Load model
        model = onnx.load(self.onnx_path)
        
        # Apply edge-specific optimizations
        # This would typically involve device-specific optimizations
        # For this example, we'll apply general optimizations
        from onnxoptimizer import optimize
        
        # Select optimization passes based on target device
        if target_device == 'cpu':
            passes = ['eliminate_unused_initializer', 'eliminate_identity',
                      'fuse_bn_into_conv', 'fuse_add_bias_into_conv']
        else:  # For GPU/NPU
            passes = ['eliminate_unused_initializer', 'eliminate_identity',
                      'fuse_bn_into_conv', 'fuse_matmul_add_bias_into_gemm']
            
        # Apply optimizations
        optimized_model = optimize(model, passes)
        
        # Save the optimized model
        onnx.save(optimized_model, output_path)
        print(f"Edge-optimized model for {target_device} saved at: {output_path}")
        
        return output_path
    
    def benchmark_inference(self, num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark inference speeds of original PyTorch model vs optimized ONNX model
        
        Args:
            num_runs: Number of inference runs to average over
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"Benchmarking inference performance over {num_runs} runs...")
        
        # Create dummy input
        dummy_input_np = np.random.randn(1, 3, 256, 256).astype(np.float32)
        dummy_input_torch = torch.tensor(dummy_input_np, device=self.vjepa_handler.device)
        
        # Benchmark PyTorch model
        print("Benchmarking PyTorch model...")
        torch_times = []
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.vjepa_handler.model(dummy_input_torch)
        
        # Benchmark
        for _ in tqdm(range(num_runs)):
            start = time.time()
            with torch.no_grad():
                _ = self.vjepa_handler.model(dummy_input_torch)
                
            if self.vjepa_handler.device in ['cuda', 'mps']:
                torch.cuda.synchronize() if self.vjepa_handler.device == 'cuda' else torch.mps.synchronize()
                
            torch_times.append((time.time() - start) * 1000)  # ms
            
        torch_avg = np.mean(torch_times)
        
        # Benchmark ONNX model if available
        onnx_avg = None
        speedup = None
        
        if self.onnx_session is not None:
            print("Benchmarking ONNX model...")
            onnx_times = []
            
            # Warmup
            for _ in range(10):
                _ = self.onnx_session.run(
                    None, 
                    {'input': dummy_input_np}
                )
            
            # Benchmark
            for _ in tqdm(range(num_runs)):
                start = time.time()
                _ = self.onnx_session.run(
                    None, 
                    {'input': dummy_input_np}
                )
                onnx_times.append((time.time() - start) * 1000)  # ms
                
            onnx_avg = np.mean(onnx_times)
            speedup = torch_avg / onnx_avg if onnx_avg > 0 else 0
            
        # Save results
        self.inference_stats = {
            'original_avg_ms': torch_avg,
            'optimized_avg_ms': onnx_avg,
            'speedup_factor': speedup
        }
        
        # Print results
        print(f"PyTorch model average inference time: {torch_avg:.2f} ms")
        if onnx_avg is not None:
            print(f"ONNX model average inference time: {onnx_avg:.2f} ms")
            print(f"Speedup factor: {speedup:.2f}x")
        
        return self.inference_stats
    
    def optimize_end_to_end_pipeline(
        self,
        output_dir: str,
        quantization: bool = True,
        bit_width: int = 8,
        target_device: str = 'cpu',
        num_threads: int = 4
    ) -> str:
        """
        Run complete end-to-end optimization pipeline for real-time inference
        
        Args:
            output_dir: Directory to save optimized models
            quantization: Whether to apply quantization
            bit_width: Quantization bit width if enabled
            target_device: Target deployment device
            num_threads: Number of threads for inference
            
        Returns:
            Path to final optimized model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Export to ONNX
        onnx_path = os.path.join(output_dir, 'vjepa2_model.onnx')
        self.export_to_onnx(onnx_path)
        
        # Step 2: Optimize ONNX model
        optimized_path = os.path.join(output_dir, 'vjepa2_model_optimized.onnx')
        self.optimize_onnx_model(optimized_path)
        
        # Step 3: Quantize if requested
        if quantization:
            quantized_path = os.path.join(output_dir, f'vjepa2_model_quantized_{bit_width}bit.onnx')
            self.quantize_onnx_model(bit_width, quantized_path)
        
        # Step 4: Edge-specific optimizations
        edge_path = os.path.join(output_dir, f'vjepa2_model_edge_{target_device}.onnx')
        final_model_path = self.optimize_for_edge_device(target_device, edge_path)
        
        # Step 5: Initialize inference session
        self.initialize_inference_session(num_threads)
        
        # Step 6: Benchmark to show improvements
        self.benchmark_inference()
        
        return final_model_path
        
    def predict_with_optimized_model(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference using optimized ONNX model
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            Model output as numpy array
        """
        if self.onnx_session is None:
            raise ValueError("ONNX session not initialized. Call initialize_inference_session first.")
            
        # Preprocess image
        if isinstance(image, np.ndarray):
            # Convert to PIL Image
            from PIL import Image
            image_pil = Image.fromarray(image.astype('uint8'))
            
            # Apply preprocessing
            preprocess = self.vjepa_handler.preprocess
            image_tensor = preprocess(image_pil).numpy()
            
            # Add batch dimension if needed
            if len(image_tensor.shape) == 3:
                image_tensor = np.expand_dims(image_tensor, 0)
        else:
            # Assume already preprocessed
            image_tensor = image
            
        # Run inference
        outputs = self.onnx_session.run(None, {'input': image_tensor})
        
        return outputs[0]
