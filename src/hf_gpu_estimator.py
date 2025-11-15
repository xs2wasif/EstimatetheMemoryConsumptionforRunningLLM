#!/usr/bin/env python3
"""
Hugging Face Model GPU Requirement Estimator
Automatically calculates GPU requirements for any HuggingFace model
"""

import json
import math
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import os

# Set Hugging Face caching paths
os.environ["HF_HOME"] = "/data/models"
os.environ["TRANSFORMERS_CACHE"] = "/data/models"
os.environ["HF_DATASETS_CACHE"] = "/data/models/datasets"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "false"
os.environ["HF_HUB_DISABLE_XET"] = "True"

# Auto-create directories if they don't exist
os.makedirs("/data/models", exist_ok=True)
os.makedirs("/data/models/datasets", exist_ok=True)



try:
    from huggingface_hub import HfApi, model_info
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Install with: pip install huggingface_hub")

class ModelArchitecture(Enum):
    """Common model architectures and their characteristics"""
    DENSE = "dense"
    MOE = "moe"  # Mixture of Experts
    VISION = "vision"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"

@dataclass
class GPUSpec:
    """GPU specifications"""
    name: str
    memory_gb: int
    type: str  # Consumer, Professional, Data Center
    fp16_tflops: float
    price_per_hour: Optional[float] = None  # Cloud pricing

# GPU Database
GPU_DATABASE = [
    GPUSpec("A100 80GB", 80, "Data Center", 19.5, 4.0),
    GPUSpec("H100 80GB", 80, "Data Center", 60.0, 5.0),
    GPUSpec("H200 141GB", 141, "Data Center", 67.0, 7.0),
    GPUSpec("B200 192GB", 192, "Data Center", 120.0, 8.5),  # Estimated specs
]


class HuggingFaceModelEstimator:
    """Estimates GPU requirements for Hugging Face models"""
    
    def __init__(self, model_id: str, trust_remote_code: bool = False):
        self.model_id = model_id
        self.trust_remote_code = trust_remote_code
        self.model_info = None
        self.config = None
        self.architecture = ModelArchitecture.UNKNOWN
        self.total_params = 0
        self.active_params = 0
        
        # Try to fetch model information
        self._fetch_model_info()
        self._detect_architecture()
        self._estimate_parameters()

    def _filter_gpus(self, selected_types: List[str]) -> List[GPUSpec]:
        mapping = {
            "A100": "A100 80GB",
            "H100": "H100 80GB",
            "H200": "H200 141GB",
            "B200": "B200 192GB"
        }
        names = {mapping[t] for t in selected_types if t in mapping}
        return [gpu for gpu in GPU_DATABASE if gpu.name in names]

    
    def _fetch_model_info(self):
        """Fetch model information from Hugging Face Hub"""
        if HF_HUB_AVAILABLE:
            try:
                api = HfApi()
                self.model_info = api.model_info(self.model_id)
                
                # Try to get config.json
                try:
                    files = api.list_repo_files(self.model_id)
                    if "config.json" in files:
                        config_url = f"https://huggingface.co/{self.model_id}/raw/main/config.json"
                        import requests
                        response = requests.get(config_url)
                        if response.status_code == 200:
                            self.config = response.json()
                except:
                    pass
                    
            except Exception as e:
                print(f"Warning: Could not fetch model info from Hub: {e}")
    
    def _detect_architecture(self):
        """Detect model architecture type"""
        model_id_lower = self.model_id.lower()
        
        # Check for MoE models
        if any(x in model_id_lower for x in ['moe', 'mixtral', 'switch', 'expert']):
            self.architecture = ModelArchitecture.MOE
        # Check for vision models
        elif any(x in model_id_lower for x in ['vision', 'vit', 'clip', 'blip', 'vl']):
            if 'vl' in model_id_lower or 'clip' in model_id_lower:
                self.architecture = ModelArchitecture.MULTIMODAL
            else:
                self.architecture = ModelArchitecture.VISION
        # Default to dense
        else:
            self.architecture = ModelArchitecture.DENSE
        
        # Check config for more accurate detection
        if self.config:
            arch = self.config.get("architectures", [""])[0] if "architectures" in self.config else ""
            if "moe" in arch.lower():
                self.architecture = ModelArchitecture.MOE
            elif any(x in arch.lower() for x in ["vision", "vit", "clip"]):
                self.architecture = ModelArchitecture.MULTIMODAL
    
    def _estimate_parameters(self):
        """Estimate model parameters from available information"""
        # Try to get from model info
        if self.model_info and hasattr(self.model_info, 'safetensors'):
            safetensors = self.model_info.safetensors
            if safetensors and 'total' in safetensors:
                self.total_params = safetensors['total'] / 1e9  # Convert to billions
        
        # Try to extract from model ID or config
        if self.total_params == 0:
            # Look for patterns like "7B", "13B", "70B" in model name
            import re
            matches = re.findall(r'(\d+\.?\d*)B', self.model_id, re.IGNORECASE)
            if matches:
                self.total_params = float(matches[0])
            else:
                # Try from config
                if self.config:
                    # Different ways models store parameter count
                    if "num_parameters" in self.config:
                        self.total_params = self.config["num_parameters"] / 1e9
                    elif "n_params" in self.config:
                        self.total_params = self.config["n_params"] / 1e9
                    else:
                        # Calculate from architecture parameters
                        self.total_params = self._calculate_params_from_config()
        
        # For MoE models, estimate active parameters
        if self.architecture == ModelArchitecture.MOE:
            # Look for active parameters in model name (e.g., "A22B" means 22B active)
            import re
            active_matches = re.findall(r'A(\d+\.?\d*)B', self.model_id, re.IGNORECASE)
            if active_matches:
                self.active_params = float(active_matches[0])
            elif self.config:
                # Try to calculate from config
                num_experts = self.config.get("num_local_experts", self.config.get("num_experts", 8))
                experts_per_tok = self.config.get("num_experts_per_tok", 2)
                if num_experts > 0:
                    self.active_params = self.total_params * (experts_per_tok / num_experts)
            else:
                # Rough estimate: typically 10-20% of total params are active
                self.active_params = self.total_params * 0.15
        else:
            self.active_params = self.total_params
    
    def _calculate_params_from_config(self) -> float:
        """Calculate parameters from model config"""
        if not self.config:
            return 0
        
        try:
            # Common config keys
            hidden_size = self.config.get("hidden_size", self.config.get("d_model", 0))
            num_layers = self.config.get("num_hidden_layers", self.config.get("n_layers", 0))
            vocab_size = self.config.get("vocab_size", 0)
            
            if hidden_size and num_layers:
                # Very rough estimation
                # Embeddings + attention + FFN + layer norms
                params = vocab_size * hidden_size  # Embeddings
                params += num_layers * (
                    4 * hidden_size * hidden_size +  # QKV + O projections
                    4 * hidden_size * hidden_size * 4  # FFN (usually 4x hidden)
                )
                return params / 1e9
        except:
            pass
        
        return 0
    
    def calculate_memory_requirements(self, 
                                     precision: int = 16,
                                     batch_size: int = 1,
                                     sequence_length: int = 2048,
                                     use_flash_attention: bool = True,
                                     include_kv_cache: bool = True,
                                     gradient_checkpointing: bool = False) -> Dict:
        """Calculate memory requirements for the model"""
        
        if self.total_params == 0:
            raise ValueError(f"Could not determine model size for {self.model_id}")
        
        memory = {}
        
        # Model weights
        bytes_per_param = precision / 8
        memory['model_weights_gb'] = (self.total_params * 1e9 * bytes_per_param) / (1024**3)
        
        # KV Cache (if applicable)
        if include_kv_cache and self.architecture in [ModelArchitecture.DENSE, ModelArchitecture.MOE]:
            # Estimate based on model size
            if self.config:
                hidden_size = self.config.get("hidden_size", int(math.sqrt(self.total_params * 1e9 / 100) * 4))
                num_layers = self.config.get("num_hidden_layers", int(self.total_params * 2))
                num_heads = self.config.get("num_attention_heads", 32)
            else:
                # Rough estimates based on model size
                hidden_size = int(math.sqrt(self.total_params * 1e9 / 100) * 4)
                num_layers = min(int(self.total_params * 2), 100)
                num_heads = 32
            
            kv_cache_params = 2 * batch_size * sequence_length * num_layers * hidden_size
            memory['kv_cache_gb'] = (kv_cache_params * bytes_per_param) / (1024**3)
        else:
            memory['kv_cache_gb'] = 0
        
        # Activations
        if use_flash_attention:
            activation_memory_gb = self.active_params * 0.5 * batch_size  # Rough estimate
        else:
            activation_memory_gb = self.active_params * 2 * batch_size  # Higher without flash attention
        
        if gradient_checkpointing:
            activation_memory_gb *= 0.3  # Significant reduction with gradient checkpointing
        
        memory['activations_gb'] = activation_memory_gb
        
        # Vision/Multimodal overhead
        if self.architecture in [ModelArchitecture.VISION, ModelArchitecture.MULTIMODAL]:
            memory['vision_overhead_gb'] = self.total_params * 0.1  # 10% overhead for vision
        else:
            memory['vision_overhead_gb'] = 0
        
        # Framework overhead (PyTorch, etc.)
        memory['overhead_gb'] = memory['model_weights_gb'] * 0.1  # 10% overhead
        
        # Total
        memory['total_inference_gb'] = sum(memory.values())
        memory['peak_inference_gb'] = memory['total_inference_gb'] * 1.2  # 20% peak overhead
        
        return memory
    
    def calculate_training_memory(self,
                                 precision: int = 16,
                                 batch_size: int = 1,
                                 sequence_length: int = 2048,
                                 optimizer: str = "adamw",
                                 use_lora: bool = False,
                                 lora_rank: int = 16,
                                 gradient_checkpointing: bool = False) -> Dict:
        """Calculate training memory requirements"""
        
        memory = {}
        bytes_per_param = precision / 8
        
        # Model weights
        memory['model_weights_gb'] = (self.total_params * 1e9 * bytes_per_param) / (1024**3)
        
        if use_lora:
            # LoRA parameters (rough estimate: 0.1% of model size)
            lora_params = self.total_params * 1e9 * 0.001 * lora_rank / 16
            memory['lora_adapters_gb'] = (lora_params * 4) / (1024**3)  # FP32
            trainable_params = lora_params
        else:
            memory['lora_adapters_gb'] = 0
            trainable_params = self.total_params * 1e9
        
        # Gradients
        memory['gradients_gb'] = (trainable_params * bytes_per_param) / (1024**3)
        
        # Optimizer states
        if optimizer.lower() == "adamw":
            memory['optimizer_gb'] = (trainable_params * 8) / (1024**3)  # 2x FP32 states
        elif optimizer.lower() == "sgd":
            memory['optimizer_gb'] = (trainable_params * 4) / (1024**3)  # 1x FP32 state
        else:
            memory['optimizer_gb'] = 0
        
        # Activations (much higher for training)
        if gradient_checkpointing:
            activation_multiplier = 4
        else:
            activation_multiplier = 12
        
        memory['activations_gb'] = self.active_params * activation_multiplier * batch_size
        
        # Total
        memory['total_training_gb'] = sum(memory.values())
        memory['peak_training_gb'] = memory['total_training_gb'] * 1.3
        
        return memory
    
    def recommend_gpus(self, memory_required: float, selected_types: List[str], allow_multi_gpu: bool = True) -> List[Dict]:

        """Recommend GPU configurations based on memory requirements.
        Always recommend something — scales up to 1024 GPUs if needed.
        """
        gpu_pool = self._filter_gpus(selected_types)

        recommendations = []

        # First check if any single GPU can handle it
        for gpu in GPU_DATABASE:
            if gpu.memory_gb >= memory_required * 1.05:  # 5% safety margin
                recommendations.append({
                    'configuration': f"1x {gpu.name}",
                    'total_memory_gb': gpu.memory_gb,
                    'gpu_type': gpu.type,
                    'estimated_cost_per_hour': gpu.price_per_hour,
                    'num_gpus': 1
                })

        # Multi-GPU search (guaranteed recommendation)
        if allow_multi_gpu and not recommendations:
            for gpu in sorted(GPU_DATABASE, key=lambda x: x.memory_gb, reverse=True):
                for num_gpus in range(2, 1025):  # Scale up to 1024 GPUs
                    total_memory = gpu.memory_gb * num_gpus
                    if total_memory >= memory_required * 1.05:
                        recommendations.append({
                            'configuration': f"{num_gpus}x {gpu.name}",
                            'total_memory_gb': total_memory,
                            'gpu_type': gpu.type,
                            'estimated_cost_per_hour': gpu.price_per_hour * num_gpus if gpu.price_per_hour else None,
                            'num_gpus': num_gpus
                        })
                        break  # stop once best match for that GPU is found

        # Sort recommendations: least GPU count → least excess memory → lowest hourly price
        recommendations.sort(key=lambda r: (r['num_gpus'],
                                            r['total_memory_gb'] - memory_required,
                                            r['estimated_cost_per_hour'] or 0))

        # Always return at least the best match
        return recommendations[:10] if recommendations else []

    
    def generate_report(self,
                    batch_size: int = 1,
                    sequence_length: int = 2048,
                    show_training: bool = True,
                    selected_gpu_types: List[str] = None) -> str:

        """Generate a comprehensive report"""
        report = []
        report.append("=" * 80)
        report.append(f"GPU REQUIREMENTS ANALYSIS FOR: {self.model_id}")
        report.append("=" * 80)
        
        # Model information
        report.append("\n### Model Information ###")
        report.append(f"Architecture: {self.architecture.value}")
        report.append(f"Total Parameters: {self.total_params:.1f}B")
        if self.architecture == ModelArchitecture.MOE:
            report.append(f"Active Parameters: {self.active_params:.1f}B")
        report.append(f"Batch Size: {batch_size}")
        report.append(f"Sequence Length: {sequence_length}")
        
        # Inference memory for different precisions
        report.append("\n### Inference Memory Requirements ###")
        precisions = [
            (32, "FP32", "Full precision"),
            (16, "FP16/BF16", "Half precision (recommended)"),
            (8, "INT8", "8-bit quantization"),
            (4, "INT4", "4-bit quantization (QLoRA)"),
        ]
        
        inference_results = []
        for bits, name, desc in precisions:
            mem = self.calculate_memory_requirements(
                precision=bits,
                batch_size=batch_size,
                sequence_length=sequence_length
            )
            inference_results.append((name, mem))
            
            report.append(f"\n{name} ({desc}):")
            report.append(f"  Model weights: {mem['model_weights_gb']:.1f} GB")
            if mem['kv_cache_gb'] > 0:
                report.append(f"  KV Cache: {mem['kv_cache_gb']:.1f} GB")
            report.append(f"  Activations: {mem['activations_gb']:.1f} GB")
            report.append(f"  Total: {mem['total_inference_gb']:.1f} GB")
            report.append(f"  Peak: {mem['peak_inference_gb']:.1f} GB")
        
        # GPU recommendations for inference
        report.append("\n### GPU Recommendations for Inference ###")
        fp16_mem = inference_results[1][1]  # FP16 results
        recommendations = self.recommend_gpus(fp16_mem['peak_inference_gb'], selected_gpu_types)

        
        if recommendations:
            report.append("\nRecommended configurations (FP16):")
            for i, rec in enumerate(recommendations[:5], 1):
                report.append(f"  {i}. {rec['configuration']} ({rec['total_memory_gb']}GB total)")
                if rec['estimated_cost_per_hour']:
                    report.append(f"     Estimated cost: ${rec['estimated_cost_per_hour']:.2f}/hour")
        else:
            report.append("  No suitable GPU configurations found in database")
        
        # Training memory
        if show_training:
            report.append("\n### Training Memory Requirements ###")
            
            # Full fine-tuning
            train_mem = self.calculate_training_memory(
                precision=16,
                batch_size=batch_size,
                sequence_length=sequence_length,
                use_lora=False
            )
            report.append("\nFull Fine-tuning (FP16):")
            report.append(f"  Model weights: {train_mem['model_weights_gb']:.1f} GB")
            report.append(f"  Gradients: {train_mem['gradients_gb']:.1f} GB")
            report.append(f"  Optimizer: {train_mem['optimizer_gb']:.1f} GB")
            report.append(f"  Activations: {train_mem['activations_gb']:.1f} GB")
            report.append(f"  Total: {train_mem['total_training_gb']:.1f} GB")
            report.append(f"  Peak: {train_mem['peak_training_gb']:.1f} GB")
            
            # LoRA fine-tuning
            lora_mem = self.calculate_training_memory(
                precision=16,
                batch_size=batch_size,
                sequence_length=sequence_length,
                use_lora=True
            )
            report.append("\nLoRA Fine-tuning (rank=16):")
            report.append(f"  Total: {lora_mem['total_training_gb']:.1f} GB")
            report.append(f"  Peak: {lora_mem['peak_training_gb']:.1f} GB")
            reduction = (1 - lora_mem['total_training_gb']/train_mem['total_training_gb']) * 100
            report.append(f"  Memory reduction: {reduction:.1f}%")
            
            # GPU recommendations for training
            report.append("\n### GPU Recommendations for Training ###")
            train_recommendations = self.recommend_gpus(train_mem['peak_training_gb'], selected_gpu_types)

            
            if train_recommendations:
                report.append("\nRecommended configurations (Full training):")
                for i, rec in enumerate(train_recommendations[:3], 1):
                    report.append(f"  {i}. {rec['configuration']} ({rec['total_memory_gb']}GB total)")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Estimate GPU requirements for Hugging Face models")
    parser.add_argument("model_id", type=str, help="Hugging Face model ID (e.g., meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--sequence-length", type=int, default=2048, help="Sequence length (default: 2048)")
    parser.add_argument("--precision", type=int, default=16, choices=[4, 8, 16, 32], 
                       help="Model precision in bits (default: 16)")
    parser.add_argument("--no-training", action="store_true", help="Skip training memory estimation")
    parser.add_argument("--trust-remote-code", action="store_true", 
                       help="Trust remote code when loading model config")
    parser.add_argument(
        "--gpu-types",
        type=str,
        default="A100,H100,H200,B200",
        help="Comma-separated GPU types to consider for recommendations. Options: A100,H100,H200,B200"
    )
    
    
    args = parser.parse_args()
    selected_gpu_types = [x.strip().upper() for x in args.gpu_types.split(",")]
    
    try:
        # Create estimator
        print(f"Analyzing model: {args.model_id}")
        estimator = HuggingFaceModelEstimator(args.model_id, args.trust_remote_code)
        
        # Generate and print report
        report = estimator.generate_report(
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            show_training=not args.no_training,
            selected_gpu_types=selected_gpu_types
        )
        print(report)
        
        # Quick summary
        mem = estimator.calculate_memory_requirements(
            precision=args.precision,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length
        )
        
        print(f"\n💡 QUICK ANSWER:")
        print(f"   Model size: {estimator.total_params:.1f}B parameters")
        print(f"   Memory needed ({args.precision}-bit): {mem['peak_inference_gb']:.1f} GB")
        
        recommendations = estimator.recommend_gpus(mem['peak_inference_gb'], selected_gpu_types)

        if recommendations:
            print(f"   Minimum GPU: {recommendations[0]['configuration']}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())