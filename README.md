# EstimatetheMemoryConsumptionforRunningLLM
- python hf_gpu_estimator.py meta-llama/Llama-2-7b-hf --batch-size 1 --sequence-length 4096
- python hf_gpu_estimator.py Qwen/Qwen3-VL-235B-A22B-Thinking-fp8 --batch-size 1 --sequence-length 4026 --precision 8 --no-training