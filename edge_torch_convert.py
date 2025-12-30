import torch
import ai_edge_torch
from transformers import AutoModelForDepthEstimation

model = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
).eval()

sample_input = (
    torch.randn(1, 3, 384, 384),
)

edge_model = ai_edge_torch.convert(
    model,
    sample_args=sample_input,
)

edge_model.export("depth_anything_metric.tflite")
