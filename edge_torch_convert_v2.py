import torch 
import ai_edge_torch 
from transformers import AutoModelForDepthEstimation  

model = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf" 
).eval()  

# Wrapper para cambiar el formato de entrada a NHWC
class NHWCWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # x viene en NHWC [1, 384, 384, 3]
        # Convertir a NCHW [1, 3, 384, 384]
        x = x.permute(0, 3, 1, 2)
        return self.model(x)

wrapped_model = NHWCWrapper(model)

sample_input = (     
    torch.randn(1, 384, 384, 3),  # NHWC format
)  

edge_model = ai_edge_torch.convert(     
    wrapped_model,     
    sample_args=sample_input, 
)  

edge_model.export("depth_anything_metric_nhwc.tflite")