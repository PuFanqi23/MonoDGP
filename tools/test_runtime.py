from thop import profile
import torch
import os
import yaml
import sys
import time
import warnings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from lib.helpers.model_helper import build_model

CUDA_VISIBLE_DEVICES=0

cfg = yaml.load(open('./configs/monodgp.yaml', 'r'), Loader=yaml.Loader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, loss = build_model(cfg['model'])
model = model.to(device)
model.eval() 

input_img = torch.randn(1, 3, 384, 1280).to(device)  
calib = torch.randn(1, 3, 4).to(device)
sizes = torch.randn(1, 2).to(device)
input_size = (3, 384, 1280) 


# warm up
for _ in range(10):
    _ = model(input_img, calib, None, sizes)

# test runtime(ms)
start_time = time.time()
for _ in range(100): 
    _ = model(input_img, calib, None, sizes)
end_time = time.time()

inference_time = (end_time - start_time) / 100 * 1000  


# precise test
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)
# torch.cuda.synchronize()

# start_event.record()
# for _ in range(100):
#     _ = model(input_img, calib, None, sizes)
# end_event.record()
# torch.cuda.synchronize()

# inference_time = start_event.elapsed_time(end_event) / 100 


# test params and flops
flops, params = profile(model, inputs=(input_img, calib, None, sizes))

print(f"Inference Time: {inference_time:.2f} ms")
print(f"Params: {params / 1e6:.2f} M")
print(f"FLOPS: {flops / 1e9:.2f} G")
