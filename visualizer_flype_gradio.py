import os
import os.path as osp
from argparse import ArgumentParser
from functools import partial

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from modeling.flype2 import FLYPE
import gradio as gr
from transformers import Blip2Model, Blip2Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_config = Blip2Config.from_pretrained("Salesforce/blip2-opt-2.7b")
model_config.pr_seq_len = 5
model_config.num_heads = 8
model = FLYPE(model_config).to(device)

state_dict = torch.load("checkpoints/flype_opt/bs_20_lr0.001_seq_len5_epochs2.pt")
model.load_state_dict(state_dict, strict=False)
model = model.cuda()
model.eval()

def predict(inp):
  inp = transforms.ToTensor()(inp).unsqueeze(0)
  with torch.no_grad():
    prediction = model(inp).logits
    confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
  return confidences


gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=3),
             examples=["lion.jpg", "cheetah.jpg"]).launch()
