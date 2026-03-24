# -*- coding: utf-8 -*-
"""Download Z-Image model"""
import torch
from diffusers import ZImagePipeline

print("=" * 50)
print("  Downloading Z-Image-Turbo Model")
print("=" * 50)
print()
print("This may take 10-30 minutes depending on your network.")
print()

ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16
)

print()
print("=" * 50)
print("  Download complete!")
print("=" * 50)
print()
input("Press Enter to exit...")

