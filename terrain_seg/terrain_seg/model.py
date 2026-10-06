"""Lazy-loaded Hugging Face SegFormer inference wrapper."""

import numpy as np


class SegformerRunner:
    """Run semantic segmentation without importing ML libraries at build time."""

    def __init__(self, model_id: str, device: str = 'auto', use_fp16=True):
        try:
            import torch
            from transformers import (AutoImageProcessor,
                                      SegformerForSemanticSegmentation)
        except ImportError as error:
            raise RuntimeError(
                'terrain_seg requires torch, transformers and Pillow. Run '
                '`python3 -m pip install -r requirements.txt`.') from error

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError(f'CUDA device requested but unavailable: {device}')

        self.torch = torch
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.use_fp16 = bool(use_fp16 and self.device.type == 'cuda')
        if self.use_fp16:
            self.model.half()

    def predict(self, bgr: np.ndarray):
        """Return uint8 labels and float32 maximum class probability."""
        torch = self.torch
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        inputs = self.processor(images=rgb, return_tensors='pt')
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        if self.use_fp16:
            inputs['pixel_values'] = inputs['pixel_values'].half()

        with torch.inference_mode():
            logits = self.model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits, size=bgr.shape[:2], mode='bilinear',
                align_corners=False)
            probabilities = logits.softmax(dim=1)
            confidence, labels = probabilities.max(dim=1)
        return (labels[0].to(dtype=torch.uint8).cpu().numpy(),
                confidence[0].to(dtype=torch.float32).cpu().numpy())
