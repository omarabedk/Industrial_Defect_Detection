# model/dataset.py
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class SteelDefectDataset(Dataset):
    """
    Dataset that matches images in img_dir to masks in mask_dir.
    If a mask is not found by name, a zero mask is used.
    Returns: (image_tensor, mask_tensor) where
      image_tensor: float32 CxHxW
      mask_tensor: int64 HxW (values 0..4)
    """

    def __init__(self, img_dir, mask_dir, img_size=(256, 512)):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        assert self.img_dir.exists(), f"Image directory not found: {self.img_dir}"
        # mask_dir can be missing (we'll create zero masks)
        self.mask_dir_exists = self.mask_dir.exists()

        # collect image files
        self.img_files = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".bmp")])
        if len(self.img_files) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

        # collect mask files if present
        self.mask_files = sorted([p for p in self.mask_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".bmp")]) if self.mask_dir_exists else []

        # build quick lookup by stem
        self.mask_by_stem = {p.stem: p for p in self.mask_files}
        # also allow stems that end with "_mask"
        self.mask_by_stem_no_mask = {}
        for p in self.mask_files:
            stem = p.stem
            if stem.endswith("_mask"):
                self.mask_by_stem_no_mask[stem[:-5]] = p

        self.img_size = img_size  # (H, W)

        # transforms for images
        self.img_transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        # mask resize (nearest to preserve labels)
        self.mask_resize = T.Resize(self.img_size, interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        stem = img_path.stem

        # find mask path
        mask_path = None
        if self.mask_dir_exists:
            # prefer exact same stem + "_mask"
            cand = self.mask_by_stem.get(stem + "_mask")
            if cand is None:
                cand = self.mask_by_stem.get(stem)
            if cand is None:
                cand = self.mask_by_stem_no_mask.get(stem)
            mask_path = cand

        # load image
        img = Image.open(img_path).convert("RGB")
        img_t = self.img_transform(img)  # C,H,W float32

        # load mask or create zero mask
        if mask_path is not None and mask_path.exists():
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.fromarray(np.zeros((img.height, img.width), dtype=np.uint8))

        mask = self.mask_resize(mask)        # nearest resized
        mask_np = np.array(mask).astype(np.int64)  # H,W ints (0..4)
        mask_t = torch.from_numpy(mask_np)

        return img_t, mask_t
