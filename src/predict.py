import argparse
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.models.segmentation import fcn_resnet50

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "defect_fcn_resnet50.pth")
PRED_DIR = os.path.join(BASE_DIR, "outputs", "predictions")

os.makedirs(PRED_DIR, exist_ok=True)


def load_model():
    print(f"Loading model from: {MODEL_PATH}")

    model = fcn_resnet50(weights=None, num_classes=5)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")

    transform = T.Compose([
        T.Resize((256, 1600)),
        T.ToTensor(),
    ])

    tensor = transform(image).unsqueeze(0)
    return image, tensor


def predict_mask(model, tensor):
    with torch.no_grad():
        output = model(tensor)["out"]
        mask = torch.sigmoid(output)[0, 0].cpu().numpy()

    # Threshold → binary mask
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask


def save_results(original_img, mask, img_path):
    fname = os.path.splitext(os.path.basename(img_path))[0]

    # Save raw mask
    mask_img = Image.fromarray(mask)
    mask_path = os.path.join(PRED_DIR, f"{fname}_mask.png")
    mask_img.save(mask_path)

    # Create overlay
    overlay = original_img.copy()
    mask_rgb = Image.fromarray(mask).convert("RGB")

    # Red mask
    red_mask = np.array(mask_rgb)
    red_mask[:, :, 0] = mask      # Red
    red_mask[:, :, 1] = 0         # Green
    red_mask[:, :, 2] = 0         # Blue
    red_mask = Image.fromarray(red_mask).resize(original_img.size)

    overlay = Image.blend(original_img, red_mask, alpha=0.5)

    overlay_path = os.path.join(PRED_DIR, f"{fname}_overlay.png")
    overlay.save(overlay_path)

    print(f"Saved mask: {mask_path}")
    print(f"Saved overlay: {overlay_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to image")
    args = parser.parse_args()

    model = load_model()

    original_img, tensor = preprocess_image(args.img)
    mask = predict_mask(model, tensor)

    save_results(original_img, mask, args.img)


if __name__ == "__main__":
    main()
