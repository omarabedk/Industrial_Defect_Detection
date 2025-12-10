import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.models.segmentation import fcn_resnet50

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "defect_fcn_resnet50.pth")
GRADCAM_DIR = os.path.join(BASE_DIR, "outputs", "gradcam")
os.makedirs(GRADCAM_DIR, exist_ok=True)

# Load model
def load_model(num_classes=5):
    model = fcn_resnet50(weights=None, num_classes=num_classes)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

# Preprocess
def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.Resize((256, 1600)),
        T.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)
    return image, tensor

# Add this to gradcam.py

def generate_gradcam_for_image(model, pil_img, target_class=1):
    """
    Generate Grad-CAM heatmap for a PIL image and return as PIL Image.
    """
    import torchvision.transforms as T
    from PIL import Image
    import numpy as np
    import torch

    # Preprocess
    transform = T.Compose([
        T.Resize((256, 1600)),
        T.ToTensor()
    ])
    tensor = transform(pil_img).unsqueeze(0)
    tensor.requires_grad_()

    # Use your existing Grad-CAM function
    cam = generate_gradcam(model, tensor, target_layer="backbone.layer4")  # keep same layer

    # Convert CAM to PIL image
    cam_img = Image.fromarray(np.uint8(cam * 255)).convert("RGB")
    cam_img = cam_img.resize(pil_img.size)
    return cam_img


# Grad-CAM
def generate_gradcam(model, tensor, target_layer="backbone.layer4"):
    gradients = []
    activations = []

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activation(module, input, output):
        activations.append(output)

    # Hook target layer
    for name, module in model.named_modules():
        if name == target_layer:
            module.register_forward_hook(save_activation)
            module.register_backward_hook(save_gradient)

    output = model(tensor)["out"]
    # Pick class 1 (defect type 1) for Grad-CAM
    target_class = 1
    score = output[0, target_class, :, :].mean()

    model.zero_grad()
    score.backward()

    grad = gradients[0][0].cpu().data.numpy()
    act = activations[0][0].cpu().data.numpy()

    weights = grad.mean(axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam + 1e-8)
    return cam

# Overlay heatmap
def overlay_heatmap(img, cam):
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(img.size)
    cam_img = cam_img.convert("L")
    cam_img = np.array(cam_img)
    plt.imshow(img)
    plt.imshow(cam_img, cmap="jet", alpha=0.5)
    return plt

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to image")
    args = parser.parse_args()

    model = load_model()
    image, tensor = preprocess_image(args.img)
    cam = generate_gradcam(model, tensor)

    # Overlay & save
    plt = overlay_heatmap(image, cam)
    fname = os.path.splitext(os.path.basename(args.img))[0]
    save_path = os.path.join(GRADCAM_DIR, f"{fname}_gradcam.png")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Grad-CAM: {save_path}")

if __name__ == "__main__":
    main()
