"""
infer.py  --  Run inference with the saved FL checkpoint on PlantVillage
Usage:
    # Random samples from the dataset
    python infer.py --data_dir ./plantvillage --samples 10

    # Specific sample by index
    python infer.py --data_dir ./plantvillage --index 42

    # Your own image file(s)
    python infer.py --image leaf.jpg
    python infer.py --image a.jpg b.jpg c.jpg

    # Save extracted samples to ./test/ for visual inspection
    python infer.py --data_dir ./plantvillage --samples 20 --save
"""

import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms

from model import PlantNet, NUM_CLASSES


VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(checkpoint_path: str, num_classes: int, device) -> torch.nn.Module:
    model = PlantNet(num_classes=num_classes, pretrained=False).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def predict_tensor(model, tensor: torch.Tensor, device) -> tuple[int, float]:
    tensor = tensor.unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1)
    conf, cls = probs.max(dim=1)
    return cls.item(), conf.item() * 100.0


def save_sample(raw_img: Image.Image, idx: int, true_label: str,
                pred_label: str, conf: float, out_dir: str) -> str:
    """Save image scaled to 224x224 with result encoded in filename."""
    img = raw_img.resize((224, 224), Image.BILINEAR)
    ok = "OK" if pred_label == true_label else "WRONG"
    # Shorten class names to keep filenames manageable
    true_short = true_label[:30]
    pred_short = pred_label[:30]
    filename = f"{idx}_{true_short}_pred-{pred_short}_{ok}_{conf:.0f}pct.jpg"
    img.save(os.path.join(out_dir, filename))
    return filename


def run_dataset_inference(model, data_dir: str, indices: list[int],
                          device, save: bool, out_dir: str):
    dataset     = datasets.ImageFolder(root=data_dir, transform=VAL_TRANSFORM)
    raw_dataset = datasets.ImageFolder(root=data_dir, transform=None)
    class_names = dataset.classes

    if save:
        os.makedirs(out_dir, exist_ok=True)

    print(f"  {'Index':<8} {'True Label':<40} {'Predicted':<40} {'Conf':>8} {'OK':>5}")
    print(f"  {'-'*105}")

    correct = 0
    for idx in indices:
        tensor, true_idx = dataset[idx]
        pred_idx, conf   = predict_tensor(model, tensor, device)
        true_label = class_names[true_idx]
        pred_label = class_names[pred_idx]
        ok = "V" if pred_idx == true_idx else "X"
        if pred_idx == true_idx:
            correct += 1

        filename = ""
        if save:
            raw_img, _ = raw_dataset[idx]
            filename = save_sample(raw_img, idx, true_label, pred_label, conf, out_dir)

        print(f"  {idx:<8} {true_label:<40} {pred_label:<40} {conf:>7.1f}% {ok:>5}"
              + (f"  -> {filename}" if save else ""))

    print(f"\n  Accuracy : {correct}/{len(indices)} ({100*correct/len(indices):.1f}%)")
    if save:
        print(f"  Saved to : {os.path.abspath(out_dir)}")


def run_image_inference(model, image_paths: list[str], class_names: list[str], device):
    for path in image_paths:
        try:
            img    = Image.open(path).convert("RGB")
            tensor = IMAGE_TRANSFORM(img)
            pred_idx, conf = predict_tensor(model, tensor, device)
            label  = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
            print(f"  Image      : {path}")
            print(f"  Prediction : {label}")
            print(f"  Confidence : {conf:.2f}%")
            print()
        except FileNotFoundError:
            print(f"  ERROR: File not found -- {path}", file=sys.stderr)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="FL Plant Disease Inference")
    parser.add_argument("--checkpoint",  type=str, default="global_model_final.pth")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES)

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--data_dir", type=str,
                     help="PlantVillage dataset folder (for sampling test images)")
    src.add_argument("--image",    type=str, nargs="+",
                     help="Path(s) to your own image file(s)")

    parser.add_argument("--samples", type=int, default=10,
                        help="Number of random samples  (default: 10)")
    parser.add_argument("--index",   type=int, default=None,
                        help="Specific dataset index to predict")
    parser.add_argument("--save",    action="store_true",
                        help="Save sample images to --out_dir for visual inspection")
    parser.add_argument("--out_dir", type=str, default="test",
                        help="Folder to save images to  (default: test)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print("      FL Inference -- Plant Disease Detection")
    print("=" * 55)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {device}")
    print("=" * 55 + "\n")

    model = load_model(args.checkpoint, args.num_classes, device)
    print("  Model loaded.\n")

    if args.data_dir:
        # Build class name list from dataset
        tmp = datasets.ImageFolder(root=args.data_dir)
        class_names = tmp.classes
        total = len(tmp)

        if args.index is not None:
            indices = [args.index]
        else:
            indices = random.sample(range(total), min(args.samples, total))

        run_dataset_inference(model, args.data_dir, indices, device,
                              args.save, args.out_dir)
    else:
        # For standalone image inference, derive class names from the checkpoint
        # folder name pattern or just show numeric index
        class_names = [str(i) for i in range(args.num_classes)]
        run_image_inference(model, args.image, class_names, device)

    print()


if __name__ == "__main__":
    main()
