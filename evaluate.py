"""
evaluate.py  --  Evaluate the saved FL checkpoint on PlantVillage
Usage:
    python evaluate.py --data_dir ./plantvillage
    python evaluate.py --data_dir ./plantvillage --checkpoint global_model_final.pth
    python evaluate.py --data_dir ./plantvillage --per_class
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import PlantNet, NUM_CLASSES


VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_loader(data_dir: str, batch_size: int = 64):
    dataset = datasets.ImageFolder(root=data_dir, transform=VAL_TRANSFORM)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)
    return loader, dataset.classes


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct    = 0
    total      = 0
    total_loss = 0.0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        total_loss += F.cross_entropy(output, target, reduction="sum").item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total   += target.size(0)

    return 100.0 * correct / total, total_loss / total


@torch.no_grad()
def per_class_accuracy(model, loader, class_names, device):
    model.eval()
    n = len(class_names)
    class_correct = [0] * n
    class_total   = [0] * n

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        pred = model(data).argmax(dim=1)
        for t, p in zip(target, pred):
            class_total[t.item()]   += 1
            if t == p:
                class_correct[t.item()] += 1

    print(f"\n  {'Class':<45} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*75}")
    for i, name in enumerate(class_names):
        acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] else 0
        print(f"  {name:<45} {class_correct[i]:>8} {class_total[i]:>8} {acc:>9.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate FL checkpoint on PlantVillage")
    parser.add_argument("--data_dir",    type=str, required=True,
                        help="Path to PlantVillage dataset (ImageFolder layout)")
    parser.add_argument("--checkpoint",  type=str, default="global_model_final.pth",
                        help="Path to checkpoint (default: global_model_final.pth)")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES,
                        help=f"Number of classes (default: {NUM_CLASSES})")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--per_class",   action="store_true",
                        help="Show per-class accuracy breakdown")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print("   FL Checkpoint Evaluation -- PlantVillage")
    print("=" * 55)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Device     : {device}")

    model = PlantNet(num_classes=args.num_classes, pretrained=False).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("  Model loaded successfully.")

    loader, class_names = get_loader(args.data_dir, args.batch_size)
    print(f"  Samples    : {len(loader.dataset)}")
    print(f"  Classes    : {len(class_names)}")
    print("=" * 55)

    accuracy, avg_loss = evaluate(model, loader, device)
    print(f"\n  Accuracy : {accuracy:.2f}%")
    print(f"  Loss     : {avg_loss:.4f}")

    if args.per_class:
        per_class_accuracy(model, loader, class_names, device)

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()
