"""
client.py  —  Federated Learning Edge Client (run on PC 2 and PC 3)
────────────────────────────────────────────────────────────────────
Each federated learning round:
  1. Download global model from server
  2. Train locally on this client's data partition
  3. Send updated weights back to server
  4. Repeat until all rounds are done

Dataset: PlantVillage
  Download from Kaggle:
      kaggle datasets download -d abdallahalidev/plantvillage-dataset
  Extract so the folder structure is:
      <data_dir>/
          Apple___Apple_scab/
          Apple___Black_rot/
          ...  (38 folders total)

Usage (PC 2):
    python client.py --client_id 1 --server http://<SERVER_IP>:5000 --data_dir ./plantvillage

Usage (PC 3):
    python client.py --client_id 2 --server http://<SERVER_IP>:5000 --data_dir ./plantvillage
"""

import argparse
import base64
import io
import time

import torch
import torch.nn as nn
import torch.optim as optim
import requests
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from model import PlantNet, NUM_CLASSES


# ─────────────────────────────────────── serialization ───────

def serialize_weights(state_dict: dict) -> str:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def deserialize_weights(encoded: str) -> dict:
    buf = io.BytesIO(base64.b64decode(encoded.encode("utf-8")))
    return torch.load(buf, map_location="cpu", weights_only=True)


# ─────────────────────────────────────── data ────────────────

# ImageNet normalization (matches MobileNetV2 pretrained weights)
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_data_partition(data_dir: str, client_id: int, num_clients: int,
                       batch_size: int = 32):
    """
    Loads PlantVillage from data_dir (ImageFolder layout) and returns a
    DataLoader for this client's partition of the training data.
    client_id is 1-based.
    """
    full_dataset = datasets.ImageFolder(root=data_dir, transform=TRAIN_TRANSFORM)

    total = len(full_dataset)
    partition_size = total // num_clients
    start = (client_id - 1) * partition_size
    end   = start + partition_size if client_id < num_clients else total
    indices = list(range(start, end))

    loader = DataLoader(
        Subset(full_dataset, indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,   # 0 avoids multiprocessing deadlocks on Windows
        pin_memory=False,
    )
    print(f"📦  Data partition: samples {start}–{end - 1}  ({len(indices)} samples, "
          f"{len(full_dataset.classes)} classes)")
    return loader, full_dataset.classes


def get_val_loader(data_dir: str, batch_size: int = 64):
    """Full dataset with val transforms — used for optional local evaluation."""
    dataset = datasets.ImageFolder(root=data_dir, transform=VAL_TRANSFORM)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)


# ─────────────────────────────────────── training ────────────

def train_local(model, dataloader, epochs: int, lr: float,
                device: torch.device) -> float:
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    last_loss  = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"    Epoch {epoch + 1}/{epochs}",
                    unit="batch", leave=True)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        last_loss = running_loss / len(dataloader)
        print(f"    Epoch {epoch + 1}/{epochs}  avg_loss={last_loss:.4f}")

    return last_loss


@torch.no_grad()
def evaluate(model, dataloader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        pred    = model(data).argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total   += target.size(0)
    return 100.0 * correct / total


# ─────────────────────────────────────── server comms ────────

def fetch_global_model(server_url: str, expected_round: int, retry_delay: int = 3):
    while True:
        try:
            resp = requests.get(f"{server_url}/get_model", timeout=10)
            resp.raise_for_status()
            body = resp.json()
            if body["round"] == expected_round:
                return deserialize_weights(body["model"])
            print(f"  ⏳  Server at round {body['round']}, waiting for round {expected_round}…")
        except Exception as exc:
            print(f"  ❌  GET /get_model failed: {exc}")
        time.sleep(retry_delay)


def submit_weights(server_url: str, client_id: int, round_num: int,
                   state_dict: dict, retry_delay: int = 5):
    payload = {
        "client_id": client_id,
        "round":     round_num,
        "weights":   serialize_weights(state_dict),
    }
    while True:
        try:
            resp = requests.post(
                f"{server_url}/submit_weights",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            print(f"  ❌  POST /submit_weights failed: {exc}. Retrying in {retry_delay}s…")
            time.sleep(retry_delay)


# ─────────────────────────────────────── main ────────────────

def main():
    parser = argparse.ArgumentParser(description="FL Edge Client — Plant Disease")
    parser.add_argument("--client_id",   type=int,   required=True,
                        help="Unique client ID (e.g. 1 or 2)")
    parser.add_argument("--server",      type=str,   default="http://localhost:5000",
                        help="Server URL  (default: http://localhost:5000)")
    parser.add_argument("--data_dir",    type=str,   required=True,
                        help="Path to PlantVillage dataset folder (ImageFolder layout)")
    parser.add_argument("--rounds",      type=int,   default=10,
                        help="Number of FL rounds  (default: 10)")
    parser.add_argument("--epochs",      type=int,   default=2,
                        help="Local training epochs per round  (default: 2)")
    parser.add_argument("--lr",          type=float, default=0.001,
                        help="SGD learning rate  (default: 0.001)")
    parser.add_argument("--batch_size",  type=int,   default=32,
                        help="Batch size  (default: 32)")
    parser.add_argument("--num_clients", type=int,   default=2,
                        help="Total number of clients  (default: 2)")
    parser.add_argument("--num_classes", type=int,   default=NUM_CLASSES,
                        help=f"Number of plant classes  (default: {NUM_CLASSES})")
    parser.add_argument("--evaluate",    action="store_true",
                        help="Run validation accuracy after each round")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # faster GPU training for fixed input sizes

    print("=" * 55)
    print(f"    Federated Learning Client  (id={args.client_id})")
    print("=" * 55)
    print(f"  Server      : {args.server}")
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Device      : {device}")
    print(f"  Rounds      : {args.rounds}")
    print(f"  Epochs/r    : {args.epochs}")
    print(f"  LR          : {args.lr}")
    print(f"  Batch size  : {args.batch_size}")
    print("=" * 55)

    train_loader, class_names = get_data_partition(
        args.data_dir, args.client_id, args.num_clients, args.batch_size
    )
    val_loader = get_val_loader(args.data_dir, args.batch_size) if args.evaluate else None

    model = PlantNet(num_classes=args.num_classes, pretrained=False).to(device)

    for round_num in range(args.rounds):
        print(f"\n{'─'*45}")
        print(f"  Round {round_num + 1}/{args.rounds}")
        print(f"{'─'*45}")

        # 1. Download global model
        print("  ⬇️   Fetching global model…")
        global_weights = fetch_global_model(args.server, round_num)
        model.load_state_dict(global_weights)
        print("  ✅  Global model loaded.")

        # 2. Local training
        print(f"  🏋️   Training locally for {args.epochs} epoch(s)…")
        avg_loss = train_local(model, train_loader, args.epochs, args.lr, device)
        print(f"  📉  Training loss: {avg_loss:.4f}")

        # Optional evaluation
        if args.evaluate and val_loader:
            acc = evaluate(model, val_loader, device)
            print(f"  🎯  Validation accuracy: {acc:.2f}%")

        # 3. Submit weights (move to CPU first for serialization)
        print("  ⬆️   Submitting weights to server…")
        result = submit_weights(
            args.server, args.client_id, round_num,
            {k: v.cpu() for k, v in model.state_dict().items()}
        )
        print(f"  📬  Server response: {result}")

    print(f"\n🎉  Client {args.client_id} finished all {args.rounds} rounds!")


if __name__ == "__main__":
    main()
