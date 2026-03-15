"""
client.py  --  Async FL Client (run on PC 2 and PC 3)
------------------------------------------------------
Architecture change from EdgeFed:

  OLD (EdgeFed): Client sent ONE conv-layer activation PER BATCH,
                 blocked waiting for the server to train + respond.
                 Every batch required a network round-trip.

  NEW (Async FL): Client trains LOCALLY for N full epochs,
                  then sends its trained weights once per round.
                  Server responds IMMEDIATELY -- no blocking.
                  Client and server train at their own pace.

Key option for CPU-only PCs:
  --freeze_backbone   Freeze MobileNetV2 backbone, train only the last
                      3 InvertedResidual blocks + classifier.
                      Reduces trainable params from ~3.4M to ~0.3M.
                      Training is ~10x faster on CPU with minimal accuracy loss.

Usage:
    python client.py --client_id 1 --server http://192.168.0.102:5000 --data_dir ./plantvillage
    python client.py --client_id 2 --server http://192.168.0.102:5000 --data_dir ./plantvillage --freeze_backbone

Optional flags:
    --rounds         10     Number of FL rounds
    --epochs         2      Local training epochs per round
    --batch_size     32     Batch size
    --lr             0.001  SGD learning rate
    --num_clients    2      Total clients (for data partitioning)
    --num_classes    38     Output classes
    --freeze_backbone       Freeze backbone for fast CPU training
    --evaluate              Run validation after each round
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


# ------------------------------------------------------------------ serialization

def serialize_weights(state_dict: dict) -> str:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def deserialize_weights(encoded: str) -> dict:
    buf = io.BytesIO(base64.b64decode(encoded.encode("utf-8")))
    return torch.load(buf, map_location="cpu", weights_only=True)


# ------------------------------------------------------------------ data loaders

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
    full  = datasets.ImageFolder(root=data_dir, transform=TRAIN_TRANSFORM)
    total = len(full)
    size  = total // num_clients
    start = (client_id - 1) * size
    end   = start + size if client_id < num_clients else total
    loader = DataLoader(
        Subset(full, list(range(start, end))),
        batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    print(f"  Data partition : samples {start}..{end - 1}  "
          f"({end - start} samples, {len(full.classes)} classes)")
    return loader, full.classes


def get_val_loader(data_dir: str, batch_size: int = 64) -> DataLoader:
    dataset = datasets.ImageFolder(root=data_dir, transform=VAL_TRANSFORM)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)


# ------------------------------------------------------------------ training helpers

def train_epoch(model: nn.Module, loader: DataLoader, optimizer,
                criterion, device, epoch: int, total_epochs: int) -> float:
    model.train()
    total_loss = correct = total = 0

    pbar = tqdm(loader,
                desc=f"    Epoch {epoch + 1}/{total_epochs}",
                unit="batch")

    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss   = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += output.argmax(1).eq(target).sum().item()
        total      += target.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}",
                         acc=f"{100 * correct / total:.1f}%")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device) -> float:
    model.eval()
    correct = total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        pred     = model(data).argmax(1)
        correct += pred.eq(target).sum().item()
        total   += target.size(0)
    return 100.0 * correct / total


# ------------------------------------------------------------------ server comms

def fetch_global_model(server_url: str, wait_for_round: int = 0,
                       retry_delay: int = 3) -> dict:
    """
    GET /get_model?wait_for_round=N
    Blocks server-side until the global model is at round >= N (i.e., FedAvg
    for round N-1 has completed). Use wait_for_round=0 for the initial
    download; use round_num for subsequent rounds.
    """
    url = f"{server_url}/get_model?wait_for_round={wait_for_round}"
    while True:
        try:
            # Long timeout: server may block up to 2 hours waiting for FedAvg
            resp = requests.get(url, timeout=7260)
            resp.raise_for_status()
            body = resp.json()
            print(f"  Downloaded global model (server round={body['round']})")
            return deserialize_weights(body["model"])
        except Exception as exc:
            print(f"  GET /get_model failed: {exc}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)


def submit_weights(server_url: str, client_id: int,
                   state_dict: dict, n_samples: int, round_num: int,
                   retry_delay: int = 3) -> dict:
    """
    POST /submit_weights -- send trained weights after local training.
    Server responds IMMEDIATELY; FedAvg fires automatically when all nodes submit.
    """
    payload = {
        "client_id": client_id,
        "round":      round_num,
        "n_samples":  n_samples,
        "weights":    serialize_weights(state_dict),
    }
    while True:
        try:
            resp = requests.post(f"{server_url}/submit_weights",
                                 json=payload, timeout=120)
            resp.raise_for_status()
            body = resp.json()
            print(f"  Weights accepted by server (server round={body.get('round','?')})")
            return body
        except Exception as exc:
            print(f"  POST /submit_weights failed: {exc}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description="Async FL Client")
    parser.add_argument("--client_id",       type=int,   required=True,
                        help="Unique client ID (1-based)")
    parser.add_argument("--server",          type=str,   default="http://localhost:5000",
                        help="Server URL (e.g. http://192.168.0.102:5000)")
    parser.add_argument("--data_dir",        type=str,   required=True,
                        help="PlantVillage dataset root (ImageFolder layout)")
    parser.add_argument("--rounds",          type=int,   default=10,
                        help="FL rounds (default: 10)")
    parser.add_argument("--epochs",          type=int,   default=2,
                        help="Local training epochs per round (default: 2)")
    parser.add_argument("--batch_size",      type=int,   default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr",              type=float, default=0.001,
                        help="SGD learning rate (default: 0.001)")
    parser.add_argument("--num_clients",     type=int,   default=2,
                        help="Total clients for data partitioning (default: 2)")
    parser.add_argument("--num_classes",     type=int,   default=NUM_CLASSES,
                        help=f"Output classes (default: {NUM_CLASSES})")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone, train only last 3 blocks + classifier. "
                             "~10x faster on CPU-only machines.")
    parser.add_argument("--evaluate",        action="store_true",
                        help="Run validation accuracy check after each round")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-enable freeze_backbone on CPU-only machines unless user explicitly
    # passed --freeze_backbone False (i.e. the flag was not set and no GPU found)
    if not args.freeze_backbone and device.type == "cpu":
        args.freeze_backbone = True
        print("  [Auto] No GPU detected -- freeze_backbone enabled automatically.")

    print("=" * 58)
    print(f"    Async FL Client  (id={args.client_id})")
    print("=" * 58)
    print(f"  Server          : {args.server}")
    print(f"  Data dir        : {args.data_dir}")
    print(f"  Device          : {device}")
    print(f"  Rounds          : {args.rounds}")
    print(f"  Epochs/round    : {args.epochs}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Freeze backbone : {args.freeze_backbone}")
    if args.freeze_backbone:
        print("    -> Only last 3 InvertedResidual blocks + classifier will train")
        print("    -> Trainable params ~0.3M instead of ~3.4M (fast on CPU)")
    print("=" * 58)

    train_loader, _ = get_data_partition(
        args.data_dir, args.client_id, args.num_clients, args.batch_size
    )
    n_train    = len(train_loader.dataset)
    val_loader = get_val_loader(args.data_dir, args.batch_size) if args.evaluate else None
    criterion  = nn.CrossEntropyLoss()

    model = PlantNet(num_classes=args.num_classes, pretrained=False).to(device)

    for round_num in range(args.rounds):
        print("\n" + "=" * 58)
        print(f"  Round {round_num + 1}/{args.rounds}")
        print("=" * 58)

        # ---- Step 1: wait for FedAvg of the previous round, then download ----
        # wait_for_round=round_num means "give me the model AFTER FedAvg round
        # (round_num-1) has completed" which is the correct starting point.
        # At round 0 this is immediately satisfied (initial model).
        print("  Fetching global model...")
        global_weights = fetch_global_model(args.server, wait_for_round=round_num)
        model.load_state_dict(global_weights)

        # ---- Step 2: configure trainable layers ----
        if args.freeze_backbone:
            model.freeze_for_client(train_last_n_blocks=3)
        else:
            model.unfreeze_all()

        trainable = model.count_trainable_params()
        print(f"  Trainable params: {trainable:,}")

        # Build optimizer only over trainable params
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, momentum=0.9, weight_decay=1e-4,
        )

        # ---- Step 3: local training (no network calls during training) ----
        print(f"  Local training: {args.epochs} epoch(s), "
              f"{len(train_loader)} batch(es)/epoch")

        for epoch in range(args.epochs):
            avg_loss = train_epoch(model, train_loader, optimizer,
                                   criterion, device, epoch, args.epochs)
            print(f"    Epoch {epoch + 1} done  avg_loss={avg_loss:.4f}")

        # ---- Step 4: submit weights to server (non-blocking) ----
        print("  Submitting weights to server...")
        submit_weights(args.server, args.client_id, model.state_dict(), n_train, round_num)

        # ---- Step 5: optional validation ----
        if args.evaluate and val_loader is not None:
            acc = evaluate(model, val_loader, device)
            print(f"  Validation accuracy: {acc:.2f}%")

        print(f"  Round {round_num + 1} complete.")

    print(f"\nClient {args.client_id} finished all {args.rounds} rounds.")


if __name__ == "__main__":
    main()
