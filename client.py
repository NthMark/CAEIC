"""
client.py  --  EdgeFed Edge Client (run on PC 2 and PC 3)
----------------------------------------------------------
Implements Algorithm 1 (ClientCompute) from:
  "EdgeFed: Optimized Federated Learning Based on Edge Computing"
  Ye et al., IEEE Access 2020

CLIENT computes ONLY the low layers (features[0] of MobileNetV2):
  Input (B,3,224,224) -> x_conv (B,32,112,112) -> sent to server

Server aggregates activations from all clients, trains the high layers,
and returns updated full model weights.  Client updates its model and
moves to the next batch.

No GPU required on the client.  Only one tiny forward pass per batch.

Usage:
    python client.py --client_id 1 --server http://<PC1_IP>:5000 --data_dir ./plantvillage
    python client.py --client_id 2 --server http://<PC1_IP>:5000 --data_dir ./plantvillage

Optional flags:
    --rounds      10        FL rounds to run (must match server)
    --epochs      2         Local epochs per round
    --batch_size  32        Batch size
    --num_clients 2         Total number of clients (for data partitioning)
    --num_classes 38        Number of output classes
    --evaluate              Run validation accuracy check after each round
"""

import argparse
import base64
import io
import time

import torch
import requests
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from model import PlantNet, NUM_CLASSES


# ------------------------------------------------------------------ serialization

def serialize_tensor(tensor: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(tensor, buf)
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
    """Split the dataset evenly across clients by index."""
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
    print(f"  Data partition: samples {start}..{end-1}  "
          f"({end - start} samples, {len(full.classes)} classes)")
    return loader, full.classes


def get_val_loader(data_dir: str, batch_size: int = 64) -> DataLoader:
    dataset = datasets.ImageFolder(root=data_dir, transform=VAL_TRANSFORM)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)


# ------------------------------------------------------------------ evaluation

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader) -> float:
    model.eval()
    device = next(model.parameters()).device
    correct = total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        pred     = model(data).argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total   += target.size(0)
    return 100.0 * correct / total


# ------------------------------------------------------------------ server comms

def fetch_global_model(server_url: str, expected_round: int,
                       retry_delay: int = 3) -> dict:
    """
    GET /get_model  -- retry until server is at the expected round.
    Returns: state_dict
    """
    while True:
        try:
            resp = requests.get(f"{server_url}/get_model", timeout=30)
            resp.raise_for_status()
            body = resp.json()
            if body["round"] == expected_round:
                return deserialize_weights(body["model"])
            print(f"  Server at round {body['round']}, waiting for round {expected_round}...")
        except Exception as exc:
            print(f"  GET /get_model failed: {exc}")
        time.sleep(retry_delay)


def submit_activations(server_url: str, client_id: int,
                       round_num: int, epoch: int, batch_id: int,
                       total_epochs: int, total_batches: int,
                       x_conv: torch.Tensor, labels: torch.Tensor,
                       retry_delay: int = 5) -> dict:
    """
    POST /submit_activations -- send low-layer output for one batch.
    Blocks until the server has collected all clients' activations,
    trained the high layers, and returned the updated model.
    Returns: {"status": "batch_done", "weights": <b64>, "round": <int>}
    """
    payload = {
        "client_id":     client_id,
        "round":         round_num,
        "epoch":         epoch,
        "batch_id":      batch_id,
        "total_epochs":  total_epochs,
        "total_batches": total_batches,
        "x_conv":        serialize_tensor(x_conv.cpu()),
        "labels":        serialize_tensor(labels.cpu()),
    }
    while True:
        try:
            resp = requests.post(
                f"{server_url}/submit_activations",
                json=payload,
                timeout=660,   # server may wait up to 10 min for other clients
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            print(f"  POST /submit_activations failed: {exc}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description="EdgeFed Client")
    parser.add_argument("--client_id",   type=int, required=True,
                        help="Unique client ID (1-based, e.g. 1 or 2)")
    parser.add_argument("--server",      type=str, default="http://localhost:5000",
                        help="Server URL (e.g. http://192.168.0.102:5000)")
    parser.add_argument("--data_dir",    type=str, required=True,
                        help="Path to PlantVillage dataset root (ImageFolder layout)")
    parser.add_argument("--rounds",      type=int, default=10,
                        help="Number of FL rounds (default: 10)")
    parser.add_argument("--epochs",      type=int, default=2,
                        help="Local epochs per round (default: 2)")
    parser.add_argument("--batch_size",  type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--num_clients", type=int, default=2,
                        help="Total number of clients for data partitioning (default: 2)")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES,
                        help=f"Number of output classes (default: {NUM_CLASSES})")
    parser.add_argument("--evaluate",    action="store_true",
                        help="Run validation accuracy check after each round")
    args = parser.parse_args()

    # Use GPU if available; EdgeFed only runs forward_low on the client so
    # even CPU is fast enough, but GPU is used if present.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print(f"    EdgeFed Client  (id={args.client_id})")
    print("=" * 55)
    print(f"  Server       : {args.server}")
    print(f"  Data dir     : {args.data_dir}")
    print(f"  Rounds       : {args.rounds}")
    print(f"  Epochs/round : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Device       : {device}")
    print("=" * 55)

    train_loader, class_names = get_data_partition(
        args.data_dir, args.client_id, args.num_clients, args.batch_size
    )
    total_batches = len(train_loader)
    val_loader = get_val_loader(args.data_dir, args.batch_size) if args.evaluate else None

    model = PlantNet(num_classes=args.num_classes, pretrained=False).to(device)

    for round_num in range(args.rounds):
        print(f"\n{'--' * 22}")
        print(f"  Round {round_num + 1}/{args.rounds}")
        print(f"{'--' * 22}")

        # ---- Step 1: download current global model ----
        print("  Fetching global model from server...")
        global_weights = fetch_global_model(args.server, round_num)
        model.load_state_dict(global_weights)
        print("  Global model loaded.")

        # ---- Step 2: EdgeFed collaborative training ----
        #   Client computes ONLY features[0] (first conv block) per batch.
        #   All heavy compute (features[1:] + backward + SGD) is on the server.
        print(f"  EdgeFed training  ({args.epochs} epoch(s), {total_batches} batch(es)/epoch)")

        for epoch in range(args.epochs):
            pbar = tqdm(
                enumerate(train_loader),
                total=total_batches,
                desc=f"    Epoch {epoch + 1}/{args.epochs}",
                unit="batch",
            )

            for batch_id, (data, target) in pbar:
                data   = data.to(device)
                target = target.to(device)

                # CLIENT: forward only through the low layers (features[0])
                # No gradient computation needed -- no backward on client side.
                with torch.no_grad():
                    x_conv = model.forward_low(data)

                # SERVER: receives x_conv + labels from all clients,
                #         trains high layers, returns updated full model.
                result = submit_activations(
                    args.server, args.client_id,
                    round_num, epoch, batch_id,
                    args.epochs, total_batches,
                    x_conv, target,
                )

                # Sync model with updated weights from server
                if result.get("weights"):
                    model.load_state_dict(deserialize_weights(result["weights"]))
                    pbar.set_postfix(server_round=result.get("round", "?"))

            print(f"    Epoch {epoch + 1}/{args.epochs} done.")

        # ---- Step 3: optional validation ----
        if args.evaluate and val_loader is not None:
            acc = evaluate(model, val_loader)
            print(f"  Validation accuracy after round {round_num + 1}: {acc:.2f}%")

        # No explicit weight submission needed:
        # Server already holds the trained high-layer weights from this round.
        print(f"  Round {round_num + 1} complete.")

    print(f"\nClient {args.client_id} finished all {args.rounds} rounds.")


if __name__ == "__main__":
    main()
