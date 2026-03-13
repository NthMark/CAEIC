"""
client.py  —  Federated Learning Edge Client (run on PC 2 and PC 3)
────────────────────────────────────────────────────────────────────
Each federated learning round:
  1. Download global model from server
  2. Train locally on this client's data partition
  3. Send updated weights back to server
  4. Repeat until all rounds are done

Usage (PC 2):
    python client.py --client_id 1 --server http://<SERVER_IP>:5000

Usage (PC 3):
    python client.py --client_id 2 --server http://<SERVER_IP>:5000

Optional flags:
    --rounds   10    (should match server setting)
    --epochs   2     (local training epochs per round)
    --lr       0.01
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

from model import SimpleNet


# ─────────────────────────────────────── serialization ───────

def serialize_weights(state_dict: dict) -> str:
    """State dict  →  base64 string."""
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def deserialize_weights(encoded: str) -> dict:
    """base64 string  →  state dict."""
    buf = io.BytesIO(base64.b64decode(encoded.encode("utf-8")))
    return torch.load(buf, map_location="cpu")


# ─────────────────────────────────────── data ────────────────

def get_data_partition(client_id: int, num_clients: int = 2):
    """
    Splits the MNIST training set evenly across clients.
    client_id is 1-based (1 or 2).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    total          = len(dataset)
    partition_size = total // num_clients
    start          = (client_id - 1) * partition_size
    end            = start + partition_size
    indices        = list(range(start, end))

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )
    print(f"📦  Data partition: samples {start}–{end - 1}  ({len(indices)} samples)")
    return loader


def get_test_loader():
    """Full MNIST test set – used only for optional local evaluation."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)


# ─────────────────────────────────────── training ────────────

def train_local(model, dataloader, epochs: int, lr: float) -> float:
    """
    Run `epochs` passes of SGD on the local dataset.
    Returns average loss over the last epoch.
    """
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    last_loss  = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        last_loss = running_loss / len(dataloader)
        print(f"    Epoch {epoch + 1}/{epochs}  loss={last_loss:.4f}")

    return last_loss


@torch.no_grad()
def evaluate(model, dataloader) -> float:
    """Returns accuracy (0–100) on the provided dataloader."""
    model.eval()
    correct = total = 0
    for data, target in dataloader:
        pred    = model(data).argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total   += target.size(0)
    return 100.0 * correct / total


# ─────────────────────────────────────── server comms ────────

def fetch_global_model(server_url: str, expected_round: int, retry_delay: int = 3):
    """
    Poll GET /get_model until the server is on `expected_round`.
    Returns the decoded state dict.
    """
    while True:
        try:
            resp = requests.get(f"{server_url}/get_model", timeout=10)
            resp.raise_for_status()
            body = resp.json()
            if body["round"] == expected_round:
                return deserialize_weights(body["model"])
            print(
                f"  ⏳  Server at round {body['round']}, "
                f"waiting for round {expected_round}…"
            )
        except Exception as exc:
            print(f"  ❌  GET /get_model failed: {exc}")
        time.sleep(retry_delay)


def submit_weights(server_url: str, client_id: int, round_num: int,
                   state_dict: dict, retry_delay: int = 5):
    """POST weights to server, retrying on failure."""
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
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            print(f"  ❌  POST /submit_weights failed: {exc}. Retrying in {retry_delay}s…")
            time.sleep(retry_delay)


# ─────────────────────────────────────── main ────────────────

def main():
    parser = argparse.ArgumentParser(description="FL Edge Client")
    parser.add_argument("--client_id",   type=int,   required=True,
                        help="Unique client ID (e.g. 1 or 2)")
    parser.add_argument("--server",      type=str,   default="http://localhost:5000",
                        help="Server URL  (default: http://localhost:5000)")
    parser.add_argument("--rounds",      type=int,   default=10,
                        help="Number of FL rounds  (default: 10)")
    parser.add_argument("--epochs",      type=int,   default=2,
                        help="Local training epochs per round  (default: 2)")
    parser.add_argument("--lr",          type=float, default=0.01,
                        help="SGD learning rate  (default: 0.01)")
    parser.add_argument("--num_clients", type=int,   default=2,
                        help="Total number of clients  (default: 2)")
    parser.add_argument("--evaluate",    action="store_true",
                        help="Run test-set evaluation after each round")
    args = parser.parse_args()

    print("=" * 50)
    print(f"    Federated Learning Client  (id={args.client_id})")
    print("=" * 50)
    print(f"  Server   : {args.server}")
    print(f"  Rounds   : {args.rounds}")
    print(f"  Epochs/r : {args.epochs}")
    print(f"  LR       : {args.lr}")
    print("=" * 50)

    # ── data ──────────────────────────────────────────────────
    train_loader = get_data_partition(args.client_id, args.num_clients)
    test_loader  = get_test_loader() if args.evaluate else None

    model = SimpleNet()

    # ── federated rounds ──────────────────────────────────────
    for round_num in range(args.rounds):
        print(f"\n{'─'*40}")
        print(f"  Round {round_num + 1}/{args.rounds}")
        print(f"{'─'*40}")

        # 1. Download global model
        print("  ⬇️   Fetching global model…")
        global_weights = fetch_global_model(args.server, round_num)
        model.load_state_dict(global_weights)
        print("  ✅  Global model loaded.")

        # 2. Local training
        print(f"  🏋️   Training locally for {args.epochs} epoch(s)…")
        avg_loss = train_local(model, train_loader, args.epochs, args.lr)
        print(f"  📉  Training loss: {avg_loss:.4f}")

        # Optional evaluation
        if args.evaluate and test_loader:
            acc = evaluate(model, test_loader)
            print(f"  🎯  Local model accuracy on test set: {acc:.2f}%")

        # 3. Submit weights
        print("  ⬆️   Submitting weights to server…")
        result = submit_weights(args.server, args.client_id, round_num, model.state_dict())
        print(f"  📬  Server response: {result}")

    print(f"\n🎉  Client {args.client_id} finished all {args.rounds} rounds!")


if __name__ == "__main__":
    main()
