"""
server.py  --  Federated Learning Server
-----------------------------------------
Correct FL architecture -- server is a PARTICIPANT, not a bystander:

  Clients train on their data partition -> submit weights once per round.
  Server trains on its own data partition -> submits weights once per round.
  FedAvg fires when ALL nodes (server + every client) have submitted.
  Global model is updated, round advances, everyone fetches it and repeats.

  If no --data_dir is given, server acts as a pure aggregator (no local
  training), and FedAvg fires when all clients submit.

Why this is correct FL:
  - Training is DISTRIBUTED: every node trains locally on private data.
  - Server SHARES the computation burden -- it does not idle.
  - No node waits during training. The only synchronisation point is
    the round boundary (FedAvg), which is O(1) network round-trips per
    round, not per batch.

Round flow:
  All nodes in parallel:
    1. GET /get_model?wait_for_round=N   (blocks until FedAvg round N-1 done)
    2. Train locally for --epochs epochs (zero network traffic)
    3. POST /submit_weights              (returns immediately)
  Server (automatic, after last submission arrives):
    4. Weighted FedAvg over all submissions
    5. current_round += 1  ->  notify waiting /get_model requests

Endpoints:
  GET  /get_model?wait_for_round=N   Block until round >= N, return model
  POST /submit_weights               Accept weights, trigger FedAvg if all in
  GET  /status                       Round info, who has submitted

Usage:
    python server.py                          # pure aggregator, 2 clients
    python server.py --data_dir ./plantvillage --clients 2 --rounds 10
    python server.py --data_dir ./plantvillage --clients 2 --rounds 10 --lr 0.001
"""

import argparse
import base64
import io
import subprocess
import sys
import threading
import time

import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, jsonify, request
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import PlantNet, NUM_CLASSES

# ------------------------------------------------------------------ app setup
app = Flask(__name__)

# Suppress Flask's per-request HTTP log lines (192.168.x.x - - "GET /..." 200 -)
# They interleave with tqdm progress bars and make output unreadable.
import logging
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)   # only show errors, not every request

# Set by CLI args in main()
NUM_CLIENTS   = 2
TOTAL_ROUNDS  = 10
SERVER_EPOCHS = 2
SERVER_LR     = 0.001
SERVER_BATCH  = 32

# ---- ALL shared mutable state is protected by a single Condition ----
# _state_cond acts as both a mutex and a notification channel.
# Condition.wait() atomically releases the lock and sleeps;
# notify_all() wakes all waiters after FedAvg advances the round.
_state_cond = threading.Condition()

global_model       = None   # PlantNet -- only updated by FedAvg
current_round      = 0
training_done      = False
_round_submissions = {}     # {client_id: (state_dict, n_samples)}
_server_weights    = None   # (state_dict, n_samples) once server is done, else None
_server_has_data   = False  # True when --data_dir was given


# ------------------------------------------------------------------ FedAvg core

def _fedavg(all_subs: dict) -> None:
    """
    Weighted FedAvg: global[key] = sum(n_i * w_i[key]) / sum(n_i)
    all_subs: {any_id: (state_dict, n_samples)}
    Modifies global_model in-place. Caller must hold _state_cond.
    """
    total_n = sum(n for _, (_, n) in all_subs.items())
    keys    = list(next(iter(all_subs.values()))[0].keys())
    new_sd  = {}
    for key in keys:
        new_sd[key] = sum(
            (n / total_n) * sd[key].float()
            for _, (sd, n) in all_subs.items()
        )
    global_model.load_state_dict(new_sd)
    ids = sorted(str(k) for k in all_subs.keys())
    print(f"[FedAvg] Round {current_round} | nodes={ids} | total_samples={total_n}")


def _check_and_run_fedavg() -> None:
    """
    If all expected submissions have arrived, run FedAvg and advance the round.
    Must be called while holding _state_cond.
    """
    global current_round, training_done, _server_weights

    clients_ready = len(_round_submissions) >= NUM_CLIENTS
    server_ready  = (not _server_has_data) or (_server_weights is not None)

    if not (clients_ready and server_ready):
        return  # Not everyone is in yet

    # Build the full submission pool
    all_subs = dict(_round_submissions)
    if _server_weights is not None:
        all_subs["server"] = _server_weights

    _fedavg(all_subs)

    # Reset per-round state
    _round_submissions.clear()
    _server_weights = None
    current_round  += 1

    if current_round >= TOTAL_ROUNDS:
        training_done = True
        torch.save(global_model.state_dict(), "global_model_final.pth")
        print(f"[Server] All {TOTAL_ROUNDS} rounds complete. Final model saved.")
    else:
        print(f"[Server] Round {current_round}/{TOTAL_ROUNDS} active.")

    _state_cond.notify_all()   # wake any /get_model?wait_for_round=N callers


# ------------------------------------------------------------------ serialization

def serialize_model(m: torch.nn.Module) -> str:
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def deserialize_weights(encoded: str) -> dict:
    buf = io.BytesIO(base64.b64decode(encoded.encode("utf-8")))
    return torch.load(buf, map_location="cpu", weights_only=True)


# ------------------------------------------------------------------ routes

@app.route("/get_model", methods=["GET"])
def get_model():
    """
    Returns the current global model.
    Optional query param: ?wait_for_round=N
      Blocks (server-side) until current_round >= N.
      Clients use this at the start of each round to wait for FedAvg to finish.
      The underlying Condition.wait() releases the lock while sleeping,
      so the Flask server remains fully responsive to other requests.
    """
    wait_for = request.args.get("wait_for_round", type=int, default=None)

    with _state_cond:
        if wait_for is not None:
            deadline = time.time() + 7200  # 2-hour max
            while current_round < wait_for and not training_done:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return jsonify({"status": "error",
                                    "message": "Timeout waiting for round"}), 504
                _state_cond.wait(timeout=min(30.0, remaining))

        return jsonify({
            "round":        current_round,
            "total_rounds": TOTAL_ROUNDS,
            "done":         training_done,
            "model":        serialize_model(global_model),
        })


@app.route("/submit_weights", methods=["POST"])
def submit_weights():
    """
    Receive trained weights from a client for the current round.
    Returns IMMEDIATELY -- FedAvg triggers automatically when all nodes submit.
    """
    data       = request.get_json(force=True)
    client_id  = int(data["client_id"])
    n_samples  = int(data.get("n_samples", 1000))
    round_num  = int(data.get("round", -1))
    state_dict = deserialize_weights(data["weights"])

    with _state_cond:
        if round_num != current_round:
            return jsonify({
                "status":  "ignored",
                "reason":  f"Round mismatch: server={current_round}, "
                           f"client submitted for round={round_num}",
                "round":   current_round,
            })

        _round_submissions[client_id] = (state_dict, n_samples)
        received = len(_round_submissions)
        print(f"[Round {current_round}] Client {client_id} submitted "
              f"({received}/{NUM_CLIENTS}, n_samples={n_samples})")

        _check_and_run_fedavg()

    return jsonify({"status": "accepted", "round": current_round})


@app.route("/status", methods=["GET"])
def status():
    with _state_cond:
        submitted  = list(_round_submissions.keys())
        svr_done   = _server_weights is not None
        rnd        = current_round
        done       = training_done

    waiting_clients = [i for i in range(1, NUM_CLIENTS + 1)
                       if i not in submitted]
    return jsonify({
        "current_round":         rnd,
        "total_rounds":          TOTAL_ROUNDS,
        "training_complete":     done,
        "clients_submitted":     submitted,
        "clients_waiting":       waiting_clients,
        "server_submitted":      svr_done,
        "server_trains_locally": _server_has_data,
    })


# ------------------------------------------------------------------ server training thread

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def server_training_loop(data_dir: str, num_classes: int) -> None:
    """
    Background daemon thread: server participates in FL as a regular node.

    For each round:
      1. Wait until current_round == round_num (ensures server uses the
         freshly FedAvg'd model and doesn't race ahead).
      2. Copy global model to a local model (snapshot without holding lock).
      3. Train locally for SERVER_EPOCHS epochs -- no locks held here.
      4. Submit trained weights; call _check_and_run_fedavg().

    The server is not special -- it trains and submits just like any client.
    It happens to ALSO run the FedAvg logic, but that's just a code role.
    """
    global _server_weights

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset   = datasets.ImageFolder(data_dir, transform=TRAIN_TRANSFORM)
    loader    = DataLoader(dataset, batch_size=SERVER_BATCH, shuffle=True,
                           num_workers=0, pin_memory=False)
    n_samples = len(dataset)
    criterion = nn.CrossEntropyLoss()

    local_model = PlantNet(num_classes=num_classes, pretrained=False).to(device)

    print(f"[Server trainer] dataset={n_samples} samples, "
          f"{len(loader)} batches/epoch, device={device}")

    for round_num in range(TOTAL_ROUNDS):
        # ---- 1. Wait for this round to become active ----
        with _state_cond:
            while current_round != round_num and not training_done:
                _state_cond.wait(timeout=10)
            if training_done:
                break
            # Snapshot global model (lock is held, snapshot is fast)
            local_model.load_state_dict(
                {k: v.clone() for k, v in global_model.state_dict().items()}
            )
            print(f"[Server trainer] Round {round_num}: starting local training")

        local_model.to(device)
        optimizer = optim.SGD(local_model.parameters(), lr=SERVER_LR,
                               momentum=0.9, weight_decay=1e-4)

        # ---- 2. Train locally (no lock held -- concurrent with clients) ----
        for epoch in range(SERVER_EPOCHS):
            local_model.train()
            total_loss = correct = total = 0

            pbar = tqdm(
                loader,
                desc=f"[Server] R{round_num + 1}/{TOTAL_ROUNDS} "
                     f"E{epoch + 1}/{SERVER_EPOCHS}",
                unit="batch",
                leave=True,
            )
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = local_model(data)
                loss   = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct    += output.argmax(1).eq(target).sum().item()
                total      += target.size(0)
                pbar.set_postfix(loss=f"{loss.item():.3f}",
                                 acc=f"{100 * correct / total:.1f}%")

            avg = total_loss / len(loader)
            print(f"[Server] R{round_num + 1} E{epoch + 1}/{SERVER_EPOCHS} "
                  f"avg_loss={avg:.4f}")

        # ---- 3. Submit weights for FedAvg ----
        cpu_sd = {k: v.cpu() for k, v in local_model.state_dict().items()}
        with _state_cond:
            # Guard: the round may have already advanced (unlikely but safe)
            if current_round == round_num:
                _server_weights = (cpu_sd, n_samples)
                print(f"[Server trainer] Round {round_num} done, "
                      f"submitted (n={n_samples})")
                _check_and_run_fedavg()

    print("[Server trainer] All rounds complete.")


# ------------------------------------------------------------------ firewall

def ensure_firewall_rule(port: int) -> None:
    if sys.platform != "win32":
        return
    rule_name = "FL Server"
    check = subprocess.run(
        ["netsh", "advfirewall", "firewall", "show", "rule", f"name={rule_name}"],
        capture_output=True, text=True,
    )
    if "No rules match" not in check.stdout and check.returncode == 0:
        print(f"  Firewall rule already exists.")
        return
    result = subprocess.run(
        ["netsh", "advfirewall", "firewall", "add", "rule",
         f"name={rule_name}", "dir=in", "action=allow",
         "protocol=TCP", f"localport={port}"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  Firewall rule added: TCP inbound port {port}")
    else:
        print("  Could not add firewall rule (run as Administrator).")


# ------------------------------------------------------------------ entry point

def main():
    global NUM_CLIENTS, TOTAL_ROUNDS, SERVER_EPOCHS, SERVER_LR, SERVER_BATCH
    global global_model, _server_has_data

    parser = argparse.ArgumentParser(description="FL FedAvg Server")
    parser.add_argument("--clients",     type=int,   default=2,
                        help="Number of remote clients (default: 2)")
    parser.add_argument("--rounds",      type=int,   default=10,
                        help="FL rounds (default: 10)")
    parser.add_argument("--port",        type=int,   default=5000,
                        help="Listening port (default: 5000)")
    parser.add_argument("--lr",          type=float, default=0.001,
                        help="SGD learning rate for server local training "
                             "(only used when --data_dir is given)")
    parser.add_argument("--epochs",      type=int,   default=2,
                        help="Server local training epochs per round "
                             "(only used when --data_dir is given)")
    parser.add_argument("--batch_size",  type=int,   default=32,
                        help="Server training batch size (default: 32)")
    parser.add_argument("--num_classes", type=int,   default=NUM_CLASSES,
                        help=f"Output classes (default: {NUM_CLASSES})")
    parser.add_argument("--data_dir",    type=str,   default=None,
                        help="Server data partition (ImageFolder layout). "
                             "If given, server trains locally and joins FedAvg. "
                             "If omitted, server is a pure aggregator.")
    args = parser.parse_args()

    NUM_CLIENTS   = args.clients
    TOTAL_ROUNDS  = args.rounds
    SERVER_EPOCHS = args.epochs
    SERVER_LR     = args.lr
    SERVER_BATCH  = args.batch_size
    _server_has_data = args.data_dir is not None

    global_model = PlantNet(num_classes=args.num_classes, pretrained=True)
    global_model.eval()

    mode = "Trainer + Aggregator" if _server_has_data else "Pure Aggregator"
    fedavg_trigger = (f"all {NUM_CLIENTS} clients"
                      + (" + server" if _server_has_data else "")
                      + " submit per round")

    print("=" * 62)
    print("    Federated Learning Server")
    print("=" * 62)
    print(f"  Mode           : {mode}")
    print(f"  Clients        : {NUM_CLIENTS}")
    print(f"  Rounds         : {TOTAL_ROUNDS}")
    print(f"  FedAvg trigger : {fedavg_trigger}")
    if _server_has_data:
        print(f"  Server data    : {args.data_dir}")
        print(f"  Server epochs  : {args.epochs}  LR={args.lr}")
    print(f"  Port           : {args.port}")
    print("=" * 62)

    ensure_firewall_rule(args.port)

    if _server_has_data:
        t = threading.Thread(
            target=server_training_loop,
            args=(args.data_dir, args.num_classes),
            daemon=True,
        )
        t.start()

    # threaded=True: each Flask request runs in its own thread,
    # allowing multiple /get_model?wait_for_round=N to sleep concurrently.
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
