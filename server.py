"""
server.py  --  EdgeFed Server (run on PC 1)
-------------------------------------------
Implements the EdgeFed architecture (Ye et al., 2020).

Role: Edge Server + Central Server (combined in our 2-client setup)

  GET  /get_model          Clients download the current global model
  POST /submit_activations Clients send per-batch low-layer activations
                           Server aggregates, trains HIGH LAYERS only,
                           returns updated model weights to all clients
  GET  /status             Health-check / progress

Key difference from FedAvg:
  - Clients NEVER do backprop (no GPU needed on client).
  - Server does ALL the heavy compute (forward_high + backward + SGD).
  - Per batch: server waits for all clients, aggregates x_conv, trains.

Usage:
    python server.py
    python server.py --clients 2 --rounds 10 --port 5000 --lr 0.001
"""

import argparse
import base64
import io
import subprocess
import sys
import threading

import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, jsonify, request

from model import PlantNet, NUM_CLASSES

# ------------------------------------------------------------------ app setup
app = Flask(__name__)

# Populated by CLI args in main()
NUM_CLIENTS  = 2
TOTAL_ROUNDS = 10
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

global_model = None  # PlantNet, initialized in main()
optimizer    = None  # SGD, initialized in main()
criterion    = nn.CrossEntropyLoss()
current_round = 0

# ------------------------------------------------------------------ per-batch sync

class BatchState:
    """
    Synchronization object for one (round, epoch, batch_id) triple.
    All NUM_CLIENTS threads block here; the last to arrive runs the
    training step, then wakes the others.
    """
    def __init__(self):
        self.contributions = {}       # {client_id: (x_conv_cpu, labels_cpu)}
        self.result_weights = None    # serialized full model state dict
        self.result_round   = None    # current_round at time of processing
        self.processed      = threading.Event()
        self.ref_count      = 0       # how many clients still need to read result
        self._lock          = threading.Lock()


_batch_states      = {}             # (round, epoch, batch_id) -> BatchState
_batch_states_lock = threading.Lock()
_round_lock        = threading.Lock()


# ------------------------------------------------------------------ helpers

def serialize_model(model: torch.nn.Module) -> str:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def deserialize_tensor(encoded: str) -> torch.Tensor:
    buf = io.BytesIO(base64.b64decode(encoded.encode("utf-8")))
    return torch.load(buf, map_location="cpu", weights_only=True)


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
        print(f"  Firewall rule '{rule_name}' already exists -- skipping.")
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
        print("  Could not add firewall rule (run as Administrator to auto-configure).")
        print(f"  Manual fix: New-NetFirewallRule -DisplayName 'FL Server' "
              f"-Direction Inbound -Protocol TCP -LocalPort {port} -Action Allow")


# ------------------------------------------------------------------ routes

@app.route("/get_model", methods=["GET"])
def get_model():
    with _round_lock:
        return jsonify({
            "round":        current_round,
            "total_rounds": TOTAL_ROUNDS,
            "model":        serialize_model(global_model),
        })


@app.route("/submit_activations", methods=["POST"])
def submit_activations():
    """
    Algorithm 1 -- EdgeUpdate (server side) + ClientCompute (client side).
    Clients POST:
      { client_id, round, epoch, batch_id, total_epochs, total_batches,
        x_conv: base64_tensor, labels: base64_tensor }
    Server:
      1. Stores activations from each client.
      2. When all NUM_CLIENTS have submitted the same batch, aggregates
         x_conv tensors, runs forward_high + backward + optimizer step.
      3. Returns updated full model weights to every waiting client.
    """
    global current_round

    data          = request.get_json(force=True)
    client_id     = data["client_id"]
    round_num     = data["round"]
    epoch         = data["epoch"]
    batch_id      = data["batch_id"]
    total_epochs  = data["total_epochs"]
    total_batches = data["total_batches"]

    x_conv = deserialize_tensor(data["x_conv"])
    labels = deserialize_tensor(data["labels"])

    key = (round_num, epoch, batch_id)

    # ---- register this client's contribution ----
    with _batch_states_lock:
        if round_num != current_round:
            return jsonify({
                "status":  "error",
                "message": f"Round mismatch: server={current_round}, client={round_num}",
            }), 400

        if key not in _batch_states:
            _batch_states[key] = BatchState()

        state = _batch_states[key]
        state.contributions[client_id] = (x_conv, labels)
        state.ref_count += 1
        received = len(state.contributions)

    should_process = (received == NUM_CLIENTS)

    if not should_process:
        # Wait for the last client to finish the training step (up to 10 min)
        fired = state.processed.wait(timeout=600)
        if not fired:
            return jsonify({"status": "error", "message": "Timeout waiting for other clients"}), 504
    else:
        # We are the last client -- aggregate and train high layers
        with _batch_states_lock:
            contribs = dict(state.contributions)

        # Concatenate activations and labels from all clients (sorted for reproducibility)
        all_x = torch.cat(
            [contribs[cid][0] for cid in sorted(contribs.keys())], dim=0
        ).to(DEVICE)
        all_y = torch.cat(
            [contribs[cid][1] for cid in sorted(contribs.keys())], dim=0
        ).to(DEVICE)

        # Algorithm 1 line 10: w <- w - lr * grad_loss(w; x_output, y_label)
        global_model.train()
        optimizer.zero_grad()
        logits = global_model.forward_high(all_x)
        loss   = criterion(logits, all_y)
        loss.backward()
        optimizer.step()

        is_last_batch = (batch_id == total_batches - 1)
        is_last_epoch = (epoch    == total_epochs  - 1)

        with _round_lock:
            if is_last_batch and is_last_epoch:
                current_round += 1
                print(f"[Round complete] -> round {current_round}/{TOTAL_ROUNDS}")
                if current_round >= TOTAL_ROUNDS:
                    torch.save(global_model.state_dict(), "global_model_final.pth")
                    print("Training complete. Final model saved to global_model_final.pth")

        state.result_weights = serialize_model(global_model)
        state.result_round   = current_round

        print(f"[R{round_num} E{epoch} B{batch_id:04d}] "
              f"loss={loss.item():.4f}  ({received}/{NUM_CLIENTS} clients)")

        # Wake up all waiting client threads
        state.processed.set()

    # ---- collect result then clean up ----
    weights      = state.result_weights
    result_round = state.result_round

    with _batch_states_lock:
        state.ref_count -= 1
        if state.ref_count == 0:
            del _batch_states[key]

    return jsonify({
        "status":  "batch_done",
        "weights": weights,
        "round":   result_round,
    })


@app.route("/status", methods=["GET"])
def status():
    with _round_lock:
        return jsonify({
            "current_round":     current_round,
            "total_rounds":      TOTAL_ROUNDS,
            "training_complete": current_round >= TOTAL_ROUNDS,
        })


# ------------------------------------------------------------------ entry point

def main():
    global NUM_CLIENTS, TOTAL_ROUNDS, global_model, optimizer

    parser = argparse.ArgumentParser(description="EdgeFed Server")
    parser.add_argument("--clients",     type=int,   default=2,
                        help="Number of edge clients (default: 2)")
    parser.add_argument("--rounds",      type=int,   default=10,
                        help="Total FL rounds (default: 10)")
    parser.add_argument("--port",        type=int,   default=5000,
                        help="Listening port (default: 5000)")
    parser.add_argument("--lr",          type=float, default=0.001,
                        help="SGD learning rate for high-layer training (default: 0.001)")
    parser.add_argument("--num_classes", type=int,   default=NUM_CLASSES,
                        help=f"Number of output classes (default: {NUM_CLASSES})")
    args = parser.parse_args()

    NUM_CLIENTS  = args.clients
    TOTAL_ROUNDS = args.rounds

    global_model = PlantNet(num_classes=args.num_classes, pretrained=True).to(DEVICE)
    optimizer    = optim.SGD(
        global_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
    )

    print("=" * 55)
    print("          EdgeFed Server")
    print("=" * 55)
    print(f"  Method         : EdgeFed (computation offloading)")
    print(f"  Device         : {DEVICE}")
    print(f"  Clients        : {NUM_CLIENTS}")
    print(f"  Rounds         : {TOTAL_ROUNDS}")
    print(f"  LR (high layers): {args.lr}")
    print(f"  Listening on   : 0.0.0.0:{args.port}")
    print("=" * 55)

    ensure_firewall_rule(args.port)

    # threaded=True is required so multiple client requests are handled concurrently
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
