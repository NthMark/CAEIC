"""Helper to write corrected server.py"""
content = '''\
"""
server.py  --  Federated Learning Aggregation Server (run on PC 1)
-------------------------------------------------------------------
Correct FL architecture:

  PURPOSE: Training computation is DISTRIBUTED across clients.
           The server's role is to AGGREGATE client updates (FedAvg),
           not to train independently. If the server trained on its own,
           clients would be irrelevant -- defeating FL entirely.

  SERVER  : Pure FedAvg aggregator. Collects trained weights from all
            clients each round, averages them (weighted by dataset size),
            updates the global model, serves it back.

  CLIENTS : Each holds a partition of the data. They do ALL the training
            work on their local data, then submit weights once per round.

  ASYNC inside a round:
    - Clients train LOCALLY with no network dependency whatsoever.
    - Server responds IMMEDIATELY to /submit_weights (no blocking the client).
    - Server waits for ALL clients to submit, then runs FedAvg and
      advances the global model.
    - Clients fetch the new model at their own pace for the next round.

  This gives the best of both worlds:
    - No per-batch synchronization (849 batches x N clients = no problem).
    - Natural round-level checkpoint where FedAvg happens.
    - Training truly distributed: server CPU/GPU not needed for training.

Per-round flow:
  Client (all in parallel, independently):
    train N epochs locally -> POST /submit_weights  (returns immediately)
    -> GET /get_model when ready -> train next round

  Server:
    collect submissions -> when ALL clients submitted: FedAvg -> round++

Endpoints:
  GET  /get_model       Returns current global model. Non-blocking.
  POST /submit_weights  Queue client weights. Returns immediately.
                        FedAvg fires automatically when all clients submit.
  GET  /status          Round progress and pending submissions.

Usage:
    python server.py
    python server.py --clients 2 --rounds 10 --port 5000
"""

import argparse
import base64
import io
import subprocess
import sys
import threading

import torch
import torch.nn as nn
from flask import Flask, jsonify, request

from model import PlantNet, NUM_CLASSES

# ------------------------------------------------------------------ app setup
app = Flask(__name__)

# Set by CLI args in main()
NUM_CLIENTS  = 2
TOTAL_ROUNDS = 10

global_model = None   # PlantNet -- updated by FedAvg each round
model_lock   = threading.RLock()   # protects global_model reads/writes

current_round = 0
training_done = False

# Per-round client submissions: {client_id: (state_dict, n_samples)}
_round_submissions  = {}
_submissions_lock   = threading.Lock()


# ------------------------------------------------------------------ serialization

def serialize_model(m: torch.nn.Module) -> str:
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def deserialize_weights(encoded: str) -> dict:
    buf = io.BytesIO(base64.b64decode(encoded.encode("utf-8")))
    return torch.load(buf, map_location="cpu", weights_only=True)


# ------------------------------------------------------------------ FedAvg

def _fedavg(submissions: dict) -> None:
    """
    Weighted FedAvg: average client weights proportional to dataset size.
    global[key] = sum(n_i * w_i[key]) / sum(n_i)

    Args:
        submissions: {client_id: (state_dict, n_samples)}
    """
    total_samples = sum(n for _, (_, n) in submissions.items())
    averaged = {}

    keys = list(next(iter(submissions.values()))[0].keys())
    for key in keys:
        weighted_sum = sum(
            (n / total_samples) * sd[key].float()
            for _, (sd, n) in submissions.items()
        )
        averaged[key] = weighted_sum

    global_model.load_state_dict(averaged)
    print(f"[FedAvg] Aggregated {len(submissions)} clients "
          f"({total_samples} total samples)")


# ------------------------------------------------------------------ routes

@app.route("/get_model", methods=["GET"])
def get_model():
    """Returns the current global model. Always responds immediately."""
    with model_lock:
        return jsonify({
            "round":        current_round,
            "total_rounds": TOTAL_ROUNDS,
            "done":         training_done,
            "model":        serialize_model(global_model),
        })


@app.route("/submit_weights", methods=["POST"])
def submit_weights():
    """
    Receive trained weights from a client.

    Responds IMMEDIATELY -- client is never blocked.
    When the submission from the LAST expected client arrives,
    FedAvg runs synchronously in this thread, the global model
    is updated, and the round counter advances.
    """
    global current_round, training_done

    data       = request.get_json(force=True)
    client_id  = int(data["client_id"])
    n_samples  = int(data.get("n_samples", 1000))
    round_num  = int(data.get("round", current_round))
    state_dict = deserialize_weights(data["weights"])

    with _submissions_lock:
        # Ignore stale submissions from a previous round
        if round_num != current_round:
            return jsonify({
                "status":  "ignored",
                "reason":  f"Round mismatch: server={current_round}, client={round_num}",
                "round":   current_round,
            })

        _round_submissions[client_id] = (state_dict, n_samples)
        received = len(_round_submissions)
        print(f"[Round {current_round}] Received weights from client {client_id} "
              f"({received}/{NUM_CLIENTS}  n_samples={n_samples})")

        if received >= NUM_CLIENTS:
            # All clients submitted -- run FedAvg now
            with model_lock:
                _fedavg(dict(_round_submissions))

            _round_submissions.clear()
            current_round += 1

            if current_round >= TOTAL_ROUNDS:
                training_done = True
                with model_lock:
                    torch.save(global_model.state_dict(), "global_model_final.pth")
                print(f"[Server] All {TOTAL_ROUNDS} rounds complete. "
                      f"Final model saved to global_model_final.pth")
            else:
                print(f"[Server] Round complete -> now at round {current_round}/{TOTAL_ROUNDS}")

    return jsonify({
        "status": "accepted",
        "round":  current_round,
    })


@app.route("/status", methods=["GET"])
def status():
    with _submissions_lock:
        pending = list(_round_submissions.keys())
    return jsonify({
        "current_round":      current_round,
        "total_rounds":       TOTAL_ROUNDS,
        "training_complete":  training_done,
        "submitted_this_round": pending,
        "waiting_for":        [i for i in range(1, NUM_CLIENTS + 1)
                               if i not in pending],
    })


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
        print(f"  Firewall rule \'{rule_name}\' already exists.")
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
    global NUM_CLIENTS, TOTAL_ROUNDS, global_model

    parser = argparse.ArgumentParser(description="FL Aggregation Server")
    parser.add_argument("--clients",     type=int, default=2,
                        help="Number of clients (default: 2)")
    parser.add_argument("--rounds",      type=int, default=10,
                        help="Number of FL rounds (default: 10)")
    parser.add_argument("--port",        type=int, default=5000,
                        help="Listening port (default: 5000)")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES,
                        help=f"Output classes (default: {NUM_CLASSES})")
    args = parser.parse_args()

    NUM_CLIENTS  = args.clients
    TOTAL_ROUNDS = args.rounds

    # Pretrained ImageNet weights as the starting global model.
    # Clients will fine-tune from this checkpoint each round.
    global_model = PlantNet(num_classes=args.num_classes, pretrained=True)
    global_model.eval()

    print("=" * 58)
    print("      Federated Learning Aggregation Server")
    print("=" * 58)
    print(f"  Role           : Pure FedAvg Aggregator")
    print(f"  Model          : MobileNetV2 (PlantNet, {args.num_classes} classes)")
    print(f"  Clients        : {NUM_CLIENTS}")
    print(f"  Rounds         : {TOTAL_ROUNDS}")
    print(f"  Port           : {args.port}")
    print(f"  FedAvg trigger : when all {NUM_CLIENTS} clients submit per round")
    print("=" * 58)

    ensure_firewall_rule(args.port)

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
'''
with open(r'c:\Users\Dell\Documents\CAEIC\CAEIC\server.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('server.py written OK')
