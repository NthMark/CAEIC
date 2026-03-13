"""
server.py  —  Federated Learning Server (run on PC 1)
─────────────────────────────────────────────────────
Responsibilities:
  1. Serve the current global model to clients  (GET  /get_model)
  2. Accept local weight updates from clients   (POST /submit_weights)
  3. Run FedAvg when all clients have submitted
  4. Expose a status endpoint                  (GET  /status)

Usage:
    python server.py
    python server.py --clients 2 --rounds 10 --port 5000
"""

import argparse
import base64
import copy
import io
import threading

import torch
from flask import Flask, jsonify, request

from model import SimpleNet

# ─────────────────────────────────────── app & shared state ──
app = Flask(__name__)

# Populated by CLI args in main()
NUM_CLIENTS: int = 2
TOTAL_ROUNDS: int = 10

global_model = SimpleNet()
client_weights: dict = {}        # { client_id: state_dict }
current_round: int = 0
lock = threading.Lock()


# ─────────────────────────────────────── helpers ─────────────

def serialize_model(model: torch.nn.Module) -> str:
    """State dict  →  base64 string (JSON-safe)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def deserialize_weights(encoded: str) -> dict:
    """base64 string  →  state dict."""
    buf = io.BytesIO(base64.b64decode(encoded.encode("utf-8")))
    return torch.load(buf, map_location="cpu")


def fedavg(weights_list: list) -> dict:
    """
    Federated Averaging  (equal weight per client).
    Returns a new averaged state dict.
    """
    avg = copy.deepcopy(weights_list[0])
    for key in avg:
        for i in range(1, len(weights_list)):
            avg[key] = avg[key] + weights_list[i][key]
        avg[key] = torch.div(avg[key], len(weights_list))
    return avg


# ─────────────────────────────────────── routes ──────────────

@app.route("/get_model", methods=["GET"])
def get_model():
    """Clients call this to download the current global model."""
    with lock:
        return jsonify({
            "round": current_round,
            "total_rounds": TOTAL_ROUNDS,
            "model": serialize_model(global_model),
        })


@app.route("/submit_weights", methods=["POST"])
def submit_weights():
    """
    Client posts:
      { "client_id": int, "round": int, "weights": <base64> }
    Server aggregates when all NUM_CLIENTS have submitted.
    """
    global current_round, global_model

    data = request.get_json(force=True)
    client_id   = data["client_id"]
    client_round = data["round"]
    encoded     = data["weights"]

    with lock:
        # ── round validation ──────────────────────────────────
        if client_round != current_round:
            return jsonify({
                "status": "error",
                "message": (
                    f"Round mismatch. Server is on round {current_round}, "
                    f"client sent round {client_round}."
                ),
            }), 400

        # ── store weights ─────────────────────────────────────
        client_weights[client_id] = deserialize_weights(encoded)
        received = len(client_weights)
        print(
            f"[Round {current_round:02d}] Client {client_id} submitted  "
            f"({received}/{NUM_CLIENTS})"
        )

        # ── aggregate when all clients have submitted ─────────
        if received == NUM_CLIENTS:
            aggregated = fedavg(list(client_weights.values()))
            global_model.load_state_dict(aggregated)
            client_weights.clear()
            current_round += 1
            print(f"✅  FedAvg done  →  starting round {current_round}\n")

            if current_round >= TOTAL_ROUNDS:
                # Optionally save the final model
                torch.save(global_model.state_dict(), "global_model_final.pth")
                print("🏁  All rounds complete. Final model saved to global_model_final.pth")

            return jsonify({
                "status": "aggregated",
                "new_round": current_round,
            })

        return jsonify({
            "status": "received",
            "clients_submitted": received,
            "waiting_for": NUM_CLIENTS - received,
        })


@app.route("/status", methods=["GET"])
def status():
    """Quick health-check / progress endpoint."""
    with lock:
        return jsonify({
            "current_round": current_round,
            "total_rounds": TOTAL_ROUNDS,
            "clients_submitted": len(client_weights),
            "waiting_for": NUM_CLIENTS - len(client_weights),
            "training_complete": current_round >= TOTAL_ROUNDS,
        })


# ─────────────────────────────────────── entry point ─────────

def main():
    global NUM_CLIENTS, TOTAL_ROUNDS

    parser = argparse.ArgumentParser(description="FL Server")
    parser.add_argument("--clients", type=int,   default=2,    help="Number of edge clients (default: 2)")
    parser.add_argument("--rounds",  type=int,   default=10,   help="Total FL rounds (default: 10)")
    parser.add_argument("--port",    type=int,   default=5000, help="Server port (default: 5000)")
    args = parser.parse_args()

    NUM_CLIENTS   = args.clients
    TOTAL_ROUNDS  = args.rounds

    print("=" * 50)
    print("       Federated Learning Server")
    print("=" * 50)
    print(f"  Clients expected : {NUM_CLIENTS}")
    print(f"  Total rounds     : {TOTAL_ROUNDS}")
    print(f"  Listening on     : 0.0.0.0:{args.port}")
    print("=" * 50)

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
