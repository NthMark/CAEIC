# Federated Learning — MVP

A minimal, working Federated Learning setup across **3 PCs** using FedAvg.

```
PC 1 (Server)  ←──────────────────────────────────────────────────┐
       ↑  GET /get_model          POST /submit_weights             │
       │                                                           │
PC 2 (Client 1) ── trains on local data partition ── sends weights ┤
PC 3 (Client 2) ── trains on local data partition ── sends weights ┘
```

---

## Architecture

| File | Where it runs | Role |
|---|---|---|
| `model.py`  | All 3 PCs | Shared neural network definition |
| `server.py` | PC 1 | Aggregates weights via FedAvg |
| `client.py` | PC 2 & PC 3 | Trains locally, communicates with server |

### FL Round (repeated N times)
1. Client downloads global model  →  `GET /get_model`
2. Client trains on its local data
3. Client uploads updated weights  →  `POST /submit_weights`
4. Server runs **FedAvg** when all clients have submitted
5. Server increments round counter → repeat

---

## Setup (all 3 PCs)

```bash
pip install -r requirements.txt
```

Copy these files to **each PC**:
- `model.py`
- `requirements.txt`
- `server.py`  (PC 1 only)
- `client.py`  (PC 2 & PC 3 only)

---

## Running

### PC 1 — Server

```bash
python server.py --clients 2 --rounds 10 --port 5000
```

### PC 2 — Client 1

```bash
python client.py --client_id 1 --server http://<PC1_IP>:5000 --rounds 10
```

### PC 3 — Client 2

```bash
python client.py --client_id 2 --server http://<PC1_IP>:5000 --rounds 10
```

> Replace `<PC1_IP>` with the actual IP address of PC 1 (e.g. `192.168.1.100`).
> Find it with `ipconfig` (Windows) or `ip a` (Linux).

---

## Optional flags

### client.py
| Flag | Default | Description |
|---|---|---|
| `--client_id` | required | Unique ID for this client (1 or 2) |
| `--server` | `http://localhost:5000` | Server URL |
| `--rounds` | 10 | Number of FL rounds |
| `--epochs` | 2 | Local training epochs per round |
| `--lr` | 0.01 | Learning rate |
| `--evaluate` | off | Print test accuracy after each round |

### server.py
| Flag | Default | Description |
|---|---|---|
| `--clients` | 2 | Number of edge clients to wait for |
| `--rounds` | 10 | Total FL rounds |
| `--port` | 5000 | Listening port |

---

## What happens after training?

The final global model is saved to `global_model_final.pth` on the server.

Load it later:

```python
from model import SimpleNet
import torch

model = SimpleNet()
model.load_state_dict(torch.load("global_model_final.pth"))
model.eval()
```

---

## Firewall note (Windows)

Allow port 5000 on PC 1:

```
Windows Defender Firewall → Inbound Rules → New Rule → Port → TCP 5000 → Allow
```

Or quickly via PowerShell (run as Administrator):

```powershell
New-NetFirewallRule -DisplayName "FL Server" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow
```
