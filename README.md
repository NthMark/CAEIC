# Federated Learning — Plant Disease Detection

A federated learning setup across **3 PCs** using FedAvg to train a
**MobileNetV2** model on the **PlantVillage** dataset (38 plant disease classes).

```
PC 1 (Server)  <──────────────────────────────────────────────────┐
       ^  GET /get_model          POST /submit_weights             │
       │                                                           │
PC 2 (Client 1) ── trains on local data partition ── sends weights ┤
PC 3 (Client 2) ── trains on local data partition ── sends weights ┘
```

---

## Architecture

| File | Where it runs | Role |
|---|---|---|
| `model.py`     | All 3 PCs   | MobileNetV2 fine-tuned for 38 plant disease classes |
| `server.py`    | PC 1        | Aggregates weights via FedAvg |
| `client.py`    | PC 2 & PC 3 | Trains locally on PlantVillage partition |
| `evaluate.py`  | Any PC      | Evaluates checkpoint accuracy on the full dataset |
| `infer.py`     | Any PC      | Runs inference on dataset samples or your own images |

### FL Round (repeated N times)
1. Client downloads global model  →  `GET /get_model`
2. Client trains locally on its PlantVillage partition
3. Client uploads updated weights  →  `POST /submit_weights`
4. Server runs **FedAvg** when all clients have submitted
5. Server increments round counter → repeat

---

## Dataset — PlantVillage

38 classes covering healthy and diseased leaves across multiple plant species.

> **Only client PCs (PC 2 & PC 3) need the dataset.** The server never touches training data — it only aggregates weight tensors.

### Download (PC 2 & PC 3 only)
```cmd
pip install kaggle
python -m kaggle datasets download -d abdallahalidev/plantvillage-dataset
tar -xf plantvillage-dataset.zip
```

> **Kaggle API key required:** Go to kaggle.com → Settings → API → Create New Token.
> Place `kaggle.json` in `C:\Users\<YourUsername>\.kaggle\kaggle.json`.

Extract so the folder structure is:
```
plantvillage/
    Apple___Apple_scab/
    Apple___Black_rot/
    Apple___Cedar_apple_rust/
    Apple___healthy/
    ...  (38 folders total)
```

Copy the `plantvillage/` folder to **each client PC**.

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

> Run as **Administrator** so the firewall rule for port 5000 is created automatically.

### PC 2 — Client 1

```bash
python client.py --client_id 1 --server http://<PC1_IP>:5000 --data_dir ./plantvillage --rounds 10
```

### PC 3 — Client 2

```bash
python client.py --client_id 2 --server http://<PC1_IP>:5000 --data_dir ./plantvillage --rounds 10
```

> Replace `<PC1_IP>` with the actual IP of PC 1.
> Find it with `ipconfig` (Windows) or `ip a` (Linux).

---

## Optional flags

### client.py
| Flag | Default | Description |
|---|---|---|
| `--client_id`   | required | Unique ID for this client (1 or 2) |
| `--server`      | `http://localhost:5000` | Server URL |
| `--data_dir`    | required | Path to PlantVillage dataset folder |
| `--rounds`      | 10 | Number of FL rounds |
| `--epochs`      | 2  | Local training epochs per round |
| `--lr`          | 0.001 | SGD learning rate |
| `--batch_size`  | 32 | Training batch size |
| `--num_clients` | 2  | Total number of clients |
| `--evaluate`    | off | Run validation accuracy after each round |

### server.py
| Flag | Default | Description |
|---|---|---|
| `--clients` | 2    | Number of edge clients to wait for |
| `--rounds`  | 10   | Total FL rounds |
| `--port`    | 5000 | Listening port |

---

## Evaluate the checkpoint

After training, `global_model_final.pth` is saved on the server. Evaluate it:

```bash
# Overall accuracy
python evaluate.py --data_dir ./plantvillage

# With per-class breakdown (all 38 classes)
python evaluate.py --data_dir ./plantvillage --per_class
```

### evaluate.py flags
| Flag | Default | Description |
|---|---|---|
| `--data_dir`    | required | Path to PlantVillage dataset |
| `--checkpoint`  | `global_model_final.pth` | Path to checkpoint |
| `--per_class`   | off | Show accuracy for each of the 38 classes |
| `--batch_size`  | 64 | Evaluation batch size |

---

## Inference

```bash
# 10 random samples from the dataset
python infer.py --data_dir ./plantvillage --samples 10

# Save images to ./test/ for visual inspection (filenames contain true/pred labels)
python infer.py --data_dir ./plantvillage --samples 20 --save

# One specific sample by dataset index
python infer.py --data_dir ./plantvillage --index 42

# Your own plant image
python infer.py --image leaf.jpg
python infer.py --image a.jpg b.jpg c.jpg
```

### infer.py flags
| Flag | Default | Description |
|---|---|---|
| `--data_dir`    | — | PlantVillage folder (mutually exclusive with `--image`) |
| `--image`       | — | One or more image file paths |
| `--checkpoint`  | `global_model_final.pth` | Path to checkpoint |
| `--samples`     | 10 | Number of random samples when using `--data_dir` |
| `--index`       | — | Specific dataset index to predict |
| `--save`        | off | Save output images to `--out_dir` |
| `--out_dir`     | `test` | Folder to save extracted images |

Saved filenames encode the result for easy visual inspection:
```
42_Apple___Apple_scab_pred-Apple___Apple_scab_OK_97pct.jpg
105_Tomato___healthy_pred-Tomato___Late_blight_WRONG_54pct.jpg
```

---

## Load the checkpoint manually

```python
from model import PlantNet
import torch

model = PlantNet(num_classes=38, pretrained=False)
model.load_state_dict(torch.load("global_model_final.pth", weights_only=True))
model.eval()
```

---

## Firewall note (Windows)

The server auto-creates a firewall rule when run as Administrator.
To add it manually:

```
Windows Defender Firewall → Inbound Rules → New Rule → Port → TCP 5000 → Allow
```

Or via PowerShell (run as Administrator):

```powershell
New-NetFirewallRule -DisplayName "FL Server" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow
```
```
