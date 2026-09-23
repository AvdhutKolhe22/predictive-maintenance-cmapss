from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
from flask import Flask, jsonify, render_template
from src.model import CNNBiLSTMMultiTask

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "raw" / "test_FD001.txt"
MODEL_DIR = ROOT / "models"

WINDOW = 30

app = Flask(__name__)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

COLUMNS = (
    ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

try:
    DATA = pd.read_csv(
        RAW_FILE,
        sep=r"\s+",
        header=None,
        names=COLUMNS
    )

    with open(MODEL_DIR / "feature_config.json", "r") as f:
        FEATURES = json.load(f)["all_features"]

    SCALER = joblib.load(
        MODEL_DIR / "feature_scaler.joblib"
    )

except Exception as exc:
    DATA = None
    FEATURES = None
    SCALER = None
    STARTUP_ERROR = str(exc)


def load_model(path):

    model = CNNBiLSTMMultiTask(
        num_features=16,
        cnn_filters=64,
        lstm_hidden=64,
        lstm_layers=2,
        dropout=0.2
    ).to(DEVICE)

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):
        state = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint)
        )
    else:
        state = checkpoint

    model.load_state_dict(state)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


try:

    V2 = load_model(
        MODEL_DIR / "cnn_bilstm_multitask_v2_best.pt"
    )

    V3 = load_model(
        MODEL_DIR / "cnn_bilstm_multitask_v3_best.pt"
    )

except Exception as exc:

    V2 = None
    V3 = None
    MODEL_ERROR = str(exc)


def get_status(rul, probability):

    if rul <= 30 or probability >= 50:
        return "CRITICAL"

    if rul <= 50 or probability >= 20:
        return "WARNING"

    return "NORMAL"


def get_recommendation(status):

    if status == "CRITICAL":
        return "Schedule maintenance immediately."

    if status == "WARNING":
        return "Increase monitoring and plan maintenance."

    return "Continue normal operation."


def predict(window):

    values = window[FEATURES].to_numpy(
        dtype=np.float32
    )

    values = SCALER.transform(values)

    tensor = torch.from_numpy(
        values.astype(np.float32)
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        rul_output = V2(tensor)
        failure_output = V3(tensor)

        rul = float(
            rul_output["rul"]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)[0]
        )

        probability = float(
            torch.sigmoid(
                failure_output["failure_logits"]
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)[0]
        )

    return max(0.0, rul), probability * 100


def analyze_engine(engine_id):

    if DATA is None:
        raise RuntimeError(
            "Dataset could not be loaded: "
            + STARTUP_ERROR
        )

    if V2 is None or V3 is None:
        raise RuntimeError(
            "Models could not be loaded: "
            + MODEL_ERROR
        )

    engine = DATA[
        DATA["unit_id"] == int(engine_id)
    ].sort_values("cycle").reset_index(drop=True)

    if engine.empty:
        raise ValueError(
            f"Engine {engine_id} does not exist."
        )

    if len(engine) < WINDOW:
        raise ValueError(
            f"Engine {engine_id} has only "
            f"{len(engine)} cycles."
        )

    trajectory = []

    for end in range(
        WINDOW,
        len(engine) + 1
    ):

        window = engine.iloc[
            end - WINDOW:end
        ]

        rul, probability = predict(window)

        status = get_status(
            rul,
            probability
        )

        trajectory.append(
            {
                "cycle": int(
                    window["cycle"].iloc[-1]
                ),
                "rul": round(rul, 2),
                "failure_probability": round(
                    probability,
                    2
                ),
                "status": status
            }
        )

    latest = trajectory[-1]

    return {
        "engine_id": int(engine_id),

        "observed_cycles": int(
            len(engine)
        ),

        "latest_cycle": latest["cycle"],

        "window_start": int(
            engine["cycle"].iloc[-WINDOW]
        ),

        "window_end": latest["cycle"],

        "predicted_rul": latest["rul"],

        "failure_probability":
            latest["failure_probability"],

        "status": latest["status"],

        "recommendation":
            get_recommendation(
                latest["status"]
            ),

        "trajectory": trajectory
    }


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route("/api/engines")
def api_engines():

    try:

        engines = sorted(
            DATA["unit_id"]
            .unique()
            .astype(int)
            .tolist()
        )

        return jsonify({
            "ok": True,
            "engines": engines
        })

    except Exception as exc:

        return jsonify({
            "ok": False,
            "error": str(exc)
        }), 500


@app.route("/api/engine/<int:engine_id>")
def api_engine(engine_id):

    try:

        result = analyze_engine(
            engine_id
        )

        return jsonify({
            "ok": True,
            "data": result
        })

    except Exception as exc:

        return jsonify({
            "ok": False,
            "error": str(exc)
        }), 500


@app.errorhandler(404)
def not_found(error):

    return jsonify({
        "ok": False,
        "error": "API route not found."
    }), 404


@app.errorhandler(500)
def server_error(error):

    return jsonify({
        "ok": False,
        "error": "Internal server error."
    }), 500


if __name__ == "__main__":

    print()
    print("=" * 64)
    print("NASA C-MAPSS PREDICTIVE MAINTENANCE")
    print("FLASK DASHBOARD")
    print("=" * 64)
    print(f"Device: {DEVICE}")

    if DATA is not None:
        print(
            f"Test rows: {len(DATA):,}"
        )
        print(
            f"Engines: {DATA['unit_id'].nunique()}"
        )

    print("=" * 64)
    print("Open: http://127.0.0.1:8000")
    print("=" * 64)
    print()

    app.run(
        host="127.0.0.1",
        port=8000,
        debug=False
    )
