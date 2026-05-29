import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch
import tensorflow as tf
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from protopnet.knn_models import ProtoEEGkNN
from protopnet.spikenet_helpers import eeg_preprocess_for_plotting
from protopnet.spikenet_helpers import (
    label_finder,
    eeg_preprocess_for_model
)

def leave_one_channel_in(signal):
    eeg = signal[0:19]

    # average reference
    x1 = eeg - eeg.mean(axis=0)

    channels = ["Fp1","F3","C3","P3","F7","T3","T5","O1",
                "Fz","Cz","Pz","Fp2","F4","C4","P4",
                "F8","T4","T6","O2"]

    bipolar_channels = [
        "Fp1-F7","F7-T3","T3-T5","T5-O1",
        "Fp2-F8","F8-T4","T4-T6","T6-O2",
        "Fp1-F3","F3-C3","C3-P3","P3-O1",
        "Fp2-F4","F4-C4","C4-P4","P4-O2",
        "Fz-Cz","Cz-Pz"
    ]

    idx = np.array([
        [channels.index(bc.split("-")[0]), channels.index(bc.split("-")[1])]
        for bc in bipolar_channels
    ])

    x2 = eeg[idx[:, 0]] - eeg[idx[:, 1]]

    x = np.array([*x1, *x2])
    z1 = x.ravel()

    z2 = np.zeros((36, len(z1)))
    signal = np.array([z1, *z2])

    return signal

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
print("\n Loading Models...")

MODEL_PATH = "./models/trained_model_2.pth"
TRAIN_DICT_PATH = "../sn2_data/organized_data/train_dict.pth"

knn_engine = ProtoEEGkNN(MODEL_PATH)
train_dict = torch.load(TRAIN_DICT_PATH, map_location="cpu")

with open("protopnet/pretrained/model_fold_1_structure.txt", "r") as f:
    keras_model = tf.keras.models.model_from_json(f.read())
keras_model.load_weights("protopnet/pretrained/model_fold_1_weight.h5")

print(" Models Loaded Successfully!\n")


ANNOTATIONS = [
    {"label": "Fp1", "x": 0.3847, "y": 0.8491}, {"label": "Fp2", "x": 0.6135, "y": 0.8491},
    {"label": "F3", "x": 0.3454, "y": 0.6885}, {"label": "F4", "x": 0.6528, "y": 0.6885},
    {"label": "C3", "x": 0.3140, "y": 0.4971}, {"label": "C4", "x": 0.6842, "y": 0.4971},
    {"label": "P3", "x": 0.3454, "y": 0.3057}, {"label": "P4", "x": 0.6528, "y": 0.3057},
    {"label": "O1", "x": 0.3847, "y": 0.1450}, {"label": "O2", "x": 0.6135, "y": 0.1450},
    {"label": "F7", "x": 0.1996, "y": 0.7147}, {"label": "F8", "x": 0.7986, "y": 0.7147},
    {"label": "T3", "x": 0.1289, "y": 0.4971}, {"label": "T4", "x": 0.8693, "y": 0.4971},
    {"label": "T5", "x": 0.1996, "y": 0.2795}, {"label": "T6", "x": 0.7986, "y": 0.2795},
    {"label": "Fz", "x": 0.4991, "y": 0.6822}, {"label": "Cz", "x": 0.4991, "y": 0.4971},
    {"label": "Pz", "x": 0.4991, "y": 0.3120}
]

CONNECTIONS = [
    ("Fp1","F7"),("F7","T3"),("T3","T5"),("T5","O1"),
    ("Fp2","F8"),("F8","T4"),("T4","T6"),("T6","O2"),
    ("Fp1","F3"),("F3","C3"),("C3","P3"),("P3","O1"),
    ("Fp2","F4"),("F4","C4"),("C4","P4"),("P4","O2"),
    ("Fz","Cz"),("Cz","Pz")
]

class EEGRequest(BaseModel):
    signal: list  # [20 x 128] or [20 x 192]


@app.post("/explain")
async def explain_eeg(req: EEGRequest):
    try:
        print("\n================ NEW REQUEST ================")

        eeg_raw = np.array(req.signal)
        print(f"[INPUT] Shape: {eeg_raw.shape}")

        # Step 1: Crop
        if eeg_raw.shape[1] == 192:
            print("[STEP 1] Cropping 192 → 128")
            eeg_for_weights = eeg_raw[:, 32:160]
        else:
            print("[STEP 1] Already 128")
            eeg_for_weights = eeg_raw

        print(f"[STEP 1] Shape: {eeg_for_weights.shape}")

        # Step 2: Preprocess
        print("[STEP 2] leave_one_channel_in...")
        signal_ready = leave_one_channel_in(eeg_for_weights)
        print(f"[STEP 2] Shape: {signal_ready.shape}")

        # Step 3: Tensor reshaping
        eeg_tensor = torch.from_numpy(signal_ready).unsqueeze(0)
        print(f"[STEP 3] Unsqueeze: {eeg_tensor.shape}")

        batched_input = eeg_tensor.unfold(-1, 128, 128)
        print(f"[STEP 3] Unfold: {batched_input.shape}")

        batched_input = batched_input.squeeze(0).transpose(0, 1)
        batched_input = batched_input.transpose(-1, -2).unsqueeze(2)

        print(f"[STEP 3] Final Keras Input: {batched_input.shape}")

        # Step 4: Keras importance
        print("[STEP 4] Keras prediction...")
        importance_scores = keras_model.predict(batched_input.numpy(), verbose=0)[:, 0]
        print(f"[STEP 4] Scores shape: {importance_scores.shape}")
        print(f"[DEBUG] Raw Weight Stats | Max: {importance_scores.max():.4f} | Min: {importance_scores.min():.4f} | Mean: {importance_scores.mean():.4f}")
        print(f"[DEBUG] Top 3 Channel Indices: {importance_scores.argsort()[-3:][::-1]}")
        input_weight = importance_scores / (np.sum(importance_scores) + 0.000001)
        def min_max_normalize_internal(arr):
            min_val = np.min(arr)
            max_val = np.max(arr)
            if max_val == min_val:
                return np.zeros_like(arr)
            return (arr - min_val) / (max_val - min_val)

        # This ensures the Topoplot and Bars actually show contrast
        visual_weights = min_max_normalize_internal(input_weight)
        # Step 5: Inject into kNN
        temp_id = "live_request"
        knn_engine.model.prototype_layer.spikenet_weight_dict[temp_id] = torch.from_numpy(importance_scores)

        eeg_tensor_for_model = eeg_preprocess_for_model(eeg_raw).cpu()
        print(f"[STEP 5] kNN input: {eeg_tensor_for_model.shape}")

        output_dict = knn_engine.forward(eeg_tensor_for_model, [temp_id])

        print(f"[STEP 5] Prediction: {output_dict['prediction']}")
        print(f"[STEP 5] Matches: {output_dict['matches'][0]}")

        prediction_pct = float(output_dict["prediction"][0]) * 100

        # Step 6: Normalize weights
        norm_weights = importance_scores / (np.sum(importance_scores) + 1e-6)

        # Step 7: Neighbors
        neighbors = []
        for fname in output_dict["matches"][0]:
            neighbor_raw = train_dict[fname]
            processed = eeg_preprocess_for_plotting(neighbor_raw).numpy().tolist()
            neighbors.append({"id": fname, "signal": processed})

        print("=============== DONE ===============\n")

        return {
            "prediction": prediction_pct,
            "weights": visual_weights.tolist(),
            "nodes": ANNOTATIONS,
            "links": CONNECTIONS,
            "neighbors": neighbors
        }

    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)