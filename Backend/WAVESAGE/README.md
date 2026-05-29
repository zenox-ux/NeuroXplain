# WAVESAGE

The main goal of this project is to solve a big problem in neurology: **Micro-Event Localization**.

Usually, AI can tell you if an EEG signal is `"Abnormal"`, but it can't point to the exact 100ms where the spike or sharp wave actually happened.

WAVESAGE uses **Wavelets** and **SHAP** to highlight these exact clinical patterns so doctors can trust the model's decision.

---

# Why WAVESAGE?

Most XAI methods (like **LIME** or **Grad-CAM**) treat EEG data like a simple list of numbers.

But EEG is a **time-series signal** where samples are highly correlated.

WAVESAGE fixes this by:

- Decomposing the signal using Wavelets (DWT)
- Using SHAP to see which frequency bands are actually contributing to the `"Abnormal"` label
- Using Majority Voting across different wavelet levels to pinpoint the micro-event

---

# Folder Structure

I've organized the code into two main parts:

## `WAVSAGE/`

This is the main framework.

It contains the logic for:
- wavelet decomposition (`db4` mother wavelet)
- SHAP attribution
- signal reconstruction
- micro-event localization

---

## `ExplainableAI/`

Since we benchmarked WAVESAGE against 11 other methods, I’ve included the code for all of them here.

You’ll find folders for:

- Grad-CAM
- Integrated Gradients
- Occlusion-Based
- LIME
- SmoothGrad
- Deep Taylor Decomposition (DTD)
- RISE

Each folder has:

| File | Purpose |
|------|---------|
| `single_window.py` | Visualize one EEG sample |
| `batch_processing.py` | Compute average metrics like IoU and F1-score across the test set |

---

# How to Get Started

## 1. Requirements

First, make sure you have the basics installed.

We used **TensorFlow 2.x** for the 1D-CNN classifier.

```bash
pip install tensorflow numpy matplotlib scikit-learn joblib pandas PyWavelets shap
```

---

# 2. Running the Model

The workflow is basically two steps:

---

## Step A: Classification

The model first takes a 2-second EEG window and decides if it’s:

- `Normal`
- `Abnormal`

---

## Step B: Localization (The XAI Bit)

Once a window is labeled `"Abnormal"`, you run the explainer.

---

# To Visualize a Single Case

Navigate to the method you want to test and run:

```bash
python ExplainableAI/GradCAM/single_window.py
```

This will pop up a plot showing:

- the raw EEG
- the `"Importance Map"` (the parts the AI is focusing on)

---

# To Get Full Results (Metrics)

If you want to see:

- IoU (Intersection over Union)
- F1-score

as reported in the paper, run:

```bash
python WAVSAGE/batch_processing.py
```

This will process abnormal windows and save the results in a CSV file.