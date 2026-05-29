# """
# Run this from INSIDE the ProtoEEG repo directory:
#     cd E:\\path\\to\\ProtoEEG
#     python explore_trained_model.py trained_model.pth
# """

# import sys
# import torch

# def explore_model(filepath):
#     print(f"\n{'='*60}")
#     print(f"  Exploring: {filepath}")
#     print(f"{'='*60}\n")

#     print("Loading model (requires protopnet module to be in path)...")
#     model = torch.load(filepath, map_location='cpu', weights_only=False)

#     print(f"[MODEL TYPE]: {type(model)}\n")

#     # ── Basic architecture ──────────────────────────────────────
#     print("── ARCHITECTURE ──────────────────────────────────────")
#     print(f"  Backbone    : {type(model.backbone).__name__}")
#     print(f"  Add-on layers: {type(model.add_on_layers).__name__}")
#     print(f"  Proto layer : {type(model.prototype_layer).__name__}")
#     print(f"  Pred head   : {type(model.prototype_prediction_head).__name__}")

#     # ── Prototype info ──────────────────────────────────────────
#     print("\n── PROTOTYPE LAYER ───────────────────────────────────")
#     pl = model.prototype_layer
#     print(f"  Num prototypes        : {pl.num_prototypes}")
#     print(f"  Num classes           : {pl.num_classes}")
#     print(f"  Prototypes per class  : {pl.num_prototypes_per_class}")
#     print(f"  Prototype tensor shape: {tuple(pl.prototype_tensors.shape)}")
#     print(f"  Latent channels       : {pl.latent_channels}")
#     print(f"  Activation function   : {type(pl.activation_function).__name__}")

#     pt = pl.prototype_tensors
#     print(f"\n  Prototype tensor stats:")
#     print(f"    Min : {pt.min().item():.6f}")
#     print(f"    Max : {pt.max().item():.6f}")
#     print(f"    Mean: {pt.float().mean().item():.6f}")
#     print(f"    Std : {pt.float().std().item():.6f}")

#     # ── Prototype info dict (which training samples became protos) ──
#     print("\n── PROTOTYPE INFO DICT ───────────────────────────────")
#     pid = pl.prototype_info_dict
#     print(f"  Number of assigned prototypes: {len(pid)}")
#     if len(pid) > 0:
#         print(f"  Sample entries:")
#         for k in list(pid.keys())[:5]:
#             print(f"    Proto #{k}: sample_id='{pid[k].sample_id}'")

#     # ── Class identity matrix ───────────────────────────────────
#     print("\n── CLASS IDENTITY ────────────────────────────────────")
#     ci = pl.prototype_class_identity
#     print(f"  Shape: {tuple(ci.shape)}  (num_prototypes x num_classes)")
#     protos_per_class = ci.sum(dim=0)
#     for c in range(pl.num_classes):
#         print(f"  Class {c}: {int(protos_per_class[c].item())} prototypes")

#     # ── Prediction head ─────────────────────────────────────────
#     print("\n── PREDICTION HEAD ───────────────────────────────────")
#     ph = model.prototype_prediction_head
#     ccl = ph.class_connection_layer
#     print(f"  Layer type   : {type(ccl).__name__}")
#     print(f"  Weight shape : {tuple(ccl.weight.shape)}")
#     print(f"  Has bias     : {ccl.bias is not None}")
#     if ccl.bias is not None:
#         print(f"  Bias value   : {ccl.bias.item():.6f}")
#     print(f"  Weight range : [{ccl.weight.min().item():.4f}, {ccl.weight.max().item():.4f}]")

#     # ── Importance by statistic (if present) ────────────────────
#     if hasattr(pl, 'importance_by_statistic'):
#         print("\n── IMPORTANCE BY STATISTIC ───────────────────────────")
#         sm = torch.nn.Softmax(dim=0)
#         weights = sm(pl.importance_by_statistic)
#         labels = ['Latent (cosine)', 'Range', 'Variance', 'FFT']
#         for label, w in zip(labels, weights):
#             print(f"  {label:20s}: {w.item():.4f} ({w.item()*100:.1f}%)")

#     # ── Backbone summary ─────────────────────────────────────────
#     print("\n── BACKBONE PARAMETERS ───────────────────────────────")
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"  Total parameters    : {total_params:,}")
#     print(f"  Trainable parameters: {trainable_params:,}")

#     # ── All named modules ────────────────────────────────────────
#     print("\n── TOP-LEVEL MODULES ─────────────────────────────────")
#     for name, module in model.named_children():
#         params = sum(p.numel() for p in module.parameters())
#         print(f"  {name:30s} | {type(module).__name__:40s} | {params:>10,} params")

#     print(f"\n{'='*60}\n")


# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python explore_trained_model.py <path_to_trained_model.pth>")
#         print("\nIMPORTANT: Run this from inside the ProtoEEG repo directory!")
#         print("  cd E:\\path\\to\\ProtoEEG")
#         print("  python explore_trained_model.py trained_model.pth")
#         sys.exit(1)
#     explore_model(sys.argv[1])

"""
Rich exploration script for ProtoEEG trained_model.pth

Run from INSIDE the ProtoEEG repo directory:
    cd E:\\path\\to\\ProtoEEG
    python explore_trained_model_rich.py trained_model.pth

Optionally save full report to a text file:
    python explore_trained_model_rich.py trained_model.pth --save report.txt
"""

import sys
import argparse
from collections import defaultdict, Counter
import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def section(title, width=70):
    print(f"\n{'─'*width}")
    print(f"  {title}")
    print(f"{'─'*width}")

def tensor_stats(t, name=""):
    t = t.float()
    prefix = f"  {name}: " if name else "  "
    print(f"{prefix}shape={tuple(t.shape)}")
    print(f"{'':>{len(prefix)}}min={t.min().item():.6f}  max={t.max().item():.6f}")
    print(f"{'':>{len(prefix)}}mean={t.mean().item():.6f}  std={t.std().item():.6f}")
    nans = torch.isnan(t).sum().item()
    infs = torch.isinf(t).sum().item()
    if nans > 0 or infs > 0:
        print(f"{'':>{len(prefix)}}⚠ NaNs={nans}  Infs={infs}")


# ─────────────────────────────────────────────
# 1. TOP LEVEL
# ─────────────────────────────────────────────

def section_toplevel(model):
    section("1. TOP-LEVEL MODEL INFO")
    print(f"  Class      : {type(model).__module__}.{type(model).__name__}")
    print(f"  k_for_topk : {getattr(model, 'k_for_topk', 'N/A')}")

    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen  = total - trainable
    print(f"\n  Total parameters    : {total:>12,}")
    print(f"  Trainable           : {trainable:>12,}  ({100*trainable/total:.1f}%)")
    print(f"  Frozen              : {frozen:>12,}  ({100*frozen/total:.1f}%)")

    print(f"\n  {'Component':<40} {'Type':<45} {'Params':>10}")
    print(f"  {'─'*40} {'─'*45} {'─'*10}")
    for name, module in model.named_children():
        p = sum(x.numel() for x in module.parameters())
        print(f"  {name:<40} {type(module).__name__:<45} {p:>10,}")


# ─────────────────────────────────────────────
# 2. BACKBONE
# ─────────────────────────────────────────────

def section_backbone(model):
    section("2. BACKBONE DETAILS")
    bb = model.backbone
    print(f"  Type            : {type(bb).__name__}")
    print(f"  Input channels  : {getattr(bb, 'input_channels', 'N/A')}")
    print(f"  Latent dimension: {getattr(bb, 'latent_dimension', 'N/A')}")

    print(f"\n  Layer breakdown:")
    print(f"  {'Layer name':<35} {'Type':<30} {'Params':>10} {'Trainable':>10}")
    print(f"  {'─'*35} {'─'*30} {'─'*10} {'─'*10}")
    for name, mod in bb.named_modules():
        if len(list(mod.children())) == 0:  # leaf modules only
            p = sum(x.numel() for x in mod.parameters())
            tr = sum(x.numel() for x in mod.parameters() if x.requires_grad)
            if p > 0:
                print(f"  {name:<35} {type(mod).__name__:<30} {p:>10,} {tr:>10,}")


# ─────────────────────────────────────────────
# 3. ADD-ON LAYERS
# ─────────────────────────────────────────────

def section_addon(model):
    section("3. ADD-ON LAYERS")
    ao = model.add_on_layers
    print(f"  Type             : {type(ao).__name__}")
    p = sum(x.numel() for x in ao.parameters())
    print(f"  Total parameters : {p:,}")
    if p == 0:
        print("  → Identity passthrough (no compression applied)")
    else:
        print(f"  Input channels   : {getattr(ao, 'input_channels', 'N/A')}")
        print(f"  Proto channels   : {getattr(ao, 'proto_channels', 'N/A')}")
        for name, mod in ao.named_modules():
            if len(list(mod.children())) == 0:
                mp = sum(x.numel() for x in mod.parameters())
                if mp > 0:
                    print(f"    {name:<30} {type(mod).__name__:<25} {mp:>8,} params")


# ─────────────────────────────────────────────
# 4. PROTOTYPE LAYER
# ─────────────────────────────────────────────

def section_prototype_layer(model):
    section("4. PROTOTYPE LAYER")
    pl = model.prototype_layer

    print(f"  Type                  : {type(pl).__name__}")
    print(f"  Num prototypes        : {pl.num_prototypes}")
    print(f"  Num classes           : {pl.num_classes}")
    print(f"  Prototypes per class  : {pl.num_prototypes_per_class}")
    print(f"  Latent channels       : {pl.latent_channels}")
    print(f"  Activation function   : {type(pl.activation_function).__name__}")
    print(f"  k_for_topk            : {pl.k_for_topk}")

    pt = pl.prototype_tensors
    print(f"\n  Prototype tensor shape: {tuple(pt.shape)}")
    print(f"  Interpretation        : ({pt.shape[0]} prototypes) x "
          f"({pt.shape[1]} feature dims) x "
          f"({pt.shape[2]} height) x "
          f"({pt.shape[3]} EEG channels)")

    # Feature dimension breakdown
    print(f"\n  Feature dim breakdown (258 total):")
    print(f"    [0:128]   = 128 latent backbone features")
    print(f"    [128]     = 1   amplitude range statistic")
    print(f"    [129]     = 1   variance statistic")
    print(f"    [130:258] = 128 FFT frequency features")

    # Stats per feature group
    print(f"\n  Stats per feature group:")
    tensor_stats(pt[:, :128],   "Latent (0:128)")
    tensor_stats(pt[:, 128:129],"Range  (128)")
    tensor_stats(pt[:, 129:130],"Var    (129)")
    tensor_stats(pt[:, 130:],   "FFT    (130:258)")

    # Per-class prototype stats
    print(f"\n  Per-class prototype tensor norms (mean L2):")
    ci = pl.prototype_class_identity
    print(f"  {'Class':<8} {'#Protos':<10} {'Mean L2 norm':<15} {'Std L2 norm'}")
    print(f"  {'─'*8} {'─'*10} {'─'*15} {'─'*12}")
    for c in range(pl.num_classes):
        proto_indices = (ci[:, c] == 1).nonzero(as_tuple=True)[0]
        class_protos = pt[proto_indices].float()
        norms = class_protos.view(class_protos.shape[0], -1).norm(dim=1)
        print(f"  {c:<8} {len(proto_indices):<10} {norms.mean().item():<15.4f} {norms.std().item():.4f}")


# ─────────────────────────────────────────────
# 5. PROTOTYPE INFO DICT — full breakdown
# ─────────────────────────────────────────────

def section_prototype_info(model):
    section("5. PROTOTYPE INFO DICT — ALL 405 PROTOTYPES")
    pl  = model.prototype_layer
    pid = pl.prototype_info_dict
    ci  = pl.prototype_class_identity

    print(f"  Total assigned prototypes : {len(pid)}")

    # Unique files
    all_files = [v.sample_id for v in pid.values()]
    unique_files = set(all_files)
    file_counts = Counter(all_files)
    print(f"  Unique source EEG files   : {len(unique_files)}")
    print(f"  Most reused files:")
    for fname, cnt in file_counts.most_common(10):
        print(f"    {cnt:>3}x  {fname}")

    # Per-class source files
    print(f"\n  Per-class unique source files:")
    print(f"  {'Class':<8} {'#Unique files':<16} {'Sample source files (first 3)'}")
    print(f"  {'─'*8} {'─'*16} {'─'*50}")
    class_files = defaultdict(list)
    for k, v in pid.items():
        c = ci[k].argmax().item()
        class_files[c].append(v.sample_id)
    for c in sorted(class_files.keys()):
        files = class_files[c]
        unique = set(files)
        samples = list(unique)[:3]
        print(f"  {c:<8} {len(unique):<16} {', '.join(samples)}")

    # Subject breakdown
    print(f"\n  Subject breakdown (from filename prefix):")
    subjects = Counter(f.split('_')[0] for f in all_files)
    print(f"  {'Subject':<20} {'#Prototype slots':<20} {'%'}")
    print(f"  {'─'*20} {'─'*20} {'─'*6}")
    for subj, cnt in subjects.most_common():
        print(f"  {subj:<20} {cnt:<20} {100*cnt/len(all_files):.1f}%")

    # Full table
    print(f"\n  Full prototype table:")
    print(f"  {'Proto#':<9} {'Class':<8} {'Source EEG file'}")
    print(f"  {'─'*9} {'─'*8} {'─'*40}")
    for k in sorted(pid.keys()):
        c = ci[k].argmax().item()
        print(f"  {k:<9} {c:<8} {pid[k].sample_id}")


# ─────────────────────────────────────────────
# 6. IMPORTANCE BY STATISTIC
# ─────────────────────────────────────────────

def section_importance(model):
    section("6. LEARNED SIMILARITY WEIGHTS")
    pl = model.prototype_layer

    if not hasattr(pl, 'importance_by_statistic'):
        print("  importance_by_statistic not found in this model.")
        return

    raw = pl.importance_by_statistic
    sm  = torch.nn.Softmax(dim=0)
    weights = sm(raw)
    labels  = ['Latent/cosine (waveform shape)', 'Range', 'Variance', 'FFT (frequency)']

    print(f"  {'Statistic':<35} {'Raw logit':>12} {'Softmax weight':>16} {'Bar'}")
    print(f"  {'─'*35} {'─'*12} {'─'*16} {'─'*20}")
    for label, r, w in zip(labels, raw, weights):
        bar = '█' * int(w.item() * 40)
        print(f"  {label:<35} {r.item():>12.4f} {w.item():>14.4f}   {bar}")

    print(f"\n  Interpretation:")
    print(f"    Waveform shape accounts for {weights[0].item()*100:.1f}% of similarity score")
    print(f"    Frequency content accounts for {weights[3].item()*100:.1f}% of similarity score")
    print(f"    Amplitude stats (range+var) account for {(weights[1]+weights[2]).item()*100:.1f}%")


# ─────────────────────────────────────────────
# 7. PREDICTION HEAD
# ─────────────────────────────────────────────

def section_prediction_head(model):
    section("7. PREDICTION HEAD DETAILS")
    ph  = model.prototype_prediction_head
    ccl = ph.class_connection_layer
    pl  = model.prototype_layer
    ci  = pl.prototype_class_identity

    print(f"  Type         : {type(ph).__name__}")
    print(f"  Layer        : {type(ccl).__name__}(in={ccl.in_features}, out={ccl.out_features})")
    print(f"  Bias         : {ccl.bias.item():.6f}")
    print(f"  Weight shape : {tuple(ccl.weight.shape)}")
    print(f"  Weight range : [{ccl.weight.min().item():.6f}, {ccl.weight.max().item():.6f}]")
    print(f"  Weight mean  : {ccl.weight.mean().item():.6f}")
    print(f"  Weight std   : {ccl.weight.std().item():.6f}")

    # Per-class weight analysis
    print(f"\n  Per-class connection weights (how much each class contributes to output):")
    print(f"  {'Class':<8} {'Label meaning':<30} {'Mean weight':>13} {'Min':>10} {'Max':>10}")
    print(f"  {'─'*8} {'─'*30} {'─'*13} {'─'*10} {'─'*10}")
    class_meanings = {
        0: "Negative (no spike)",
        1: "Positive (spike)",
    }
    w = ccl.weight.squeeze(0)  # shape [405]
    for c in range(pl.num_classes):
        idx = (ci[:, c] == 1).nonzero(as_tuple=True)[0]
        cw  = w[idx]
        meaning = class_meanings.get(c, "")
        print(f"  {c:<8} {meaning:<30} {cw.mean().item():>13.6f} "
              f"{cw.min().item():>10.6f} {cw.max().item():>10.6f}")

    print(f"\n  Prediction formula:")
    print(f"    sigmoid( dot(prototype_similarities, weights) + {ccl.bias.item():.2f} )")
    print(f"    → output is probability in [0, 1], thresholded at 0.5 for binary class")


# ─────────────────────────────────────────────
# 8. CLASS IDENTITY MATRIX
# ─────────────────────────────────────────────

def section_class_identity(model):
    section("8. CLASS IDENTITY MATRIX")
    pl = model.prototype_layer
    ci = pl.prototype_class_identity.float()

    print(f"  Shape: {tuple(ci.shape)}  ({ci.shape[0]} prototypes × {ci.shape[1]} classes)")
    print(f"  Each row = one prototype, each col = one class")
    print(f"  Value 1 = prototype belongs to that class, 0 = does not")

    print(f"\n  Verification — prototypes per class:")
    for c in range(pl.num_classes):
        count = int(ci[:, c].sum().item())
        bar = '█' * (count // 3)
        print(f"    Class {c}: {count:>3} prototypes  {bar}")

    print(f"\n  Verification — classes per prototype (should all be 1):")
    classes_per_proto = ci.sum(dim=1)
    unique_vals = classes_per_proto.unique()
    print(f"    Unique values: {unique_vals.tolist()}")
    if (unique_vals == torch.tensor([1.0])).all():
        print(f"    ✓ Every prototype belongs to exactly 1 class")


# ─────────────────────────────────────────────
# 9. SPIKENET WEIGHT DICT (channel weights)
# ─────────────────────────────────────────────

def section_spikenet_weights(model):
    section("9. SPIKENET CHANNEL WEIGHT DICT")
    pl = model.prototype_layer

    if not hasattr(pl, 'spikenet_weight_dict'):
        print("  spikenet_weight_dict not found.")
        return

    swd = pl.spikenet_weight_dict
    print(f"  Number of EEG clips with channel weights : {len(swd)}")

    keys = list(swd.keys())
    sample_key = keys[0]
    sample_val = swd[sample_key]
    print(f"  Weight vector shape per clip             : {tuple(sample_val.shape)}")
    print(f"  (37 values = one importance weight per EEG channel derivation)")

    # Stats across all weights
    all_weights = torch.stack(list(swd.values())).float()
    print(f"\n  Across all {len(swd)} clips:")
    print(f"    Global min  : {all_weights.min().item():.6f}")
    print(f"    Global max  : {all_weights.max().item():.6f}")
    print(f"    Global mean : {all_weights.mean().item():.6f}")
    print(f"    Global std  : {all_weights.std().item():.6f}")

    # Per-channel average importance
    channel_names = [
        "Fp1-Avg","F3-Avg","C3-Avg","P3-Avg","F7-Avg","T3-Avg","T5-Avg","O1-Avg",
        "Fz-Avg","Cz-Avg","Pz-Avg","Fp2-Avg","F4-Avg","C4-Avg","P4-Avg","F8-Avg",
        "T4-Avg","T6-Avg","O2-Avg",
        "Fp1-F7","F7-T3","T3-T5","T5-O1","Fp2-F8","F8-T4","T4-T6","T6-O2",
        "Fp1-F3","F3-C3","C3-P3","P3-O1","Fp2-F4","F4-C4","C4-P4","P4-O2",
        "Fz-Cz","Cz-Pz"
    ]
    per_channel_mean = all_weights.mean(dim=0)
    sorted_idx = per_channel_mean.argsort(descending=True)

    print(f"\n  Top 10 most important channels (averaged across all training clips):")
    print(f"  {'Rank':<6} {'Channel':<12} {'Avg importance':>16} {'Bar'}")
    print(f"  {'─'*6} {'─'*12} {'─'*16} {'─'*20}")
    for rank, idx in enumerate(sorted_idx[:10]):
        val = per_channel_mean[idx].item()
        bar = '█' * int(val * 500)
        name = channel_names[idx] if idx < len(channel_names) else f"ch{idx}"
        print(f"  {rank+1:<6} {name:<12} {val:>16.6f}   {bar}")

    print(f"\n  Bottom 5 least important channels:")
    for rank, idx in enumerate(sorted_idx[-5:]):
        val = per_channel_mean[idx].item()
        name = channel_names[idx] if idx < len(channel_names) else f"ch{idx}"
        print(f"    {name:<12} {val:.6f}")

    # Sample a few clips
    print(f"\n  Sample clip channel weights:")
    for key in keys[:3]:
        w = swd[key]
        top_ch = w.argmax().item()
        name = channel_names[top_ch] if top_ch < len(channel_names) else f"ch{top_ch}"
        print(f"    {key:<35} sum={w.sum().item():.4f}  "
              f"top_channel={name} ({w.max().item():.4f})")


# ─────────────────────────────────────────────
# 10. FULL NAMED PARAMETER TABLE
# ─────────────────────────────────────────────

def section_parameters(model):
    section("10. FULL NAMED PARAMETER TABLE")
    print(f"  {'Parameter name':<65} {'Shape':<25} {'#Params':>10} {'Requires grad':>14}")
    print(f"  {'─'*65} {'─'*25} {'─'*10} {'─'*14}")
    for name, param in model.named_parameters():
        shape_str = str(tuple(param.shape))
        print(f"  {name:<65} {shape_str:<25} {param.numel():>10,} {str(param.requires_grad):>14}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Path to trained_model.pth")
    parser.add_argument("--save", help="Optional: save output to this text file", default=None)
    parser.add_argument("--sections", help="Comma-separated section numbers to run (e.g. 1,2,5). Default: all", default=None)
    args = parser.parse_args()

    # Optionally redirect stdout to file
    if args.save:
        import io
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer

    print(f"\n{'='*70}")
    print(f"  ProtoEEG Model — Rich Exploration Report")
    print(f"  File: {args.filepath}")
    print(f"{'='*70}")

    print("\nLoading model...")
    model = torch.load(args.filepath, map_location='cpu', weights_only=False)
    print("Model loaded successfully.\n")

    requested = set(args.sections.split(',')) if args.sections else None

    def should_run(n):
        return requested is None or str(n) in requested

    if should_run(1):  section_toplevel(model)
    if should_run(2):  section_backbone(model)
    if should_run(3):  section_addon(model)
    if should_run(4):  section_prototype_layer(model)
    if should_run(5):  section_prototype_info(model)
    if should_run(6):  section_importance(model)
    if should_run(7):  section_prediction_head(model)
    if should_run(8):  section_class_identity(model)
    if should_run(9):  section_spikenet_weights(model)
    if should_run(10): section_parameters(model)

    print(f"\n{'='*70}")
    print(f"  End of report")
    print(f"{'='*70}\n")

    if args.save:
        sys.stdout = old_stdout
        with open(args.save, 'w', encoding='utf-8') as f:
            f.write(buffer.getvalue())
        print(buffer.getvalue())
        print(f"\n✓ Report saved to: {args.save}")


if __name__ == "__main__":
    main()
