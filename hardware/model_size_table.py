"""Model size (memory footprint) of the trained CNN checkpoints, and the extra memory
the early exits cost.

For each dataset x backbone found under --models (checkpoints from train_and_save_best.py):
    seen_KB    = ALL parameters of the *_SEEN_* checkpoint
    default_KB = backbone-only parameters of the *_Baseline_* checkpoint
                 (early_exits.* / late_input.* excluded — the plain model)
    overhead   = (seen_KB - default_KB) / default_KB
All counts are params x 4 bytes (fp32) / 1024.

Usage:  python model_size_table.py [--models ../seen_cnn/ckpts] [--out model_size_table.csv]
"""
import torch, glob, os, csv, argparse


def kb(sd, keep=lambda k: True):
    return sum(v.numel() for k, v in sd.items() if hasattr(v, "numel") and keep(k)) * 4 / 1024.0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(here, "..", "seen_cnn", "ckpts"))
    ap.add_argument("--out", default="model_size_table.csv")
    a = ap.parse_args()

    rows = {}
    for f in sorted(glob.glob(os.path.join(a.models, "**", "*.ckpt"), recursive=True)):
        b = os.path.basename(f)
        parts = b.split("_")
        ds, backbone, variant = parts[0], parts[1], parts[2]
        which = "candidate" if os.path.basename(os.path.dirname(f)) == "candidates" else "current"
        sd = torch.load(f, map_location="cpu")
        rec = rows.setdefault((ds, backbone, which), {})
        if variant == "SEEN":
            rec["seen_KB"] = kb(sd)
            rec["ckpt"] = b
        elif variant == "Baseline":
            rec["default_KB"] = kb(sd, lambda k: not (k.startswith("early_exits") or k.startswith("late_input")))

    # a candidate has no Baseline ckpt of its own: its default is the current one
    for (ds, bb, which), rec in rows.items():
        if which == "candidate" and "default_KB" not in rec:
            cur = rows.get((ds, bb, "current"), {})
            if "default_KB" in cur:
                rec["default_KB"] = cur["default_KB"]

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "backbone", "which", "seen_KB", "default_KB", "overhead_pct", "seen_ckpt"])
        for (ds, bb, which), r in sorted(rows.items()):
            seen, dflt = r.get("seen_KB"), r.get("default_KB")
            ov = round((seen - dflt) / dflt * 100, 1) if seen and dflt else ""
            w.writerow([ds, bb, which,
                        round(seen, 1) if seen else "", round(dflt, 1) if dflt else "", ov,
                        r.get("ckpt", "")])
            print(f"{ds:20s} {bb:7s} {which:9s}  SEEN {seen and round(seen, 1)} KB   "
                  f"default {dflt and round(dflt, 1)} KB   overhead {ov}%")
    print(f"[done] wrote {a.out}  ({len(rows)} models)")


if __name__ == "__main__":
    main()
