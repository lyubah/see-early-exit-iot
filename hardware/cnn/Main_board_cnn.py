"""
Main_board_cnn.py  --  on-device accuracy / latency / power run for the CNN1D and AlexNet
early-exit models (same measurement rig and CSV schema as the SEE-RF board scripts),
covering THREE variants x TWO backbones:
    backbone in {CNN1D, AlexNet}  x  variant in {Baseline, EE, SEEN}
      Baseline : plain backbone, NO early exits (regular CNN1D / AlexNet)
      EE       : full-window early exits (BranchyNet-style)   -> CNN1D_extended_EENN / AlexNetEENN
      SEEN     : sensor-aware early exits (partial window)     -> *_partialsampling / AlexNetPartial

The Baseline runs the SEE model's backbone (conv/layer stack -> final classifier) with the
exit branches skipped, i.e. the "no early exit" model. Timing uses the same board timers
as the other variants.

Writes <dataset>_accuracy_results_<backbone>_<variant>.csv with one row per test window:
    t_start, t1, t2, t3, t4, total, true_label, prediction, correctness, exit_taken, data%
Energy is added downstream by process_power_energy.py from the INA219 power log.

--with_power launches data_logger.py (Raspberry Pi + INA219).  --model_ckpt loads a trained
model (checkpoint names from train_and_save_best.py carry the exit config); omit it to
smoke-test the timing path with an untrained model.
"""
import os, sys, csv, time, argparse, pickle
import numpy as np

_CNN_DIR = os.path.dirname(os.path.abspath(__file__))
# the timed model classes build on the network definitions in seen_cnn/
_REPO_ROOT = os.path.abspath(os.path.join(_CNN_DIR, "..", "..", "seen_cnn"))
for _p in (_CNN_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import torch


def load_dataset(dataset_name):
    roots = ["", _REPO_ROOT]
    for root in roots:
        native = os.path.join(root, dataset_name + ".pkl") if root else dataset_name + ".pkl"
        if os.path.exists(native):
            d = pickle.load(open(native, "rb"))
            return (np.asarray(d["Data"]), np.asarray(d["labels_array"]), int(d["n_channels"]),
                    int(d["num_activities"]), int(d["n_window"]), int(d["n_data"]))
    for root in roots:
        for rel in (f"Datasets/{dataset_name}_dataLabels.pkl", f"{dataset_name}_dataLabels.pkl"):
            p = os.path.join(root, rel) if root else rel
            if os.path.exists(p):
                dl = pickle.load(open(p, "rb"))
                data = np.asarray(dl["data"]); labels = np.asarray(dl["labels"])
                n_window, n_channel, n_data = data.shape
                return (data, labels, int(n_channel), int(len(np.unique(labels))),
                        int(n_window), int(n_data))
    raise FileNotFoundError(f"No dataset pkl for {dataset_name} (cwd or {_REPO_ROOT})")


def build_model(backbone, variant, n_channel, num_activities, input_seq_len,
                thresholds, num_exits, n_data, exit_placement, new_data_perc):
    if backbone == "CNN1D":
        import EENN_functions_timers as mod
        if variant == "SEEN":   M = mod.CNN1D_extended_EENN_partialsampling
        else:                   M = mod.CNN1D_extended_EENN          # EE + Baseline share this backbone
    elif backbone == "AlexNet":
        import AlexNet_EENN_functions_timers as mod
        if variant == "SEEN":   M = mod.AlexNetPartial
        else:                   M = mod.AlexNetEENN
    else:
        raise ValueError(f"--backbone must be CNN1D or AlexNet, got {backbone}")
    if variant == "SEEN":
        model = M(n_channel, num_activities, input_seq_len, thresholds, num_exits, 0,
                  n_data, exit_placement, new_data_perc)
    else:                                                            # EE / Baseline = 8-arg
        model = M(n_channel, num_activities, input_seq_len, thresholds, num_exits, 0,
                  n_data, exit_placement)
    model.eval()
    return model


def _mk_result(output, is_conf):
    return {"output": np.asarray(output), "is_confident": np.asarray(is_conf)}


def baseline_timed_forward(model, backbone, x):
    """Time the plain backbone -> final classifier (NO early-exit branches)."""
    start = time.time()
    if backbone == "CNN1D":
        for conv in (model.conv1, model.conv2, model.conv3, model.conv4, model.conv5):
            x = model.maxpool(model.relu(conv(x)))
        x = x.view(x.size(0), -1)
        out = model.fc2(model.relu(model.fc1(x)))
    else:                                                            # AlexNet
        for layer in (model.layer1, model.layer2, model.layer3, model.layer4, model.layer5):
            x = layer(x)
        x = x.view(x.size(0), -1)
        out = model.classifier(x)
    dt = time.time() - start
    _, lab = torch.max(out.data, 1)
    return [_mk_result(lab.detach().cpu().numpy(), np.ones(1) * True)], [dt], [start]


def cnn1d_ee_timed_forward(model, x):
    """CNN1D_extended_EENN has no timing in-class -> mirror AlexNetEENN's timed full pass."""
    results, times, start_time_arr = [], [], []
    idx = 0
    ops = [(model.conv1, 1), (model.conv2, 2), (model.conv3, 3), (model.conv4, 4), (model.conv5, 5)]
    start = time.time(); start_time_arr.append(start)
    for conv, layer_num in ops:
        x = model.maxpool(model.relu(conv(x)))
        if layer_num in model.exit_placements and layer_num != 5:
            eo = model.early_exits[idx](x)
            _, lab = torch.max(eo["logits"].data, 1)
            times.append(time.time() - start)
            results.append(_mk_result(lab.detach().cpu().numpy(), eo["is_confident"].cpu().numpy()))
            start = time.time(); start_time_arr.append(start); idx += 1
        else:
            results.append(_mk_result(-1 * np.ones(1, dtype=int), -1 * np.ones(1, dtype=int)))
            times.append(0)
    results.pop(); times.pop()
    xf = x.view(x.size(0), -1)
    fo = model.fc2(model.relu(model.fc1(xf)))
    _, flab = torch.max(fo.data, 1)
    times.append(time.time() - start)
    results.append(_mk_result(flab.detach().cpu().numpy(), np.ones(1) * True))
    start_time_arr.append(time.time())
    return results, times, start_time_arr


def run_inference(model, backbone, variant, X_test, device):
    if variant == "SEEN":
        return model.inference(torch.tensor(X_test, dtype=torch.float32), device)
    results_all, times_all, starts_all = [], [], []
    with torch.no_grad():
        for w in range(len(X_test)):
            x = torch.tensor(np.expand_dims(X_test[w, :, :], axis=0), dtype=torch.float32, device=device)
            if variant == "Baseline":
                r, t, s = baseline_timed_forward(model, backbone, x)
            elif backbone == "AlexNet":
                r, t, s = model.forward_inference(x)          # AlexNetEENN is already timed
            else:
                r, t, s = cnn1d_ee_timed_forward(model, x)
            results_all.append(r); times_all.append(t); starts_all.append(s)
    return results_all, times_all, starts_all


def _is_confident(dic):
    v = dic.get("is_confident")
    if v is None:
        return False
    try:
        v = np.asarray(v).ravel()[0]
    except Exception:
        pass
    try:
        return float(v) == 1.0
    except Exception:
        return bool(v)


def _to_pred(output, num_activities):
    o = output
    if torch.is_tensor(o):
        o = o.detach().cpu().numpy()
    o = np.asarray(o).squeeze()
    if o.ndim >= 1 and o.size == num_activities:
        return int(np.argmax(o))
    return int(np.asarray(o).reshape(-1)[0])


def results_to_rows(results, times, start_time_arr, y_test, new_data_perc, num_activities, variant):
    MAX_T = 4
    rows = []
    for i in range(len(results)):
        per_exit = results[i]
        chosen, pred = None, None
        for idx, dic in enumerate(per_exit):
            if _is_confident(dic):
                chosen, pred = idx, _to_pred(dic["output"], num_activities); break
        if chosen is None:
            chosen = len(per_exit) - 1
            pred = _to_pred(per_exit[-1]["output"], num_activities)
        tvec = list(times[i]) if times[i] is not None else []
        sa = start_time_arr[i]
        t_start = float(np.asarray(sa).ravel()[0]) if (isinstance(sa, (list, tuple, np.ndarray)) and len(sa)) else float(sa or 0.0)
        cum, tcols = 0.0, [-1] * MAX_T
        for k in range(chosen + 1):
            if k < len(tvec):
                cum += float(tvec[k])
            if k < MAX_T:
                tcols[k] = t_start + cum
        total = cum
        if variant == "SEEN":
            # cumulative fraction sensed at the exit taken (the model records it per exit);
            # new_data_perc holds INCREMENTAL chunks and `chosen` is a layer index, so it
            # cannot be used directly.
            pp = per_exit[chosen].get("partial_percentage") if isinstance(per_exit[chosen], dict) else None
            if pp is None:
                data_pct = 100
            else:
                data_pct = float(np.asarray(pp).ravel()[0])
                if data_pct <= 1.0:
                    data_pct *= 100.0
        else:
            data_pct = 100
        yt = int(y_test[i])
        rows.append({"t_start": t_start, "t1": tcols[0], "t2": tcols[1], "t3": tcols[2],
                     "t4": tcols[3], "total": total, "true_label": yt, "prediction": pred,
                     "correctness": (yt == pred), "exit_taken": chosen + 1, "data%": data_pct})
    return rows


def config_from_ckpt_name(path):
    """Checkpoint filenames are self-describing:
    <ds>_<bb>_<var>_thresholds[..]_exit_placement[..]_new_data_perc[..].ckpt
    Returns (thresholds, exit_placement, new_data_perc) or None if the name doesn't match."""
    import re
    m = re.search(r"thresholds\[(.*?)\]_exit_placement\[(.*?)\]_new_data_perc\[(.*?)\]",
                  os.path.basename(path))
    if not m:
        return None
    return ([float(x) for x in m.group(1).split(",")],
            [int(x) for x in m.group(2).split(",")],
            [int(x) for x in m.group(3).split(",")])


def main():
    ap = argparse.ArgumentParser(description="Board rig for CNN1D/AlexNet x Baseline/EE/SEEN")
    ap.add_argument("--dataset_name", default="Epilepsy")
    ap.add_argument("--backbone", default="CNN1D", choices=["CNN1D", "AlexNet"])
    ap.add_argument("--variant", default="SEEN", choices=["Baseline", "EE", "SEEN"])
    ap.add_argument("--model_ckpt", default=None)
    ap.add_argument("--thresholds", nargs="+", type=float, default=None)
    ap.add_argument("--num_exits", type=int, default=None)
    ap.add_argument("--exit_placement", nargs="+", type=int, default=None)
    ap.add_argument("--new_data_perc", nargs="+", type=int, default=None)
    ap.add_argument("--input_seq_len", type=int, default=24)
    ap.add_argument("--max_windows", type=int, default=0)
    ap.add_argument("--with_power", action="store_true")
    ap.add_argument("--power_logger", default=os.path.join(_CNN_DIR, "..", "data_logger.py"))
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    # plug and play: unless the config flags are passed explicitly, read
    # thresholds / exit_placement / new_data_perc straight from the ckpt filename
    parsed = config_from_ckpt_name(args.model_ckpt) if args.model_ckpt else None
    if parsed:
        thr, exits, perc = parsed
        if args.thresholds is None: args.thresholds = thr
        if args.exit_placement is None: args.exit_placement = exits
        if args.new_data_perc is None: args.new_data_perc = perc
        print(f"[cfg] from ckpt name: exits={args.exit_placement} perc={args.new_data_perc} thr={args.thresholds}")
    if args.thresholds is None: args.thresholds = [0.9, 0.9]
    if args.exit_placement is None: args.exit_placement = [2, 4]
    if args.new_data_perc is None: args.new_data_perc = [30, 40]
    if args.num_exits is None: args.num_exits = len(args.exit_placement)

    device = torch.device("cpu")
    from sklearn.model_selection import train_test_split
    data, labels, n_channel, num_activities, n_window, n_data = load_dataset(args.dataset_name)
    print(f"[data] {args.dataset_name}: windows={n_window} ch={n_channel} n_data={n_data} classes={num_activities}")
    _, te, _, _ = train_test_split(list(range(n_window)), labels, test_size=0.40, random_state=0)
    X_test = data[te, :, :]; y_test = labels[te]
    if args.max_windows and args.max_windows < len(X_test):
        X_test = X_test[:args.max_windows]; y_test = y_test[:args.max_windows]
    print(f"[cfg] backbone={args.backbone} variant={args.variant} -> {len(X_test)} windows")

    model = build_model(args.backbone, args.variant, n_channel, num_activities, args.input_seq_len,
                        args.thresholds, args.num_exits, n_data, args.exit_placement, args.new_data_perc)
    if args.model_ckpt:
        model.load_state_dict(torch.load(args.model_ckpt, map_location="cpu"))
        print(f"[model] loaded {args.model_ckpt}")
    else:
        print("[model] WARNING: untrained (timing-only smoke test)")
    model.to(device); model.is_training = 0

    logger_proc = None
    if args.with_power:
        import subprocess
        try:
            logger_proc = subprocess.Popen([sys.executable, args.power_logger, args.dataset_name],
                                           cwd=os.getcwd())
            time.sleep(1.0); print(f"[power] logger pid {logger_proc.pid} -> {args.dataset_name}_shared_Power.csv "
                                   f"(rename to {args.dataset_name}_{args.backbone}_{args.variant}_power.csv "
                                   f"to keep runs apart for process_power_energy.py)")
        except Exception as e:
            print(f"[power] could not start logger ({e})")

    t0 = time.time()
    results, times, starts = run_inference(model, args.backbone, args.variant, X_test, device)
    print(f"[infer] {len(results)} windows in {time.time()-t0:.3f}s")

    if args.with_power:
        os.system("pkill -f data_logger.py")
        if logger_proc:
            try: logger_proc.wait(timeout=5)
            except Exception: pass

    rows = results_to_rows(results, times, starts, y_test, args.new_data_perc, num_activities, args.variant)
    header = ["t_start", "t1", "t2", "t3", "t4", "total", "true_label", "prediction",
              "correctness", "exit_taken", "data%"]
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{args.dataset_name}_accuracy_results_{args.backbone}_{args.variant}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow([r[k] for k in header])
    acc = np.mean([r["correctness"] for r in rows]) if rows else float("nan")
    ms = np.mean([r["total"] for r in rows]) * 1000 if rows else float("nan")
    dist = {}
    for r in rows: dist[r["exit_taken"]] = dist.get(r["exit_taken"], 0) + 1
    print(f"[done] {out}  rows={len(rows)} acc={acc:.3f} mean_exec={ms:.2f}ms exit_dist={dist}")


if __name__ == "__main__":
    main()
