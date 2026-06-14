# baselines/run_sasrec.py
"""Train + full-catalog eval SASRec on one Amazon dataset (LOO). Emits results.json +
per_user_hits.npz (baseline + matching generative vanilla, aligned by user_id) + MLflow.

Usage: python -u baselines/run_sasrec.py --dataset Sports \
       --vanilla-npz experiments/sports_scorer/per_user_hits.npz --output-dir experiments/sasrec_Sports
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from baselines.data_amazon import load_amazon_loo
from baselines.sasrec import SASRec
from baselines.eval_fullcatalog import eval_full_catalog, write_per_user_hits
from mind_genrec.tracking import MlflowRunLogger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _pad(hist, max_len):
    h = hist[-max_len:]
    return [0] * (max_len - len(h)) + h

def _batched_scores(model, seqs, max_len, bs=512):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            b = torch.tensor([_pad(h, max_len) for h, _, _ in seqs[i:i + bs]], device=DEVICE)
            out.append(model.full_scores(b).cpu().numpy())
    scores = np.concatenate(out, 0)
    # Exclude the PAD column (item id 0) from ranking: targets are 1..n_items so 0 is never a
    # valid candidate. Without this, PAD could outrank the target and inflate ranks (review #5).
    scores[:, 0] = -np.inf
    return scores

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--vanilla-npz", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-history", type=int, default=20)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--smoke-users", type=int, default=0)
    cli = p.parse_args()
    t0 = time.time()
    data = load_amazon_loo(cli.dataset, cli.max_history)
    model = SASRec(n_items=data.n_items, max_len=cli.max_history).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train = data.train
    rng = np.random.default_rng(42)
    def train_epoch():
        model.train(); rng.shuffle(train); tot = 0.0
        for i in range(0, len(train), 256):
            batch = train[i:i + 256]
            seq = torch.tensor([_pad(h, cli.max_history) for h, _ in batch], device=DEVICE)
            tgt = torch.tensor([t for _, t in batch], device=DEVICE)
            loss = F.cross_entropy(model.full_scores(seq), tgt)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        return tot

    def val_r10():
        s = _batched_scores(model, data.val, cli.max_history)
        tgt = np.array([t for _, t, _ in data.val])
        uid = np.array([u for _, _, u in data.val])
        agg, _ = eval_full_catalog(s, tgt, uid)
        return agg["R@10"]

    # Init best_state to the initial weights so load_state_dict never receives None
    # (e.g. if NO epoch improves over the -1.0 sentinel) (review #7).
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best, bad = -1.0, 0
    for ep in range(cli.epochs):
        train_epoch(); r10 = val_r10()
        if r10 > best:
            best, best_state, bad = r10, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cli.patience:
                break
    model.load_state_dict(best_state)

    test = data.test if not cli.smoke_users else data.test[:cli.smoke_users]
    scores = _batched_scores(model, test, cli.max_history)
    tgt = np.array([t for _, t, _ in test]); uid = np.array([u for _, _, u in test])
    agg, hits = eval_full_catalog(scores, tgt, uid)

    # Align vanilla hits to THIS test user order (by user_id).
    from baselines.metrics import KS
    van = np.load(cli.vanilla_npz, allow_pickle=True)
    van_uid = list(van["user_ids"])
    pos = {u: i for i, u in enumerate(van_uid)}
    keep = [i for i, u in enumerate(uid) if u in pos]
    assert len(keep) > 0, (
        f"0 SASRec test users pair with the generative vanilla npz "
        f"({cli.vanilla_npz}); check user_id formats match (review #7)")
    sel = np.array([pos[uid[i]] for i in keep])
    # Compare only the cutoffs the generative vanilla actually persisted (the scorer writes @1/@10).
    avail = [k for k in KS if f"vanilla_hit{k}" in van.files]
    vh = {k: van[f"vanilla_hit{k}"][sel] for k in avail}
    bh = {k: hits[k][keep] for k in avail}
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    write_per_user_hits(cli.output_dir, uid[keep], bh, vh)

    results = {"dataset": cli.dataset, "model": "SASRec", "n_items": data.n_items,
               "n_test": len(test), "rows": {"sasrec": agg}, "runtime_sec": time.time() - t0}
    (cli.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    with MlflowRunLogger(enabled=True, tracking_uri="http://localhost:5000",
                         experiment_name="mind_genrec", run_name=f"sasrec_{cli.dataset}",
                         tags={"dataset": cli.dataset, "model": "SASRec", "eval": "full_catalog_LOO"}) as lg:
        lg.log_params({"n_items": data.n_items, "max_history": cli.max_history, "epochs": cli.epochs})
        lg.log_metrics({f"sasrec.{k}": v for k, v in agg.items()})
        lg.log_dict(results, f"sasrec_{cli.dataset}_results.json")
    print(json.dumps(results["rows"], indent=2))

if __name__ == "__main__":
    main()
