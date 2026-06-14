# baselines/run_nrms.py
"""Train + full-catalog eval NRMS on MIND (next-click retrieval). Emits results.json +
per_user_hits.npz (baseline + main_table generative vanilla, aligned by user_id) + MLflow.

Training: in-batch negatives — score each user against the batch's targets, cross-entropy to the
diagonal. Eval: score every user against ALL news title vectors (full catalog).

Usage: python -u baselines/run_nrms.py --vanilla-npz experiments/main_table/per_user_hits.npz \
       --output-dir experiments/nrms
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from baselines.data_mind import load_mind_for_nrms
from baselines.nrms import NRMS
from baselines.eval_fullcatalog import eval_full_catalog, write_per_user_hits
from mind_genrec.tracking import MlflowRunLogger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _hist_tensor(samples, news2idx, title_tokens, max_history):
    """(B, H, L) history title tokens + (B, H) mask, using the catalog title tensor."""
    L = title_tokens.shape[1]
    B = len(samples)
    out = np.zeros((B, max_history, L), dtype=np.int64)
    mask = np.zeros((B, max_history), dtype=bool)
    for i, s in enumerate(samples):
        hist = [h for h in s["history"] if h in news2idx][-max_history:]
        for j, nid in enumerate(hist):
            out[i, j] = title_tokens[news2idx[nid]]
            mask[i, j] = True
    return torch.tensor(out), torch.tensor(mask)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vanilla-npz", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-history", type=int, default=50)
    p.add_argument("--max-title-len", type=int, default=30)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--smoke-users", type=int, default=0)
    cli = p.parse_args()
    t0 = time.time()
    d = load_mind_for_nrms(cli.max_history, cli.max_title_len)
    model = NRMS(vocab=d.vocab_size).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    titles = torch.tensor(d.title_tokens, device=DEVICE)  # (M, L)

    # Smoke truncates TRAIN + VAL too (not just test) so Stage-1 is actually fast.
    train_samples = d.train_samples
    val_samples = d.val_samples
    if cli.smoke_users:
        train_samples = train_samples[:20 * cli.smoke_users]
        val_samples = val_samples[:cli.smoke_users]
        print(f"  SMOKE: train={len(train_samples)} val={len(val_samples)}", flush=True)

    def batch_titles(samples):
        ht, hm = _hist_tensor(samples, d.news2idx, d.title_tokens, cli.max_history)
        tgt = torch.tensor([d.news2idx[s["target"]] for s in samples])
        return ht.to(DEVICE), hm.to(DEVICE), tgt.to(DEVICE)

    rng = np.random.default_rng(42)
    tr = list(train_samples)
    def train_epoch():
        model.train(); rng.shuffle(tr); tot = 0.0
        for i in range(0, len(tr), 128):
            ht, hm, tgt = batch_titles(tr[i:i + 128])
            u = model.encode_user(ht, hm)                 # (B, dim)
            cv = model.encode_news(titles[tgt])           # in-batch positives (B, dim)
            logits = u @ cv.T                             # (B, B) — diagonal is positive
            loss = F.cross_entropy(logits, torch.arange(len(tgt), device=DEVICE))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        return tot

    @torch.no_grad()
    def full_scores(samples):
        model.eval()
        news_vecs = []
        for i in range(0, titles.shape[0], 1024):
            news_vecs.append(model.encode_news(titles[i:i + 1024]))
        NV = torch.cat(news_vecs)                         # (M, dim)
        out = []
        for i in range(0, len(samples), 256):
            ht, hm, _ = batch_titles(samples[i:i + 256])
            out.append((model.encode_user(ht, hm) @ NV.T).cpu().numpy())
        return np.concatenate(out, 0)

    def val_r10():
        s = full_scores(val_samples)
        tgt = np.array([d.news2idx[x["target"]] for x in val_samples])
        uid = np.array([x["user_id"] for x in val_samples])
        return eval_full_catalog(s, tgt, uid)[0]["R@10"]

    # Init best_state to the initial weights so load_state_dict never receives None
    # (e.g. if NO epoch improves over the -1.0 sentinel) (review #7).
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best, bad = -1.0, 0
    for ep in range(cli.epochs):
        tl = train_epoch(); r10 = val_r10()
        print(f"  ep{ep} train_loss={tl:.1f} val_R@10={r10:.4f}", flush=True)
        if r10 > best:
            best, best_state, bad = r10, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cli.patience:
                break
    model.load_state_dict(best_state)

    test = d.test_samples if not cli.smoke_users else d.test_samples[:cli.smoke_users]
    scores = full_scores(test)
    tgt = np.array([d.news2idx[x["target"]] for x in test])
    uid = np.array([x["user_id"] for x in test])
    agg, hits = eval_full_catalog(scores, tgt, uid)

    from baselines.metrics import KS
    van = np.load(cli.vanilla_npz, allow_pickle=True)
    pos = {u: i for i, u in enumerate(list(van["user_ids"]))}
    keep = [i for i, u in enumerate(uid) if u in pos]
    assert len(keep) > 0, (
        f"0 NRMS test users pair with the generative vanilla npz "
        f"({cli.vanilla_npz}); check user_id formats match (review #7)")
    sel = np.array([pos[uid[i]] for i in keep])
    # Compare only the cutoffs the generative vanilla actually persisted (the scorer writes @1/@10).
    avail = [k for k in KS if f"vanilla_hit{k}" in van.files]
    vh = {k: van[f"vanilla_hit{k}"][sel] for k in avail}
    bh = {k: hits[k][keep] for k in avail}
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    write_per_user_hits(cli.output_dir, uid[keep], bh, vh)

    results = {"model": "NRMS", "n_news": len(d.news2idx), "n_test": len(test),
               "rows": {"nrms": agg}, "runtime_sec": time.time() - t0}
    (cli.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    with MlflowRunLogger(enabled=True, tracking_uri="http://localhost:5000",
                         experiment_name="mind_genrec", run_name="nrms_mind",
                         tags={"dataset": "MIND-small", "model": "NRMS", "eval": "full_catalog"}) as lg:
        lg.log_params({"n_news": len(d.news2idx), "vocab": d.vocab_size, "max_history": cli.max_history})
        lg.log_metrics({f"nrms.{k}": v for k, v in agg.items()})
        lg.log_dict(results, "nrms_results.json")
    print(json.dumps(results["rows"], indent=2))

if __name__ == "__main__":
    main()
