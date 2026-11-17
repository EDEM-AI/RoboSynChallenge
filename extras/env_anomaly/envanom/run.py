"""跑一批：每集一行（检测器、结论、证据）+ 批级汇总 + 训练视图清单。

  python -m envanom.run rsrec    <录制目录…>  --out <输出目录> [--label 名]

输出：episodes.jsonl（每集一行：来源、任务、种子、集号、episode_action、results 全量）· findings.csv（每条非 clear / na 一行）·
summary.json / summary.md（任务 × 检测器 × 结论计数）· train_view.json（exclude / spot_check / official_only / expert_fail 清单）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict

import numpy as np

from . import VERSION
from .ctx import Ctx
from .detectors import DETECTORS, NOT_AUTO, RULESET, opening_ref_e02
from .load import load_rsrec, rsrec_files

ORDER = ("exclude", "spot_check", "official_only", "expert_fail", "flag", "none")


def episode_action(results):
    acts = {r["action"] for r in results}
    for a in ORDER:
        if a in acts:
            return a
    return "none"


def check_ctx(c, batch=None):
    res = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for d in DETECTORS:
            try:
                res += d(c, batch)
            except Exception as e:  # noqa: BLE001
                res.append(dict(det=d.__name__, bugs="?", verdict="unknown", kind=None, action="flag", t0=None, t1=None, values={},
                                rule=f"{RULESET}:{d.__name__}", missing=[f"检测器出错 {type(e).__name__}: {e}"[:200]], why=None))
    return res


def check_episode(ep, batch=None):
    """一集 → 记录行（v2 骨架接这一个函数即可；batch 参考没有时 E02 记 unknown）。"""
    c = Ctx(ep)
    res = check_ctx(c, batch)
    h = ep.get("header") or {}
    return dict(tool="env_anomaly", version=VERSION, ruleset=RULESET, source=ep.get("source"), task=h.get("task"), seed=h.get("seed"),
                episode=h.get("episode"), path=ep.get("path"), n_rows=c.n, contacts=c.has_contacts, hand_src=c.tips()[1],
                success=(ep.get("footer") or {}).get("success"), episode_action=episode_action(res), results=res)


def check_batch(eps):
    """一批（同任务同配置更好）：先算批级参考（E02 开局中位），再逐集。"""
    ctxs = [Ctx(ep) for ep in eps]
    batch = dict(e02=opening_ref_e02(ctxs))
    out = []
    for ep, c in zip(eps, ctxs):
        res = check_ctx(c, batch)
        h = ep.get("header") or {}
        out.append(dict(tool="env_anomaly", version=VERSION, ruleset=RULESET, source=ep.get("source"), task=h.get("task"), seed=h.get("seed"),
                        episode=h.get("episode"), path=ep.get("path"), n_rows=c.n, contacts=c.has_contacts, hand_src=c.tips()[1],
                        success=(ep.get("footer") or {}).get("success"), episode_action=episode_action(res), results=res))
    return out


# ------------------------------------------------------------------ 汇总
def summarize(recs, label=""):
    by = defaultdict(Counter)        # (task, det) → verdict 计数（每集每检测器取最重的结论计一次）
    ep_act = defaultdict(Counter)
    rank = dict(confirmed=0, expert=1, candidate=2, unknown=3, clear=4, na=5)
    for r in recs:
        r["task"] = r["task"] or "?"          # 没 header 的残片（.part 死在开头）
    for r in recs:
        ep_act[r["task"]][r["episode_action"]] += 1
        worst = {}
        for x in r["results"]:
            k = x["det"]
            if k not in worst or rank[x["verdict"]] < rank[worst[k]]:
                worst[k] = x["verdict"]
        for k, v in worst.items():
            by[(r["task"], k)][v] += 1
    tasks = sorted({r["task"] for r in recs})
    return dict(label=label, version=None, n=len(recs), tasks={t: dict(n=sum(ep_act[t].values()), episode_action=dict(ep_act[t])) for t in tasks},
                det={f"{t}|{d}": dict(c) for (t, d), c in sorted(by.items())})


def summary_md(S, title):
    L = [f"# {title}", "", f"集数 {S['n']}。每集每检测器取最重的结论计一次（confirmed > expert > candidate > unknown > clear > na）。", "",
         "## 每集处置（episode_action）", "", "| 任务 | 集 | exclude | spot_check | official_only | expert_fail | flag | none |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t, v in S["tasks"].items():
        a = v["episode_action"]
        L.append(f"| {t} | {v['n']} | " + " | ".join(str(a.get(k, 0)) for k in ORDER) + " |")
    L += ["", "## 检测器 × 结论（na / 全 clear 的行省略）", "", "| 任务 | 检测器 | confirmed | candidate | expert | unknown | clear |", "|---|---|---:|---:|---:|---:|---:|"]
    for k, c in S["det"].items():
        t, d = k.split("|")
        if set(c) <= {"na", "clear"}:
            continue
        L.append(f"| {t} | {d} | {c.get('confirmed', 0)} | {c.get('candidate', 0)} | {c.get('expert', 0)} | {c.get('unknown', 0)} | {c.get('clear', 0)} |")
    return "\n".join(L) + "\n"


def train_view(recs):
    """造数据接口：confirmed 且 action=exclude 的集从训练视图拿掉（原数据留）；candidate 抽核；official_only 进单独一册；expert_fail 进修专家清单。"""
    v = defaultdict(list)
    for r in recs:
        key = dict(task=r["task"], seed=r["seed"], episode=r["episode"], path=r["path"])
        for a in ("exclude", "spot_check", "official_only", "expert_fail"):
            hits = [dict(det=x["det"], verdict=x["verdict"], t0=x["t0"], why=x["why"]) for x in r["results"] if x["action"] == a]
            if hits:
                v[a].append(dict(key, hits=hits[:6]))
    return dict(ruleset=RULESET, **{a: v.get(a, []) for a in ("exclude", "spot_check", "official_only", "expert_fail")})


def write_all(recs, out, label):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "episodes.jsonl"), "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False, default=_js) + "\n")
    with open(os.path.join(out, "findings.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "task", "seed", "episode", "det", "bugs", "verdict", "kind", "action", "t0", "t1", "rule", "values", "missing", "why"])
        for r in recs:
            for x in r["results"]:
                if x["verdict"] in ("clear", "na"):
                    continue
                w.writerow([r["source"], r["task"], r["seed"], r["episode"], x["det"], x["bugs"], x["verdict"], x["kind"], x["action"], x["t0"], x["t1"],
                            x["rule"], json.dumps(x["values"], ensure_ascii=False, default=_js)[:400], ";".join(x["missing"]), x["why"]])
    S = summarize(recs, label)
    S["version"] = RULESET
    json.dump(S, open(os.path.join(out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write(summary_md(S, f"环境异常读数 · {label}"))
    json.dump(train_view(recs), open(os.path.join(out, "train_view.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=_js)
    return S


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rsrec"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    t0 = time.time()
    recs = []
    if a.mode == "rsrec":
        for root in a.paths:
            groups = defaultdict(list)            # 按所在目录分批（一个目录 = 一次采集 / 评测），读完一批就丢，省内存
            for f in rsrec_files(root):
                groups[os.path.dirname(f)].append(f)
            for d, fs in sorted(groups.items()):
                eps = [load_rsrec(f) for f in fs]
                by = defaultdict(list)
                for ep in eps:
                    by[(ep.get("header") or {}).get("task")].append(ep)
                for _, g in by.items():
                    recs += check_batch(g)
                del eps, by
                print(f"{d}: {len(fs)} 集 · {time.time() - t0:.0f} s", file=sys.stderr)
    S = write_all(recs, a.out, a.label or a.mode)
    print(json.dumps(dict(n=S["n"], out=a.out, s=round(time.time() - t0, 1)), ensure_ascii=False))


if __name__ == "__main__":
    main()
