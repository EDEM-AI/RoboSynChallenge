"""Pick 5-8 library shapes per class for ware_ready.py.

Filters are geometric and scale-invariant (Objaverse has no real units), plus a
name check done in ware_import.py.  The result is a hand-checkable table and
``selection.json``, so the choice is reproducible and reviewable:

    python3 ware_select.py            # print the pool + write selection.json
    python3 ware_select.py --top 20   # show more of the pool
"""

import glob
import json
import os
import sys

import numpy as np

import render

HERE = os.path.dirname(os.path.abspath(__file__))
NORM = os.path.expanduser("~/objaverse_ware/normalised")
SEL = os.path.join(HERE, "selection.json")

# reference size envelope: the size that is set per variant (m)
# candidates rejected by eye, with the reason kept here
MANUAL_DROP = {
    "bell": {"fc203fd849ea46f0b84d4843ba1aafcd": "钟形（Campana Texturizada），平座+顶钮不明确 → 不用"},
}

TARGETS = {
    "bell": [0.075, 0.084, 0.093, 0.102, 0.110],             # base diameter
    "plate": [0.165, 0.180, 0.195, 0.210, 0.225, 0.240],     # rim diameter
    "fork": [0.175, 0.183, 0.190, 0.198, 0.206, 0.215],      # length
    "spoon": [0.135, 0.144, 0.153, 0.162, 0.171],            # length
}

# (kind of check) -> limits; see check()
LIMITS = {
    "bell": {"round": 0.25, "h_over_d": (0.40, 1.00), "comp": 120, "share": 0.10,
             "contact": 0.70, "top_r": 0.40},
    "plate": {"round": 0.18, "h_over_d": (0.04, 0.32), "comp": 30, "share": 0.28},
    "fork": {"l_over_w": 4.0, "w_over_t": 1.05, "comp": 25, "share": 0.30,
             "handle_frac": (0.012, 0.100), "head_over_handle": 1.25},
    "spoon": {"l_over_w": 2.8, "w_over_t": 1.05, "comp": 25, "share": 0.30,
              "handle_frac": (0.012, 0.120), "head_over_handle": 1.25},
}


def load_pool():
    rows = []
    for p in sorted(glob.glob(os.path.join(NORM, "*.json"))):
        m = json.load(open(p, encoding="utf-8"))
        cls = m["classes"][0]
        e = m["extent"]
        r = {"cls": cls, "key": m["key"], "name": m["name"], "author": m["author"],
             "license": m["license"], "faces": m["faces"],
             "components": m["components"], "share": m["largest_component_share"],
             "watertight": m["watertight"], "ext": e,
             "body_w": m.get("body_w"), "head_w": m.get("head_w")}
        if cls in ("bell", "plate"):
            D = max(e[0], e[1])
            r.update({"H": e[2], "D": D, "round": abs(e[0] - e[1]) / max(e[0], e[1]),
                      "h_over_d": e[2] / max(D, 1e-9)})
        else:
            L, W, T = e[0], e[1], e[2]
            r.update({"L": L, "W": W, "T": T, "l_over_w": L / max(W, 1e-9),
                      "w_over_t": W / max(T, 1e-9),
                      "handle_frac": (r["body_w"] / L) if r["body_w"] else None})
        rows.append(r)
    return rows


def check(r):
    lim = LIMITS[r["cls"]]
    why = []
    if r["components"] > lim["comp"]:
        why.append(f"components {r['components']}>{lim['comp']}")
    if r["share"] < lim["share"]:
        why.append(f"largest share {r['share']:.2f}<{lim['share']}")
    if r["faces"] < 200:
        why.append("faces<200")
    if r["cls"] == "bell":
        st = bell_stats(r["key"])
        r.update(st)
        if st["contact"] < lim["contact"]:
            why.append(f"contact {st['contact']:.2f}<{lim['contact']} (flat seat)")
        if st["top_r"] > lim["top_r"]:
            why.append(f"top r {st['top_r']:.2f}>{lim['top_r']} (no button, handle?)")
    if r["cls"] in ("bell", "plate"):
        if r["round"] > lim["round"]:
            why.append(f"round {r['round']:.2f}>{lim['round']}")
        if not (lim["h_over_d"][0] <= r["h_over_d"] <= lim["h_over_d"][1]):
            why.append(f"H/D {r['h_over_d']:.2f} outside {lim['h_over_d']}")
    else:
        if r["l_over_w"] < lim["l_over_w"]:
            why.append(f"L/W {r['l_over_w']:.1f}<{lim['l_over_w']}")
        if r["w_over_t"] < lim["w_over_t"]:
            why.append(f"W/T {r['w_over_t']:.1f}<{lim['w_over_t']}")
        if r["handle_frac"] is None or not (lim["handle_frac"][0] <= r["handle_frac"]
                                           <= lim["handle_frac"][1]):
            why.append(f"handle/L {r['handle_frac']} outside {lim['handle_frac']}")
        stray = stray_sheet(r["key"])
        r["stray"] = stray
        if stray:
            why.append(f"stray sheet: {stray}")
        hh = r["head_w"] / max(r["body_w"] or 1e-9, 1e-9) if r["head_w"] else 0
        r["head_over_handle"] = hh
        if hh < lim["head_over_handle"]:
            why.append(f"head/handle {hh:.2f}<{lim['head_over_handle']}")
    return why


def bell_stats(key):
    """「平座 + 顶钮」: contact radius / max radius, and the top-band radius ratio."""
    v = np.load(os.path.join(NORM, key + ".v.npy"))
    v = v - np.array([0.0, 0.0, v[:, 2].min()])
    H = float(v[:, 2].max())
    r = np.linalg.norm(v[:, :2], axis=1)
    maxr = float(r.max())
    bot = r[v[:, 2] < 0.02 * H]
    top = r[v[:, 2] > 0.90 * H]
    return {"contact": float(np.percentile(bot, 90) / maxr) if len(bot) else 1.0,
            "top_r": float(np.percentile(top, 90) / maxr) if len(top) else 1.0}


def stray_sheet(key):
    """A small, thin, long detached component (a stray sliver on the handle)."""
    try:
        v = np.load(os.path.join(NORM, key + ".v.npy"))
        f = np.load(os.path.join(NORM, key + ".f.npy"))
        c = np.load(os.path.join(NORM, key + ".c.npy"))
    except FileNotFoundError:
        return ""
    L = float(v[:, 0].max() - v[:, 0].min())
    for cid in sorted(set(c.tolist())):
        sel = c == cid
        vv = np.unique(f[sel])
        e = v[vv].max(0) - v[vv].min(0)
        frac = sel.sum() / len(f)
        thin = e.min() / max(e.max(), 1e-12)
        if frac >= 0.003 and e.min() <= 0.0:
            return f"comp{int(cid)} {frac*100:.1f}% degenerate (zero thickness)"
        if 0.003 <= frac < 0.02 and thin < 0.05 and e.max() > 0.02 * L:
            return f"comp{int(cid)} {frac*100:.1f}% thin {thin:.3f} len {e.max()/L:.2f}L"
    return ""


def score(r):
    """Prefer candidates in the middle of the envelope, few components."""
    s = 0.0
    if r["cls"] in ("bell", "plate"):
        s += abs(r["h_over_d"] - (sum(LIMITS[r["cls"]]["h_over_d"]) / 2))
    else:
        s += abs(r["l_over_w"] - 6.0) / 6.0
        s += abs(r["handle_frac"] - 0.02) / 0.02
    s += 0.30 * r["components"] / max(LIMITS[r["cls"]]["comp"], 1)
    s -= 0.40 if r["watertight"] else 0.0
    s -= 1.00 if r["key"].startswith("ycb_") else 0.0
    return s


def candidate_sheets(rows, n=12):
    """One contact sheet per class: what the pool actually contains (for humans)."""
    imgs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    title = {"bell": "铃", "plate": "盘", "fork": "叉", "spoon": "勺"}
    for cls in ("bell", "plate", "fork", "spoon"):
        pool = [r for r in rows if r["cls"] == cls]
        pool.sort(key=lambda r: (bool(check(r)), score(r)))
        tiles, meta = [], []
        for r in pool[:n]:
            v = np.load(os.path.join(NORM, r["key"] + ".v.npy"))
            f = np.load(os.path.join(NORM, r["key"] + ".f.npy"))
            if len(f) > 60000:                      # keep the render time sane
                f = f[:: max(1, len(f) // 60000)]
                used = np.unique(f)
                remap = {int(a): i for i, a in enumerate(used)}
                v = v[used]
                f = np.array([[remap[int(x)] for x in tri] for tri in f], dtype=np.int64)
            try:
                eye, tgt, fov = render.frame(v, direction=(0.35, -0.80, 0.42), fill=1.05)
                img = render.render(v, f, size=300, eye=eye, target=tgt, fov=fov,
                                    base_color=(0.55, 0.62, 0.70), floor_half=0.35, floor_tiles=8)
            except Exception:
                continue
            bad = check(r)
            cap = f"{r['key'][:6]} {'OK' if not bad else 'x'}"
            h, wd, _ = img.shape
            tw, th = render.text_size(cap, scale=2)
            tile = np.full((h + th + 16, max(wd, tw + 12), 3), 255, np.uint8)
            tile[8:8 + h, (tile.shape[1] - wd) // 2:(tile.shape[1] - wd) // 2 + wd] = img
            render.draw_text(tile, (tile.shape[1] - tw) // 2, h + 8, cap, scale=2)
            tiles.append(tile)
            meta.append({"key": r["key"], "name": r["name"], "cls": cls,
                         "pass": not bad, "why": bad, "license": r["license"],
                         "ext": [round(x, 4) for x in r["ext"]]})
        if not tiles:
            continue
        sheet = render.montage(tiles, cols=4, pad=8, backdrop=250)
        png = os.path.join(imgs, f"_候选_{title[cls]}.png")
        render.write_png(png, sheet)
        with open(os.path.join(os.path.dirname(NORM), f"candidates_{cls}.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1, ensure_ascii=False)
        print("candidates ->", png)


def main(argv):
    top = 20
    if "--top" in argv:
        top = int(argv[argv.index("--top") + 1])
    rows = load_pool()
    sel = []
    for cls in ("bell", "plate", "fork", "spoon"):
        pool = [r for r in rows if r["cls"] == cls]
        ok, bad = [], []
        for r in pool:
            why = check(r)
            if r["key"] in MANUAL_DROP.get(cls, {}):
                why = why + ["manual: " + MANUAL_DROP[cls][r["key"]]]
            (ok if not why else bad).append((r, why))
        ok.sort(key=lambda rw: score(rw[0]))
        print(f"\n== {cls}: {len(ok)} pass / {len(pool)} measured")
        for r, _ in ok[:top]:
            if cls in ("bell", "plate"):
                print(f"   OK {r['key'][:12]:13s} D={r['D']*1e3:7.1f} H={r['H']*1e3:7.1f} "
                      f"H/D={r['h_over_d']:5.2f} round={r['round']:.3f} c={r['components']:3d} "
                      f"{r['license']:9s} {r['name'][:36]}")
            else:
                print(f"   OK {r['key'][:12]:13s} L={r['L']*1e3:7.1f} W={r['W']*1e3:6.1f} "
                      f"T={r['T']*1e3:5.1f} L/W={r['l_over_w']:5.2f} "
                      f"hand/L={r['handle_frac']*1e2 if r['handle_frac'] else -1:5.2f}% "
                      f"{r['license']:9s} {r['name'][:32]}")
        for r, why in bad[:6]:
            print(f"   -- {r['key'][:12]:13s} {r['name'][:40]:42s} {', '.join(why)[:70]}")
        import ware_ready as WR
        picked, i = [], 0
        for r, _ in ok:
            if len(picked) == len(TARGETS[cls]):
                break
            tgt = TARGETS[cls][len(picked)]
            name = f"{cls}_{len(picked)+1:02d}_{r['key'][:6]}_{int(round(tgt*1e3))}"
            try:
                WR.build(cls, r["key"], tgt, name)
            except Exception as e:
                print(f"   !! build rejects {r['key'][:8]}: {str(e)[:90]}")
                continue
            picked.append((r, tgt))
        if len(picked) < len(TARGETS[cls]):
            print(f"   !! only {len(picked)} build, want {len(TARGETS[cls])}")
        for i, (r, tgt) in enumerate(picked, start=1):
            sel.append({"cls": cls, "index": i, "key": r["key"], "target_m": tgt,
                        "name": f"{cls}_{i:02d}_{r['key'][:6]}_{int(round(tgt*1e3))}",
                        "base": r["name"], "license": r["license"], "author": r["author"],
                        "measured": {k: (round(v, 6) if isinstance(v, float) else v)
                                     for k, v in r.items() if k in
                                     ("ext", "D", "H", "L", "W", "T", "round",
                                      "h_over_d", "l_over_w", "w_over_t", "handle_frac",
                                      "components", "share", "faces", "watertight")}})
    if "--candidates" in argv:
        print("\nrendering candidate sheets ...")
        candidate_sheets(rows)
    with open(SEL, "w", encoding="utf-8") as fh:
        json.dump(sel, fh, indent=1, ensure_ascii=False)
    print(f"\nselection -> {SEL}  ({len(sel)} variants)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
