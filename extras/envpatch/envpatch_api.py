# -*- coding: utf-8 -*-
"""envpatch patch layer: parts-registration interface (a formal entry for parts added by patches).

补丁（Pxxx_*.py）在环境建好、每集 reset 之后把自己的部位注册进来；ctx.parts 建快照时
合并，专家照字段表读，不另开一条私有读法。

接口：
  register(patch_id, version, task, provider, assets=(), note="")   # provider(ad, st) -> 见下
  collect(ad, st, task) -> dict(parts/identity/state_plan/frames/joints/hulls/missing/patches)
  augment(snapshot, ad, st, task)                                   # 就地并进本集快照（由宿主运行时调用）

provider 返回（全部可选，缺 = 这条补丁本集没东西可给）：
  dict(parts={uid: {部位名: {kind, ...}, ...}},
       identity={uid: {src, source_kind, asset_sha256, ...}},      # 并进 snapshot.identity["uids"][uid]
       state_plan={uid: {kind: "rigid"|"articulation", links: [...], joints: [...]}},   # 逐帧读回要它
       frames={uid: {up: 0/1/2, ...}},                             # 开局检查 / 世界查询用
       joints={uid: {q_name: {axis, type, limits, child_link, parent_link}}},
       hulls={uid: [[x,y,z], ...]},                                # 当集局部、已乘 body_scale 的凸包（碰撞用）
       missing=[{path, reason}])

规矩（失败也要好：failures must be loud）：
- 默认重名记 missing；显式 refine_parts 仅可细化同一已核URDF的网格推导部位，逐部位留账；
- provider 抛异常 / 返回坏形状 ⇒ 记 missing（envpatch:<id>），链上不能当「没这条补丁」；
- identity 里逐 uid 记 src = envpatch:<id>@<version>，assets 的 sha 由补丁自己给，这里只登记；
- 不 import 仿真；显式细化时只读登记URDF核完整SHA，不读取任意路径。
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional

REGISTRY: List[Dict[str, Any]] = []


def register(patch_id: str, version: str, task: str, provider: Callable[[Any, Any], Dict[str, Any]], *,
             assets: Optional[Dict[str, str]] = None, note: str = "", required: bool = False, refine_parts=None) -> Dict[str, Any]:
    """登记一条补丁的部位接口。同 (patch_id, task) 重复登记 ⇒ 后一条覆盖前一条（重载补丁时不叠）。"""
    rec = dict(id=str(patch_id), version=str(version), task=str(task), provider=provider,
               assets=dict(assets or {}), note=str(note), required=bool(required), refine_parts=copy.deepcopy(refine_parts or {}))
    REGISTRY[:] = [r for r in REGISTRY if not (r["id"] == rec["id"] and r["task"] == rec["task"])]
    REGISTRY.append(rec)
    return rec


def unregister(patch_id: Optional[str] = None):
    if patch_id is None:
        REGISTRY.clear()
    else:
        REGISTRY[:] = [r for r in REGISTRY if r["id"] != patch_id]


def registrations(task: Optional[str] = None) -> List[Dict[str, Any]]:
    return [r for r in REGISTRY if task is None or r["task"] == str(task)]


def _clean_parts(uid: str, parts: Any, missing: List[Dict[str, str]]) -> Dict[str, Any]:
    if not isinstance(parts, dict):
        missing.append(dict(path=uid, reason="envpatch 部位表不是 dict"))
        return {}
    out = {}
    for name, p in parts.items():
        if not isinstance(p, dict) or "kind" not in p:
            missing.append(dict(path=f"{uid}.{name}", reason="envpatch 部位不是 {kind, ...} 的 dict"))
            continue
        out[str(name)] = copy.deepcopy(p)
    return out


def collect(ad, st, task: str) -> Dict[str, Any]:
    """收当集全部已登记补丁的部位。绝不抛异常；坏补丁变成 missing 行。"""
    out: Dict[str, Any] = dict(parts={}, identity={}, state_plan={}, frames={}, joints={}, hulls={},
                               missing=[], patches=[], required_missing=[], required_uids=[], refine_parts={})
    for rec in registrations(task):
        missing_before=len(out["missing"])
        pid = f"{rec['id']}@{rec['version']}"
        out["patches"].append(dict(id=rec["id"], version=rec["version"], assets=dict(rec["assets"]), note=rec["note"]))
        try:
            got = rec["provider"](ad, st) or {}
        except Exception as ex:  # noqa: BLE001
            item=dict(path=f"envpatch:{rec['id']}", reason=f"provider 失败：{type(ex).__name__}: {ex}")
            out['missing'].append(item)
            if rec.get('required'): out['required_missing'].append(item)
            continue
        if not isinstance(got, dict):
            item=dict(path=f"envpatch:{rec['id']}", reason="provider 返回值不是 dict")
            out['missing'].append(item)
            if rec.get('required'): out['required_missing'].append(item)
            continue
        # 字段坏形状属于provider缺项，不越过异常门崩进半合并状态。
        for field in ('parts','identity','state_plan','frames','joints','hulls'):
            value=got.get(field)
            if value is not None and not isinstance(value,dict):
                out['missing'].append(dict(path=f'envpatch:{rec["id"]}.{field}',reason='provider字段不是映射'))
                got=dict(got);got[field]={}
        if rec.get('required') and not got.get('parts'):
            out['missing'].append(dict(path=f'envpatch:{rec["id"]}.parts',reason='已启用必需provider未返回部位'))
        if got.get('missing') is not None and not isinstance(got['missing'],(list,tuple)):
            out['missing'].append(dict(path=f'envpatch:{rec["id"]}.missing',reason='provider缺项表不是序列'))
            got=dict(got);got['missing']=[]
        for uid, parts in (got.get("parts") or {}).items():
            uid = str(uid)
            if uid in out["parts"]:
                out["missing"].append(dict(path=uid, reason=f"duplicate_uid：{pid} 与已收补丁重复"))
                continue
            cl = _clean_parts(uid, parts, out["missing"])
            if not cl:
                continue
            for name, p in cl.items():
                p.setdefault("src", f"envpatch:{pid}")
            out["parts"][uid] = cl
            if uid in rec["refine_parts"]:
                out["refine_parts"][uid] = dict(rule=copy.deepcopy(rec["refine_parts"][uid]), assets=dict(rec["assets"]))
        for uid, ident in (got.get("identity") or {}).items():
            uid = str(uid)
            if uid in out["identity"]:
                out["missing"].append(dict(path=uid, reason=f"duplicate_identity：{pid}"))
                continue
            d = copy.deepcopy(ident) if isinstance(ident, dict) else {}
            d.setdefault("src", f"envpatch:{pid}")
            d.setdefault("source_kind", "envpatch")
            d["envpatch"] = dict(id=rec["id"], version=rec["version"], assets=dict(rec["assets"]))
            out["identity"][uid] = d
        for uid, plan in (got.get("state_plan") or {}).items():
            out["state_plan"][str(uid)] = copy.deepcopy(plan)
        for uid, fr in (got.get("frames") or {}).items():
            out["frames"][str(uid)] = copy.deepcopy(fr)
        for uid, jj in (got.get("joints") or {}).items():
            out["joints"][str(uid)] = copy.deepcopy(jj)
        for uid, H in (got.get("hulls") or {}).items():
            try:
                import numpy as np
                Ha = np.asarray(H, float).reshape(-1, 3)
                if len(Ha) < 4 or not np.isfinite(Ha).all():
                    out["missing"].append(dict(path=f"envpatch:{rec['id']}.{uid}", reason="补丁凸包顶点不合格"))
                    continue
                out["hulls"][str(uid)] = Ha
            except Exception as ex:  # noqa: BLE001
                out["missing"].append(dict(path=f"envpatch:{rec['id']}.{uid}", reason=f"补丁凸包解析失败：{type(ex).__name__}"))
        for m in (got.get("missing") or []):
            if isinstance(m, dict) and m.get("path"):
                out["missing"].append(dict(path=str(m["path"]), reason=str(m.get("reason", ""))))
        if rec.get('required'):
            out['required_missing'] += copy.deepcopy(out['missing'][missing_before:])
            out['required_uids'] += list((got.get('parts') or {}).keys())
    return out


def _compatible(old,new):
    if isinstance(old,dict) and isinstance(new,dict):
        return all(k not in old or _compatible(old[k],v) for k,v in new.items())
    try:
        import numpy as np
        a,b=np.asarray(old),np.asarray(new)
        if a.shape!=b.shape:return False
        if a.dtype.kind in 'iuf' and b.dtype.kind in 'iuf':return bool(np.allclose(a,b,atol=1e-7,rtol=0,equal_nan=False))
        return bool(np.array_equal(a,b))
    except Exception:return str(old)==str(new)


def _refinement(snapshot, got, uid, names):
    """只细化同一已核补丁URDF的网格推导部位；其余同名仍拒收。"""
    rec=got['refine_parts'].get(uid)
    if not rec:return False
    from pathlib import Path
    import hashlib
    ident=snapshot.identity.get('uids',{}).get(uid,{})
    rule=rec['rule']
    try:
        path=Path(rule['asset_path']).resolve()
        digest=rule['asset_sha256']
        if path.suffix.lower()!='.urdf':return False
        if ident.get('source_kind')!='mesh' or ident.get('annotation'):return False
        if Path(ident.get('fpath') or '').resolve()!=path:return False
        if digest not in rec['assets'].values() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:return False
        if not set(names).issubset(set(rule['names'])):return False
        incoming=got['joints'].get(uid,{})
        topology=ident.get('joint_topology') or {}
        if not incoming or any(k not in topology or not _compatible(topology[k],v) for k,v in incoming.items()):return False
        if got['identity'].get(uid,{}).get('source_kind')!='envpatch':return False
        return True
    except (KeyError,TypeError,ValueError,OSError):return False


def augment(snapshot, ad, st, task: str) -> Dict[str, Any]:
    """把 collect() 的结果并进本集快照（就地改）。返回合并摘要（给录制头 / 日志）。"""
    got = collect(ad, st, task)
    summary = dict(patches=got["patches"], parts={u: sorted(v) for u, v in got["parts"].items()},
                   missing=got["missing"], required_missing=got['required_missing'], refined_parts={})
    if not got["patches"]:
        return summary
    for uid, parts in got["parts"].items():
        if uid in snapshot.parts:
            dup = [k for k in parts if k in snapshot.parts[uid]]
            refined=bool(dup) and _refinement(snapshot,got,uid,dup)
            if refined:
                rule=got['refine_parts'][uid]['rule']
                summary['refined_parts'][uid]=dict(names=dup,asset_path=rule['asset_path'],asset_sha256=rule['asset_sha256'],previous_source='mesh')
                snapshot.identity['uids'][uid]['envpatch_refinement']=copy.deepcopy(summary['refined_parts'][uid])
            if dup and not refined:
                got["missing"].append(dict(path=f"{uid}.{dup[0]}", reason="duplicate_part：补丁部位与官方部位重名，保留官方"))
            for k, v in parts.items():
                if k not in snapshot.parts[uid] or (refined and k in dup):
                    snapshot.parts[uid][k] = v
                    ident = snapshot.identity.setdefault("uids", {}).setdefault(uid, {})
                    ident.setdefault('part_sources',{})[k]=copy.deepcopy(got['identity'].get(uid) or dict(src=v.get('src')))
                    ident.setdefault("parts", [])
                    if k not in ident["parts"]:
                        ident["parts"].append(k)
        else:
            snapshot.parts[uid] = parts
            snapshot.identity.setdefault("uids", {})[uid] = dict(got["identity"].get(uid) or {})
            snapshot.identity["uids"][uid].setdefault("parts", sorted(parts))
    for uid, ident in got["identity"].items():
        if uid not in snapshot.identity.get("uids", {}):
            snapshot.identity.setdefault("uids", {})[uid] = ident
    for bucket in ("state_plan", "frames", "joints"):
        tgt = getattr(snapshot, bucket)
        for uid, v in got[bucket].items():
            if not isinstance(v,dict):
                got['missing'].append(dict(path=f'{uid}.{bucket}',reason='provider子项不是映射'))
                continue
            if uid not in tgt:
                tgt[uid]=copy.deepcopy(v)
                continue
            if bucket=='state_plan':
                if tgt[uid].get('kind') != v.get('kind'):
                    got['missing'].append(dict(path=f'{uid}.state_plan.kind',reason='provider与实际部件种类冲突'))
                    continue
                for key in ('links','joints'):
                    old=tgt[uid].get(key) or []
                    more=v.get(key) or []
                    if not isinstance(more,(list,tuple)):
                        got['missing'].append(dict(path=f'{uid}.state_plan.{key}',reason='provider名单不是序列'))
                        continue
                    tgt[uid][key]=list(dict.fromkeys(list(old)+list(more)))
            else:
                for key,value in v.items():
                    if key in tgt[uid] and not _compatible(tgt[uid][key],value):
                        got['missing'].append(dict(path=f'{uid}.{bucket}.{key}',reason='provider与现有字段冲突，保留现有值'))
                    else:
                        tgt[uid].setdefault(key,copy.deepcopy(value))
    # CP.inject/parts_world读取identity.joint_topology，不能只合snapshot.joints而漏世界查询。
    import numpy as np
    for uid, joints in got['joints'].items():
        if not isinstance(joints,dict): continue
        topology=snapshot.identity.setdefault('uids',{}).setdefault(uid,{}).setdefault('joint_topology',{})
        for name,joint in joints.items():
            try:
                if not isinstance(joint,dict) or joint.get('type') not in ('prismatic','revolute'):
                    raise ValueError('关节type未提供/不支持')
                axis=np.asarray(joint.get('axis'),float);limits=np.asarray(joint.get('limits'),float)
                if axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-9:
                    raise ValueError('轴不是有限非零3向量')
                if limits.shape!=(2,) or not np.isfinite(limits).all() or limits[0]>=limits[1]:
                    raise ValueError('限位不是有限正行程2元组')
                if not joint.get('child_link') or not joint.get('parent_link'):
                    raise ValueError('父/子link缺失')
                if name in topology and not _compatible(topology[name],joint):
                    raise ValueError('provider与既有joint_topology冲突')
                topology.setdefault(name,copy.deepcopy(joint))
            except (ValueError,TypeError) as ex:
                got['missing'].append(dict(path=f'{uid}.joints.{name}',reason=str(ex)))
    if got["hulls"]:
        if not hasattr(snapshot, "hulls"):
            snapshot.hulls = {}
        for uid, H in got["hulls"].items():
            snapshot.hulls.setdefault(uid, H)
    # 合并时的重名/拓扑冲突同样不能让必需provider假装成功。
    for item in got['missing']:
        if any(str(item.get('path','')).split('.')[0] == str(uid) for uid in got['required_uids']):
            if item not in got['required_missing']:
                got['required_missing'].append(copy.deepcopy(item))
    if got["missing"]:
        snapshot.missing.extend([dict(path=m["path"], reason=m["reason"]) for m in got["missing"]])
    summary["missing"] = got["missing"]
    if got["patches"]:
        snapshot.identity["envpatch"] = copy.deepcopy(got["patches"])
    return summary
