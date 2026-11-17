# ware_* — 库导入的铃 / 盘 / 叉 / 勺

-1108--外形库导入。和 的 `shape_import.py`（瓶 / 杯 / 马克笔 / 篮子 / 奶盒）是兄弟模块：同一套目录、同一套标注格式、同一套 `index.json` / `variant.yaml` 字段，互不改对方的文件。

| 文件 | 作用 |
|---|---|
| `ware_import.py` | 导入步（**唯一碰 trimesh / scipy 的地方**）：Objaverse 元数据 + Sketchfab 搜索 → 许可筛（只留 CC-BY / CC0，NC / SA / ND 全剔）→ 下载 → 量（PCA、连通块、水密）→ 归一 → 写 `~/objaverse_ware/normalised/<uid>.v.npy / .f.npy / .c.npy / .json` |
| `ware_select.py` | 按类筛（长径比 / 圆度 / 平座+顶钮 / 多余连通块 / 柄宽）→ 打分 → 选件、尺寸铺开 → `selection.json`；`--candidates` 出 `_候选_<类>.png` |
| `ware_ready.py` | 生成步（**只 numpy**）：缩放到目标尺寸 → 摆位 → 自动标 ctx.parts 第 3 版（表 sha a84a3926，11-08 勘误版） → 关系自检 → `out/libimport/<类>/<变体>/` + `index.json` + `<类>_库_对比图.png` |
| `verify_ware.py` | 一次跑完的复核（给代码审）：重建逐文件比 sha、逐件许可 / 出处表、index 对盘 |
| `test_ware.py` | 12 条单测：关系、parts 第 3 版、框与质心、坐标系、mesh_parts 粒度与覆盖、确定性、index 与许可、source 键、不引仿真、表外部位 |

```bash
python3 ware_import.py normalise     # 建/刷新 npy 缓存（下载 + 归一，~2 min）
python3 ware_select.py               # 筛 + 选 + selection.json
python3 ware_ready.py                # 22 个变体 + 4 张对比图（没有产物 / 有关系被拒时退出码 1）
python3 verify_ware.py               # 复核（重建比 sha / 许可表 / index 对盘）
python3 -m unittest test_ware -v     # 12 条
```

## 坐标系（写在每份 `annotations.origin` 里）

| 类 | 原点 | 朝上 | 长轴 |
|---|---|---|---|
| bell | 底面中心，支撑面 z=0 | +z | — |
| plate | 底面中心，支撑面 z=0 | +z（碗面朝上） | — |
| fork / spoon | 包围盒中心 | +z（正面朝上） | +x，**头在 +L/2** |

朝向不是人工看的：PCA 主轴 → 铃 / 盘取「去掉后剩下两个方向最接近相等」的那根当对称轴，铃按「宽的一头朝下」、盘按面积加权外法向定正反；叉 / 勺按两端宽度判头（齿 / 碗那头宽），按外法向定正面。

## 筛法（`ware_select.py`，每条都可复算）

- **铃**：`contact/overall ≥ 0.70`（**平座**：触地环几乎和整铃同宽，悬挂钟只有 0.1–0.3）、`top band r/overall ≤ 0.40`（**顶上是个钮**，不是提梁 / 冠环）、`H/D ∈ [0.40, 1.00]`（低圆顶，不是高钟）、圆度 ≤0.25、连通块 ≤120、最大块占比 ≥0.10。另有**人工剔除表** `MANUAL_DROP`（候选图看过、形状不对的，写明理由）：`Campana Texturizada` 钟形、平座+顶钮不明确。
- **叉 / 勺**：长径比、头 / 柄宽比、柄宽、`|握点 − 质心| ≤ 0.25 L`，外加**多余连通块筛**：某个连通块面数 <2%、厚度比 <0.05、长度 >2% 全长（或零厚度）→ 判为柄上的独立薄片，整件不用。被这条剔掉：`8aea6a`（0.5×138.8×9.5 mm 剃刀片，0.6% 面）、`02c0dd`（28.6×588×99 mm 薄条 + 10 个碎条，1.4%）、`611288`（零厚度块）、`1c5728`（叉，0.4%×0.5 L 薄片）。
- **盘**：口径、高 / 径、圆度、底半径、沿厚。

## 尺寸（整体缩放，不改比例）

Objaverse 没有真实单位，所以只按**长径比**筛、按目标尺寸**整体缩放**；YCB 三件是真尺度真件，同样缩放到目标尺寸以铺满档位（真不真另说，档位是我们要覆盖的范围）。

| 类 | 档位 | 依据 |
|---|---|---|
| bell（5 件） | 总径 75 / 84 / 93 / 102 / 110 mm | 参考资料显示实物铃比仿真（Ø63）大；档位是工程包络（ENG），**没有实物实测** |
| plate（6 件） | 口径 165 / 180 / 195 / 210 / 225 / 240 mm | 常见餐盘 6.5–9.5 英寸 |
| fork（6 件） | 长 175 / 183 / 190 / 198 / 206 / 215 mm | 叉 203 |
| spoon（5 件） | 长 135 / 144 / 153 / 162 / 171 mm | 勺 145 |

## 关系（生成后全查，任一不过就不写这个变体）

- **铃** BK1 总径 70–110 mm（参考资料显示实物更大）· BK2 高 / 总径 0.40–1.00 · BK3 圆度 ≤0.25 · BK4 触地半径 ≥ 0.30×总半径 · BK5 行程 == 5 mm（声明关节）· BK6 按区半径 ∈ [0.15, 0.95]×盖半径 · **BK7 触地半径 / 总半径 ≥ 0.70（平座）** · **BK8 顶部带半径 / 总半径 ≤ 0.40（顶钮）**。
- **盘** PL1 口径 150–260 mm · PL2 高 / 径 0.04–0.32 · PL3 圆度 ≤0.18 · PL4 底半径 ≤ 口径半径 · PL5 沿厚 > 0。
- **叉 / 勺** F1/S1 长在包络内 · F2/S2 头宽 · F3/S3 柄宽 2–18 mm（参考资料显示实物柄更宽）· F4/S4 握点在柄上（颈以前）· **F5/S5 握点到质心 ≤ 0.25 全长**（握颈部在重心附近）· F6/S6 柄宽 ≤ 85 mm 夹爪 · F7/S7 质量 5–180 g。

## ctx.parts 第 3 版（每份 `annotations.json` 的 `parts` 块）

- 铃 `button`：`origin` / `com` / `press`（joint：q_name=press、axis −z、range [−5,0] mm、travel、unit m、link=button_cover）/ `cover_top`（disk：盖顶、normal +z、section.r_out=**按区半径**，另给盖直径、盖高、总径，`provisional: true`）/ `base`（disk：触地圆心、normal −z、section.r_out=触地半径）。
- 盘 `plate`：`origin` / `com` / `center`（disk：盘心、normal +z、section.r_out=口径半径）/ `bottom`（disk：外底、normal −z、`placement_surface: true`——**它就是放置面**，对应 ctx.parts 的 placement，指挥  定「留」）/ `rim`（list，4 个最外点 + 外法向 + 沿厚）。
- 叉 / 勺：`origin` / `head` / `tail`（点，+x 是头）/ `body`（axis +x、全长、`profile` 沿轴外半径（s 从 tail 量，颈部就是 profile 变宽的那一段）、`grasp_s`）/ `handle`（list，柄中心线 5 点，每点 p / axis / size.width / thickness）/ `com`（`com_from: mass`，与 `mass.com` 同一个值）。
  - 这四件都在字段表第 3 版「摆桌 table_rearrangement」那几行里：`head` / `tail`（point）、`body`（axis，写「profile（颈部在哪）」）、`handle`（list）、`com`。**`neck` 不再作为单独部位**（第 3 版没有这一条），颈的位置从 `body.profile` 读。
- 另有 `mesh_parts`（按连通块的面区间，给下游碰撞分块；**连续段超过 64 条时只写一条全 face range + `mesh_parts_note: not decomposed`**，扫描件碎成 3 万块时下游没法用）、`geometry`（水密 / 朝向 / 面的实测结果）、`mass`（体积与质量 + `mass.source` 写清是「网格体积」还是「非水密壳的带符号体积估计」）。

## 出处与许可（每件都在 `annotations.source` / `variant.yaml` 里）

`base`（原名）、`license`、`author`、`uid` / `url`、`base_sha256`（下载的原始 glb/tgz 的 sha256）、`source_kind`（`objaverse_lvis` / `sketchfab_search` / `ycb_real_scan`）、`import_sha`（导入脚本前 16 位）、`native_extent_m`、`target_size_m`、`scale`。**只用 CC-BY / CC0**；NC / SA / ND 一件没用。

## 已知边界

1. 现成网格基本**不水密、多壳体**：`geometry.watertight` 照实写；质量是估计（非水密时用带符号网格体积，`mass.source` 写明），惯性用均匀包围盒，**不是真质量分布**。
2. 铃的按区是**几何顶部**，不是颜色（网格里没有「黄色」这回事）：`cover_top.r_out` 取顶盖顶部那一圈半径，标 `provisional: true`；黄按钮比例若有实测值，改一个参数即可。
3. 铃是**一个刚体网格 + 声明式 press 关节**（`joint_simulated: false`），URDF 两连杆归下游；底座是触地环，不是官方座环样式。
4. 叉 / 勺是**扫描 / 美术网格**：齿缝、碗深是原网格带的（比参数化版真），但**金属材质要靠仿真侧材质参数**，网格里没有。
5. 盘 / 铃的「中心」用口径平面（`center` 在 rim 高度），不是碗底；内底没单独量。
6. Objaverse 没有真实单位：**尺寸是按包络定的，不是量出来的**；只保证比例来自真件。
7. 铃的池子里严格符合「平座 + 顶钮」的只有 5 件（其余是教堂钟 / 手摇铃 / 提梁钟）；要更多得上更多关键词或人工建模。
