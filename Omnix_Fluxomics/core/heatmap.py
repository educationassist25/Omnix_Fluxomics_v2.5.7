"""
core.heatmap — Step 5: Z-scores, hierarchical clustering and publication heatmaps.
Z = (value − mean) / SD, computed across samples for each row (row-wise, default) or across rows for each sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.cluster import hierarchy as hc
from scipy.spatial.distance import pdist

from .style import DIVERGING, SEQUENTIAL, INK2, group_colors

DISTANCES = {"euclidean": "Euclidean", "correlation": "Pearson correlation (1 − r)",
             "spearman": "Spearman correlation (1 − ρ)", "cityblock": "Manhattan", "cosine": "Cosine"}
LINKAGES = {"average": "Average (UPGMA)", "complete": "Complete", "ward": "Ward (Euclidean only)", "single": "Single"}


def zscore(M: pd.DataFrame, axis: str = "row", clip: float | None = 3.0) -> pd.DataFrame:
    if axis == "row":
        Z = M.sub(M.mean(axis=1), axis=0).div(M.std(axis=1, ddof=1), axis=0)
    else:
        Z = (M - M.mean()) / M.std(ddof=1)
    Z = Z.replace([np.inf, -np.inf], np.nan)
    return Z.clip(-clip, clip) if clip else Z


def _dist(X: np.ndarray, metric: str) -> np.ndarray:
    if metric == "spearman":
        R = pd.DataFrame(X.T).rank().to_numpy().T
        return pdist(R, "correlation")
    return pdist(X, metric)


def cluster_order(M: pd.DataFrame, metric: str = "euclidean", method: str = "average"):
    """Leaf order + linkage for the rows of M (missing values filled with the row mean, then 0)."""
    if len(M) < 3:
        return list(range(len(M))), None
    X = M.apply(lambda r: r.fillna(r.mean()), axis=1).fillna(0).to_numpy(float)
    if method == "ward":
        metric = "euclidean"
    d = np.nan_to_num(_dist(X, metric), nan=np.nanmax(_dist(X, metric)) if np.isfinite(_dist(X, metric)).any() else 1.0)
    L = hc.linkage(d, method=method)
    return list(hc.leaves_list(L)), L


CMAPS = {"Omnix diverging": "omnix_div", "Red–blue (RdBu)": "RdBu_r", "Viridis": "viridis", "Cividis": "cividis",
         "Plasma": "plasma", "Magma": "magma", "Purples": "Purples", "Blues": "Blues", "Red–yellow–blue": "RdYlBu_r",
         "Purple–green (PRGn)": "PRGn"}
PLOTLY_SCALE = {"Red–blue (RdBu)": "RdBu_r", "Viridis": "Viridis", "Cividis": "Cividis", "Plasma": "Plasma",
                "Magma": "Magma", "Purples": "Purples", "Blues": "Blues", "Red–yellow–blue": "RdYlBu_r",
                "Purple–green (PRGn)": "PRGn"}
ANNOT_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
                 "#7fb8e8", "#9c5a2e", "#8fd3b5", "#b5a900"]


def get_cmap(name: str, diverging: bool):
    if name in (None, "", "Omnix diverging"):
        return DIVERGING if diverging else SEQUENTIAL
    return plt.get_cmap(CMAPS.get(name, name))


def annotation_colors(info: pd.DataFrame, tracks: list, group_order: list | None = None) -> dict:
    """{track: {value: colour}}. Group uses the app's group colours; numeric tracks a green ramp."""
    import matplotlib.colors as mc
    out = {}
    for t in tracks:
        v = info[t]
        vals = list(dict.fromkeys(v.dropna()))
        if t == "Group":
            gc = group_colors(group_order or vals)
            out[t] = {x: gc.get(x, "#cccccc") for x in vals}
        elif pd.api.types.is_numeric_dtype(v) and len(vals) > 1:
            sv = sorted(vals)
            out[t] = {x: mc.to_hex(plt.get_cmap("Greens")(0.3 + 0.6 * i / max(1, len(sv) - 1))) for i, x in enumerate(sv)}
        else:
            if t in ("Time", "TimeLabel"):
                sv = sorted(vals, key=lambda x: float(str(x).split()[0]) if str(x).split()[0].replace(".", "").isdigit() else 1e9)
                out[t] = {x: mc.to_hex(plt.get_cmap("Greens")(0.3 + 0.6 * i / max(1, len(sv) - 1))) for i, x in enumerate(sv)}
            else:
                out[t] = {x: ANNOT_PALETTE[i % len(ANNOT_PALETTE)] for i, x in enumerate(vals)}
    return out


FONT_FAMILIES = ["Arial", "Helvetica", "Calibri", "Verdana", "Times New Roman", "Georgia", "Courier New",
                 "DejaVu Sans"]
DEFAULT_FONTS = dict(family="Arial", title=11.0, row=0.0, col=6.5, ann=6.5, scale=6.0,
                     group=0.0)   # row 0 = auto-fit; group = metabolite bracket labels (0 = auto from the row font)


def row_runs(groups: list) -> list:
    """Contiguous runs of equal group names: [(name, first index, last index), ...] (empty names skipped)."""
    out, start = [], 0
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[start]:
            if groups[start]:
                out.append((groups[start], start, i - 1))
            start = i
    return out


def channel_text(idx, ns: set | None = None) -> dict:
    """Short row text for 'Compound | M+i' rows: 'M+i', with ' (n.s.)' for channels in ns."""
    ns = ns or set()
    out = {}
    for r in idx:
        iso = str(r).rsplit(" | ", 1)[-1]
        out[r] = iso + (" (n.s.)" if r in ns else "")
    return out


def _iso_num(r) -> int:
    try:
        return int(str(r).rsplit("M+", 1)[-1])
    except ValueError:
        return 0


def grouped_order(Z: pd.DataFrame, cluster: bool = True, metric: str = "euclidean", method: str = "average",
                  names: dict | None = None) -> list:
    """
    Row order that keeps the isotopologue channels of each metabolite together ('Compound | M+i' rows), M+0 … M+n
    within a metabolite. Metabolite blocks are ordered by hierarchical clustering of their mean channel profile
    (cluster=True, ≥ 3 metabolites) or alphabetically by display name.
    """
    comp = [str(r).rsplit(" | ", 1)[0] for r in Z.index]
    blocks = {}
    for r, c in zip(Z.index, comp):
        blocks.setdefault(c, []).append(r)
    for c in blocks:
        blocks[c].sort(key=_iso_num)
    keys = list(blocks)
    if cluster and len(keys) > 2:
        P = Z.fillna(0).groupby(np.array(comp)).mean().reindex(keys)
        keys = [keys[i] for i in cluster_order(P, metric, method)[0]]
    else:
        keys.sort(key=lambda c: str((names or {}).get(c, c)).lower())
    return [r for c in keys for r in blocks[c]]


def balanced_cells(n_rows: int, n_cols: int) -> tuple:
    """
    Cell (width, height) in inches for a balanced, publication-proportioned heatmap body: rows 0.11–0.22 in (taller
    for short heatmaps so labels stay readable), body width ≈ 0.8 × body height within 2.6–7 in, cells 0.14–0.45 in
    wide. Few rows and few samples give near-square cells; long heatmaps stay ≤ 7 in wide.
    """
    ch = min(0.22, max(0.11, 5.5 / max(n_rows, 1)))
    target_w = min(7.0, max(2.6, ch * n_rows * 0.8))
    cw = min(0.45, max(0.14, target_w / max(n_cols, 1)))
    return round(cw, 3), round(ch, 3)


def _families(name: str) -> list:
    """Requested family first, then metric-compatible fallbacks; only installed fonts (no findfont warnings)."""
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    alt = {"Arial": ["Liberation Sans", "Arimo"], "Helvetica": ["Arial", "Liberation Sans"],
           "Times New Roman": ["Liberation Serif", "Tinos", "DejaVu Serif"], "Georgia": ["DejaVu Serif"],
           "Courier New": ["Liberation Mono", "DejaVu Sans Mono"], "Calibri": ["Carlito", "Arial"]}
    out = [f for f in [name] + alt.get(name, []) if f in have]
    return out + ["DejaVu Sans"]


def text_width(texts, fs: float, family: list, bold: bool = False) -> float:
    """Width (in) of the widest string when typeset at fs pt in the given font family (exact measurement)."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath
    prop = FontProperties(family=family, size=fs, weight="bold" if bold else "normal")
    ttp = TextToPath()
    best = 0.0
    for t in texts:
        try:
            w = ttp.get_text_width_height_descent(str(t), prop, ismath=False)[0]
        except Exception:
            w = len(str(t)) * 0.6 * fs
        best = max(best, w)
    return best / 72


def draw_heatmap(M: pd.DataFrame, col_info: pd.DataFrame, *, title: str = "", value_label: str = "Z-score",
                 diverging: bool = True, vmin=None, vmax=None, cluster_rows=True, cluster_cols=False,
                 metric="euclidean", method="average", row_labels: bool = True, width: float = 10,
                 height: float | None = None, group_order: list | None = None, highlight: list | None = None,
                 annotations: list | None = None, cmap: str | None = None, fonts: dict | None = None,
                 cell_w: float | None = None, cell_h_in: float = 0.24, row_text: dict | None = None,
                 row_groups: dict | None = None, separators: bool = True):
    """
    Publication heatmap with optional dendrograms and column annotation bars (any columns of col_info).
    Annotation legend (group, time, …) on the right side, vertically centred; colour scale underneath the heatmap.
    fonts: family, title, row (0 = auto-fit to the row height), col, ann (annotation labels and legend), scale (pt).
    Every band (title, annotation bars, column labels, row labels, legend) is sized from its text, so larger fonts
    never overlap. Returns (figure, row order list, column order list).
    row_text: row id -> displayed label (e.g. 'M+2 (n.s.)'); row_groups: row id -> group name (e.g. the metabolite):
    contiguous runs get a bracket and the group name to the right of the row labels (pass cluster_rows=False and
    rows already ordered by group); separators draws a thin white line between groups.
    """
    fo = {**DEFAULT_FONTS, **(fonts or {})}
    fam = _families(fo["family"])
    cw = lambda fs: 0.54 * fs / 72                  # mean character width (in) for a font size (pt)
    M = M.loc[M.notna().sum(axis=1) > 1]
    rows = list(M.index); cols = list(M.columns)
    ro = cluster_order(M, metric, method) if cluster_rows else (list(range(len(rows))), None)
    co = cluster_order(M.T, metric, method) if cluster_cols else (list(range(len(cols))), None)
    M = M.iloc[ro[0], co[0]]
    n_r, n_c = M.shape
    info = col_info.reindex(M.columns)
    tracks = [t for t in (annotations if annotations is not None else ["Group", "TimeLabel", "Replicate"])
              if t in info.columns]
    acol = annotation_colors(info, tracks, group_order)
    tname = lambda t: "Time" if t == "TimeLabel" else t
    row_fs_fixed = float(fo["row"]) if fo["row"] and fo["row"] > 0 else None
    cell_h = (max(cell_h_in, row_fs_fixed * 1.3 / 72) if row_fs_fixed else cell_h_in) if height is None else None
    H_title = (fo["title"] / 72 * 1.5 + 0.2) if title else 0.15
    H_dend = 0.75 if cluster_cols else 0.0
    H_leg = 0.0                                   # v2.3: annotation legend moved to the right side
    trk = max(0.135, fo["ann"] / 72 * 1.3)        # height of one annotation bar
    H_ann, H_gap = len(tracks) * (trk + 0.015), 0.06
    H_xlab = text_width(M.columns, fo["col"], fam) + 0.12
    H_scale = fo["scale"] / 72 * 3.2 + 0.32        # colour scale underneath the heatmap: title + bar + ticks
    H_hm = (n_r * cell_h) if cell_h else max(height - (H_title + H_leg + H_dend + H_ann + H_gap + H_xlab + H_scale),
                                             1.0)
    # small heatmaps: keep room for the annotation legend (right side) inside the figure
    _leg_h = (sum(1 + len(acol[t]) for t in tracks) + max(len(tracks) - 1, 0) * 0.5) * max(0.17, fo["ann"] / 72 * 1.55)
    H_pad = max(0.0, _leg_h - (H_hm + H_xlab + H_scale) + 0.1) if tracks else 0.0
    H = H_title + H_leg + H_dend + H_ann + H_gap + H_hm + H_xlab + H_scale + H_pad
    row_fs = row_fs_fixed or max(3.5, min(7.5, 0.8 * 72 * H_hm / max(n_r, 1)))
    W_dend = 0.9 if cluster_rows else 0.0
    W_tl = text_width([tname(t) for t in tracks], fo["ann"], fam) + 0.12          # annotation-bar names (left)
    W_left = max(W_dend, W_tl)
    disp = [str((row_text or {}).get(r, r)) for r in M.index]
    W_lab = (text_width(disp, row_fs, fam) + 0.1) if row_labels else 0.15
    runs = row_runs([str((row_groups or {}).get(r, "")) for r in M.index]) if row_groups else []
    grp_fs = float(fo.get("group") or 0) or max(row_fs * 1.25, fo["ann"])
    W_grp = (0.16 + text_width([r[0] for r in runs], grp_fs, fam, bold=True) + 0.1) if runs else 0.0
    leg_txt = max([len(tname(t)) for t in tracks] +
                  [len(f"{v:g}" if isinstance(v, float) else str(v)) for t in tracks for v in acol[t]] + [6])
    leg_w = max(text_width([tname(t) for t in tracks], fo["ann"], fam, bold=True),
                text_width([f"{v:g}" if isinstance(v, float) else str(v) for t in tracks for v in acol[t]],
                           fo["ann"], fam) + min(0.16, max(0.17, fo["ann"] / 72 * 1.55) * 0.95) + 0.1)
    W_leg = (leg_w + 0.12) if tracks else 0.0
    W_hm = n_c * cell_w if cell_w else max(width - W_left - W_lab - W_grp - W_leg - 0.25, 2.0)
    Wt = 0.1 + W_left + W_hm + W_lab + W_grp + W_leg + 0.15
    title_w = text_width([title], fo["title"], fam, bold=True) + 0.3 if title else 0.0
    cb_tw = text_width([value_label], fo["scale"], fam) + 0.1
    Wt = max(Wt, title_w + 0.2, 0.1 + W_left + max(cb_tw, 1.0) + 0.3)
    fig = plt.figure(figsize=(Wt, H))
    fx = lambda inch: inch / Wt
    fy = lambda inch_from_top: 1 - inch_from_top / H
    x0 = 0.1 + W_left
    y_hm_top = H_title + H_leg + H_dend + H_ann + H_gap
    ax = fig.add_axes([fx(x0), fy(y_hm_top + H_hm), fx(W_hm), H_hm / H])
    if vmin is None or vmax is None:
        lim = np.nanmax(np.abs(M.values)) if diverging else None
        vmin = -lim if diverging else np.nanmin(M.values)
        vmax = lim if diverging else np.nanmax(M.values)
    im = ax.imshow(M.values, aspect="auto", cmap=get_cmap(cmap, diverging), vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(n_c)); ax.set_xticklabels(M.columns, rotation=90, fontsize=fo["col"], color="black")
    if row_labels:
        ax.yaxis.tick_right(); ax.set_yticks(range(n_r))
        ax.set_yticklabels(disp, fontsize=row_fs, color="black")
        if highlight:
            for t in ax.get_yticklabels():
                if M.index[int(round(t.get_position()[1]))] in highlight:
                    t.set_fontweight("bold"); t.set_color("#a8402c")
    else:
        ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    # ---- annotation bars
    for k, t in enumerate(tracks):
        yt = H_title + H_leg + H_dend + k * (trk + 0.015)
        a = fig.add_axes([fx(x0), fy(yt + trk), fx(W_hm), trk / H])
        for j, v in enumerate(info[t]):
            a.add_patch(Rectangle((j - 0.5, 0), 1, 1, color=acol[t].get(v, "#eeeeee"), lw=0))
        a.set_xlim(-0.5, n_c - 0.5); a.set_ylim(0, 1); a.axis("off")
        fig.text(fx(x0 - 0.06), fy(yt + trk / 2), tname(t), ha="right", va="center", fontsize=fo["ann"],
                 color="black")
    # ---- dendrograms
    if cluster_rows and ro[1] is not None:
        a = fig.add_axes([fx(0.1 + W_left - W_dend), fy(y_hm_top + H_hm), fx(W_dend * 0.95), H_hm / H])
        hc.dendrogram(ro[1], orientation="left", ax=a, no_labels=True, color_threshold=0,
                      above_threshold_color="#555555")
        a.invert_yaxis(); a.axis("off")
    if cluster_cols and co[1] is not None:
        a = fig.add_axes([fx(x0), fy(H_title + H_leg + H_dend), fx(W_hm), (H_dend * 0.95) / H])
        hc.dendrogram(co[1], ax=a, no_labels=True, color_threshold=0, above_threshold_color="#555555")
        a.axis("off")
    # ---- grouped rows: separators, brackets and group names (right of the row labels)
    if runs:
        row_h = H_hm / max(n_r, 1)
        xb = x0 + W_hm + W_lab + 0.04
        inset = min(0.18 * row_h, 0.03)
        from matplotlib.lines import Line2D
        for k, (name, a0, a1) in enumerate(runs):
            if separators and k > 0:
                ax.axhline(a0 - 0.5, color="white", lw=1.0)
            yt, yb = y_hm_top + a0 * row_h + inset, y_hm_top + (a1 + 1) * row_h - inset
            fig.add_artist(Line2D([fx(xb), fx(xb)], [fy(yt), fy(yb)], color="black", lw=1.1,
                                  solid_capstyle="butt"))
            fig.text(fx(xb + 0.08), fy((yt + yb) / 2), name, ha="left", va="center", fontsize=grp_fs,
                     fontweight="bold", color="black")
    # ---- annotation legend: right side, vertically centred on the heatmap
    xl = x0 + W_hm + W_lab + W_grp + 0.05
    n_lines = sum(1 + len(acol[t]) for t in tracks) + max(len(tracks) - 1, 0) * 0.5
    line = max(0.17, fo["ann"] / 72 * 1.55)
    leg_h = n_lines * line
    y_cur = y_hm_top + max((H_hm - leg_h) / 2, 0)
    if tracks:
        lg = fig.add_axes([fx(xl), fy(y_cur + leg_h), fx(W_leg), leg_h / H]); lg.axis("off")
        lg.set_xlim(0, W_leg); lg.set_ylim(leg_h, 0)
        yy = 0.0
        sw = min(0.16, line * 0.95)
        for t in tracks:
            lg.text(0, yy + line / 2, tname(t), va="center", fontsize=fo["ann"], fontweight="bold", color="black")
            yy += line
            for v, c in acol[t].items():
                lab = f"{v:g}" if isinstance(v, float) else str(v)
                lg.add_patch(Rectangle((0.02, yy + (line - sw * 0.75) / 2), sw, sw * 0.75, color=c))
                lg.text(0.04 + sw + 0.06, yy + line / 2, lab, va="center", fontsize=fo["ann"], color="black")
                yy += line
            yy += line * 0.5
    # ---- colour scale at the bottom-right, directly underneath the heatmap (below the column labels); its title
    #      sits above the bar and the bar is at least as wide as the title, so the label never overlaps
    cb_title = fo["scale"] / 72 * 1.6 + 0.04                 # title line, clear of the bar
    cb_t = 0.13                                               # bar thickness (in)
    cb_w = max(min(1.3, 0.3 * W_hm), cb_tw, 0.9)              # short bar, never narrower than its title
    cb_x = max(x0 + W_hm - cb_w, 0.1)
    cax = fig.add_axes([fx(cb_x), fy(y_hm_top + H_hm + H_xlab + cb_title + 0.06 + cb_t), fx(cb_w), cb_t / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=fo["scale"], length=1.5, pad=1, colors="black")
    cb.ax.set_title(value_label, fontsize=fo["scale"], pad=4, loc="right", color="black")
    cb.outline.set_linewidth(0.4)
    cb.set_ticks(cb.get_ticks()[(cb.get_ticks() >= vmin) & (cb.get_ticks() <= vmax)])
    if title:
        fig.text(0.5, fy(0.12), title, fontsize=fo["title"], fontweight="bold", va="top", ha="center",
                 color="black")
    import matplotlib.text as mtext
    for t in fig.findobj(mtext.Text):
        t.set_fontfamily(fam)
    return fig, list(M.index), list(M.columns)
