"""LMC association on aligned symlib Rockstar and Symfind main branches.

The LMC analog indices below were identified in the original
LMC_idx_indentifier_mwest notebook. This module reproduces the geometric
main-branch selection, not the separate corrected raw-HLIST pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np

LMC_INDEX_LEGACY = {
    0: 1, 1: 5, 2: 1, 3: 1, 4: 1, 5: 7, 6: 4, 7: 2,
    8: 1, 9: 2, 10: 1, 11: 1, 12: 2, 13: 1, 14: 1,
    15: 6, 16: 3, 17: 3, 18: 1, 19: 1,
}


@dataclass
class Catalogs:
    host_index: int
    host_name: str
    sim_dir: str
    scale: np.ndarray
    rockstar: np.ndarray
    symfind: np.ndarray
    rockstar_hist: np.ndarray
    symfind_hist: np.ndarray
    lmc_idx: int | None
    legacy_lmc_idx: int | None


def _validate_alignment(r, s, rh, sh, scale):
    if r.shape != s.shape or r.ndim != 2 or len(rh) != len(sh) or r.shape[0] != len(rh):
        raise ValueError("Rockstar and Symfind do not have aligned branch dimensions")
    if r.shape[1] != len(scale):
        raise ValueError("Snapshot count does not match scale factors")
    if "branch_idx" in rh.dtype.names and "branch_idx" in sh.dtype.names:
        if not np.array_equal(rh["branch_idx"], sh["branch_idx"]):
            raise ValueError("Rockstar/Symfind branch_idx entries are not aligned")
    for name, arr in (("Rockstar", r), ("Symfind", s)):
        if not {"x", "ok"}.issubset(arr.dtype.names):
            raise ValueError(f"{name} is missing x/ok fields: {arr.dtype.names}")
    if "rvir" not in r.dtype.names:
        raise ValueError("Rockstar must supply rvir for the virial-radius selection")


def _mapped_lmc_index(legacy_hist, rockstar_hist, legacy_idx):
    """Map an index found using read_subhalos onto read_rockstar ordering."""
    if ("branch_idx" in legacy_hist.dtype.names and
            "branch_idx" in rockstar_hist.dtype.names):
        target_branch = legacy_hist["branch_idx"][legacy_idx]
        matches = np.flatnonzero(rockstar_hist["branch_idx"] == target_branch)
        if len(matches) != 1:
            raise ValueError(f"Could not uniquely match LMC branch_idx={target_branch}")
        return int(matches[0])
    if len(legacy_hist) != len(rockstar_hist):
        raise ValueError("Legacy and Rockstar lengths differ; cannot safely reuse LMC indices")
    # No branch identifiers available to test alignment. In this case the
    # published symlib index convention is assumed; inspect the LMC track.
    return legacy_idx


def load_catalogs(base_dir, suite, host_index):
    """Load matched main branches for one symlib host."""
    import symlib

    sim_dir = symlib.get_host_directory(str(base_dir), suite, host_index)
    scale = np.asarray(symlib.scale_factors(sim_dir), dtype=float)
    r, rh = symlib.read_rockstar(sim_dir)
    s, sh = symlib.read_symfind(sim_dir)
    _validate_alignment(r, s, rh, sh, scale)

    legacy_idx = LMC_INDEX_LEGACY.get(host_index) if suite == "MWest" else None
    lmc_idx = legacy_idx
    if legacy_idx is not None:
        if hasattr(symlib, "read_subhalos"):
            legacy_h, legacy_hist = symlib.read_subhalos(sim_dir)
            if not (0 < legacy_idx < len(legacy_h)):
                raise ValueError(f"Invalid legacy LMC index {legacy_idx} for {sim_dir}")
            lmc_idx = _mapped_lmc_index(legacy_hist, rh, legacy_idx)
        if not (0 < lmc_idx < len(r)):
            raise ValueError(f"Invalid mapped LMC index {lmc_idx} for {sim_dir}")

    return Catalogs(
        host_index=host_index, host_name=Path(sim_dir).name, sim_dir=str(sim_dir),
        scale=scale, rockstar=r, symfind=s, rockstar_hist=rh,
        symfind_hist=sh, lmc_idx=lmc_idx, legacy_lmc_idx=legacy_idx,
    )


def pre_infall_snapshot(r, lmc_idx, mode="crossing"):
    """MW–LMC last pre-infall snapshot, defined using Rockstar Rvir.

    crossing: one snapshot before the FIRST entry inside Rvir,MW.
    outside_both: last snapshot with d > Rvir,MW + Rvir,LMC.
    """
    d = np.linalg.norm(r["x"][lmc_idx] - r["x"][0], axis=1)
    host_r = r["rvir"][0]
    lmc_r = r["rvir"][lmc_idx]
    good = (r["ok"][0] & r["ok"][lmc_idx] &
            np.isfinite(d) & np.isfinite(host_r) & (host_r > 0))
    if mode == "crossing":
        indices = np.flatnonzero(good & (d < host_r))
        return None if len(indices) == 0 or indices[0] == 0 else int(indices[0] - 1)
    if mode == "outside_both":
        good &= np.isfinite(lmc_r) & (lmc_r > 0)
        indices = np.flatnonzero(good & (d > host_r + lmc_r))
        return None if len(indices) == 0 else int(indices[-1])
    raise ValueError("mode must be 'crossing' or 'outside_both'")


def membership_mask(halos, rockstar, lmc_idx, i_pre, *,
                    mode="ever_before_infall", radius_factor=1.0):
    """Historically LMC-associated branches for one finder.

    The LMC boundary is consistently taken from Rockstar Rvir,LMC;
    positions and existence flags are taken from the requested finder.
    """
    if i_pre is None or not (0 <= i_pre < halos.shape[1]):
        raise ValueError("No valid pre-infall snapshot")
    if radius_factor <= 0:
        raise ValueError("radius_factor must be positive")
    sl = slice(0, i_pre + 1)
    dx = halos["x"][:, sl] - halos["x"][lmc_idx, sl][None, :, :]
    distance = np.linalg.norm(dx, axis=-1)
    rvir_lmc = rockstar["rvir"][lmc_idx, sl]
    valid = (halos["ok"][:, sl] & halos["ok"][lmc_idx, sl][None, :] &
             rockstar["ok"][lmc_idx, sl][None, :] &
             np.isfinite(distance) & np.isfinite(rvir_lmc)[None, :] &
             (rvir_lmc[None, :] > 0))
    inside = valid & (distance < radius_factor * rvir_lmc[None, :])
    if mode == "ever_before_infall":
        selected = inside.any(axis=1)
    elif mode == "at_pre_infall":
        selected = inside[:, -1].copy()
    else:
        raise ValueError("mode must be 'ever_before_infall' or 'at_pre_infall'")
    selected[0] = False  # MW host itself
    selected[lmc_idx] = False  # LMC analog itself
    return selected


def classify(catalogs, *, infall_mode="crossing",
             membership_mode="ever_before_infall", radius_factor=1.0):
    """Return (i_pre, rockstar_mask, symfind_mask) for an MWest host."""
    if catalogs.lmc_idx is None:
        raise ValueError("Only MWest hosts have an identified LMC analog")
    i_pre = pre_infall_snapshot(catalogs.rockstar, catalogs.lmc_idx, infall_mode)
    if i_pre is None:
        raise ValueError(f"No pre-infall snapshot for {catalogs.host_name}")
    kwargs = dict(lmc_idx=catalogs.lmc_idx, i_pre=i_pre,
                  mode=membership_mode, radius_factor=radius_factor)
    rock_mask = membership_mask(catalogs.rockstar, catalogs.rockstar, **kwargs)
    sym_mask = membership_mask(catalogs.symfind, catalogs.rockstar, **kwargs)
    return i_pre, rock_mask, sym_mask


def membership_table(catalogs, i_pre, rock_mask, sym_mask):
    """One row per associated branch, including finder-specific flags."""
    import pandas as pd
    indices = np.flatnonzero(rock_mask | sym_mask)
    return pd.DataFrame({
        "host_index": catalogs.host_index,
        "host_name": catalogs.host_name,
        "sub_idx": indices,
        "lmc_idx": catalogs.lmc_idx,
        "i_pre": i_pre,
        "a_pre": float(catalogs.scale[i_pre]),
        "associated_rockstar": rock_mask[indices],
        "associated_symfind": sym_mask[indices],
        "ok_z0_rockstar": catalogs.rockstar["ok"][indices, -1],
        "ok_z0_symfind": catalogs.symfind["ok"][indices, -1],
    })


def raw_hlist_preinfall_selection(df, raw_pre_snap):
    """Reconstruct pre-infall membership among supplied raw-HLIST nodes.

    A node is associated if within Rvir,LMC at this epoch, or if one of its
    supplied progenitor nodes was associated and points to this halo_id via
    desc_id. Peaks use only progenitor masses through the requested epoch.

    This cannot recover nodes omitted from an already-filtered input CSV.
    Pass the full raw-HLIST selection file for the most complete result.
    """
    import pandas as pd

    needed = {"raw_snapshot", "a", "halo_id", "desc_id", "Mvir", "inside_Rvir_LMC"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing raw-HLIST fields: {needed - set(df.columns)}")
    d = df.dropna(subset=["raw_snapshot", "a", "halo_id"]).copy()
    d = d.sort_values(["a", "raw_snapshot"])
    d = d.drop_duplicates(["raw_snapshot", "halo_id"], keep="last")
    d["inside"] = d["inside_Rvir_LMC"].astype(str).str.strip().str.lower().isin(
        ["true", "t", "1", "yes"]
    )
    d["mvir_numeric"] = pd.to_numeric(d["Mvir"], errors="coerce")

    # Integer parsing preserves exact 64-bit tree IDs provided as CSV strings.
    def key(value):
        if pd.isna(value):
            return None
        value = str(value).strip()
        if value.endswith(".0"):
            value = value[:-2]
        try:
            return int(value)
        except ValueError:
            return None

    next_assoc = set()
    next_peak = {}
    epochs = d[["raw_snapshot", "a"]].drop_duplicates("raw_snapshot").sort_values("a")
    for epoch in epochs.itertuples(index=False):
        block = d[d["raw_snapshot"] == epoch.raw_snapshot].copy()
        members = []
        peaks = []
        inherited = set()
        inherited_mass = {}
        for row in block.itertuples(index=False):
            hid, did = key(row.halo_id), key(row.desc_id)
            assoc = bool(row.inside) or (hid is not None and hid in next_assoc)
            earlier = next_peak.get(hid, np.nan)
            usable = [x for x in (row.mvir_numeric, earlier) if np.isfinite(x) and x > 0]
            peak = max(usable) if usable else np.nan
            members.append(assoc)
            peaks.append(peak)
            if did is not None and did >= 0:
                if assoc:
                    inherited.add(did)
                if np.isfinite(peak):
                    inherited_mass[did] = max(inherited_mass.get(did, 0), peak)
        block["causal_assoc"] = members
        block["Mpeak_upto_infall"] = peaks
        if int(epoch.raw_snapshot) == int(raw_pre_snap):
            return block
        next_assoc, next_peak = inherited, inherited_mass

    raise ValueError(f"raw_pre_snap={raw_pre_snap} is absent from the supplied raw data")
