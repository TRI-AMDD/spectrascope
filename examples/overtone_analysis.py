"""Grape NIR spectral-region masking analysis.

Probes which wavelength regions carry the °Brix (sugar) signal, using
fixed-parameter Random Forest and LCEN models on two featurizations (pointwise
raw-intensity and polynomial-fit features). Masked regions are replaced by
N(0, 1) noise (spectrum dimensionality preserved). For each band, two
complementary experiments are run:

* ``exclude`` -- the band is masked, the rest of the spectrum is kept.
* ``only``    -- only the band is kept, the rest of the spectrum is masked.

Run as a script to produce a single 2x2 Δ(%RMSE)-vs-region figure (rows: RF,
LCEN; columns: pointwise, polynomial), with both experiments overlaid per panel,
plus a CSV of all results under ``examples/Figures``.
"""

import os
import sys

# Ensure the repo root (parent of examples/) is importable when run as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# SpectraScope is imported lazily inside fit_rf so the pure-numpy masking helpers
# (and their tests) do not require the full ML stack.

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "tests", "test_files", "grapes_example"
)

# (lo, hi) wavelength bands, processed one at a time -- inclusive on both ends.
# Active slicing: coarse 100-nm bins.
SLICES = [
    (400, 500), (500, 600), (600, 700), (700, 800), (800, 900),
    (900, 1000), (1000, 1100), (1100, 1200), (1200, 1300),
]

# Previous fine slicing -- uncomment (and comment the block above) to toggle.
# SLICES = [
#     (401, 443), (444, 489), (490, 519), (520, 539), (540, 606),
#     (607, 622), (623, 652), (653, 666), (667, 680), (681, 689),
#     (690, 734), (735, 773), (774, 812), (813, 879), (880, 965),
#     (966, 991), (992, 1107), (1108, 1165), (1166, 1166), (1167, 1229),
#     (1230, 1299),
# ]

# Featurizations swept for every band (and friendly labels for plot titles).
FEATS = ["pointwise_features", "polynomial_featuresall"]
FEAT_LABELS = {"pointwise_features": "Pointwise", "polynomial_featuresall": "Polynomial"}

# Fixed RF hyperparameters per featurization (no grid search).
RF_PARAMS = {
    "pointwise_features": dict(
        random_state=42, n_estimators=200, criterion="squared_error", max_depth=50
    ),
    "polynomial_featuresall": dict(
        random_state=42, n_estimators=50, criterion="squared_error", max_depth=50
    ),
}

# Fixed LCEN hyperparameters per featurization (no grid search: cv_method=None,
# single-value lists). Alpha/l1_ratio are the values previously found for NIR.
_LCEN_BASE = dict(
    l1_ratio=[0.001], cv_method=None, degree=[1], lag=[0], LCEN_cutoff=5e-5,
    trans_type="poly", LCEN_interaction=False, scale_X=True, scale_y=True,
    metric=root_mean_squared_error,
)
LCEN_PARAMS = {
    "pointwise_features": dict(_LCEN_BASE, alpha=[0.0006]),
    "polynomial_featuresall": dict(_LCEN_BASE, alpha=[0.08414]),
}

# Models swept for every band.
MODELS = ["RF", "LCEN"]


# --------------------------------------------------------------------------- #
# Masking helpers (pure numpy, unit-tested)
# --------------------------------------------------------------------------- #
def band_keep_mask(wls, lo, hi):
    """Boolean mask, True = keep (wavelength NOT in [lo, hi] inclusive)."""
    return ~((wls >= lo) & (wls <= hi))


def mask_noise(X, keep, rng=None):
    """Replace excluded columns with N(0, 1) noise; preserve shape."""
    rng = np.random.default_rng() if rng is None else rng
    out = X.copy()
    n_excluded = int((~keep).sum())
    out[:, ~keep] = rng.standard_normal((X.shape[0], n_excluded))
    return out


# --------------------------------------------------------------------------- #
# Feature-level masking helpers (pure numpy, unit-tested)
#
# These operate on a *baseline* feature matrix generated once from the full
# spectrum, rather than re-featurizing a noise-masked spectrum. Each column is
# tagged with the wavelength it represents (``col_wls``) plus a flag marking
# columns that are not tied to any single band (``is_global``, e.g. the
# polynomial max-position column), which are always kept.
# --------------------------------------------------------------------------- #
def pointwise_feature_wls(xrange):
    """Per-column wavelength tags for pointwise features.

    Each interpolated point is its own feature, so the column wavelengths are
    the interpolation axis itself and no column is global.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(col_wls, is_global)``.
    """
    col_wls = np.asarray(xrange, dtype=float)
    is_global = np.zeros(col_wls.shape[0], dtype=bool)
    return col_wls, is_global


def polynomial_feature_wls(xaxis, Narr=(4, 5, 10, 20), degree=3):
    """Per-column wavelength tags for ``polynomial_featuresall`` output.

    Replicates the column order of ``polyfeatures_all``: for each ``N`` in
    ``Narr``, the spectrum's wavelength range is split into ``N`` equal-width
    bins, and each bin contributes ``degree + 1`` polynomial-coefficient columns.
    Every column of a bin is tagged with that bin's center wavelength (the
    nominal equal-width midpoint, matching ``generate_bins``). A final global
    column (wavelength of max intensity) is appended and flagged.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(col_wls, is_global)`` of length ``(degree + 1) * sum(Narr) + 1``.
    """
    lo, hi = float(np.min(xaxis)), float(np.max(xaxis))
    centers = []
    for N in Narr:
        binwidth = (hi - lo) / N
        bin_centers = lo + (np.arange(N) + 0.5) * binwidth
        centers.append(np.repeat(bin_centers, degree + 1))
    col_wls = np.concatenate(centers + [np.array([np.nan])])
    is_global = np.zeros(col_wls.shape[0], dtype=bool)
    is_global[-1] = True
    return col_wls, is_global


def mask_features(F, col_wls, is_global, lo, hi, mode, mechanism, rng=None):
    """Mask a baseline feature matrix by wavelength band.

    A column is "in band" when its tagged wavelength falls in ``[lo, hi]``
    (inclusive) and it is not global. ``mode="only"`` keeps the in-band columns;
    ``mode="exclude"`` keeps the out-of-band columns. Global columns are always
    kept regardless of mode.

    ``mechanism="drop"`` returns the kept columns only (reduced width);
    ``mechanism="noise"`` preserves shape, overwriting obscured columns with
    N(0, 1) noise.
    """
    col_wls = np.asarray(col_wls, dtype=float)
    in_band = (col_wls >= lo) & (col_wls <= hi) & ~is_global

    if mode == "only":
        keep = in_band | is_global
    elif mode == "exclude":
        keep = ~in_band  # global columns have in_band False, so kept here too
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    if mechanism == "drop":
        return F[:, keep]
    elif mechanism == "noise":
        rng = np.random.default_rng() if rng is None else rng
        out = F.copy()
        n_obscured = int((~keep).sum())
        out[:, ~keep] = rng.standard_normal((F.shape[0], n_obscured))
        return out
    else:
        raise ValueError(f"unknown mechanism: {mechanism!r}")


# --------------------------------------------------------------------------- #
# Data prep
# --------------------------------------------------------------------------- #
def load_grapes_nir():
    """Load and preprocess NIR grape spectra, restricted to 400-1300 nm.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (scaled spectra, wavelength axis, br targets)
    """
    XSF = np.load(os.path.join(DATA_DIR, "XSF_br.npy"))
    br = np.load(os.path.join(DATA_DIR, "br.npy"))
    nir_wls = np.load(os.path.join(DATA_DIR, "nir_xaxis.npy"))

    centered = XSF - XSF.mean(axis=1, keepdims=True)
    scaled = StandardScaler().fit_transform(centered)

    sel = (nir_wls > 400) & (nir_wls < 1300)
    return scaled[:, sel], nir_wls[sel], br


# --------------------------------------------------------------------------- #
# Model fitting
# --------------------------------------------------------------------------- #
def fit_model(X, xaxis, br, feat, model):
    """Fit one fixed-param model (``"RF"`` or ``"LCEN"``) on featurization
    ``feat``; return test RMSE or NaN.

    A non-converging or otherwise failing fit is logged and recorded as NaN so a
    single bad band does not crash the sweep.
    """
    from spectrascope.SpectraScope import SpectraScope

    try:
        ss = SpectraScope(
            [X], y=br, specnames=["nir"], yname="br",
            random_seed=42, save_path="output/",
        )
        ss.add_x_axis("nir", xaxis)
        if feat == "pointwise_features":
            ss.add_spectral_feature("nir", feat, xrange=xaxis)
        else:
            ss.add_spectral_feature("nir", feat)

        attr = f"nir_{feat}"
        if model == "RF":
            ss.do_sk(
                attr, "br", 0.25, use_pipeline=False, scale_X=False,
                estimator=RandomForestRegressor(**RF_PARAMS[feat]),
                metric=root_mean_squared_error,
            )
            return getattr(ss, f"{attr}_br_RandomForestRegressor")["score"]
        else:
            ss.do_LCEN(attr, "br", 0.25, **LCEN_PARAMS[feat])
            return getattr(ss, f"{attr}_br_LCEN")["score"]
    except Exception as exc:
        print(f"[WARN] {model} {feat} fit failed: {exc}")
        return np.nan


def baseline_feature_matrix(X, xaxis, feat):
    """Generate the full (unmasked) feature matrix for ``feat`` once.

    Uses the same ``add_spectral_feature`` calls as ``fit_model`` so the matrix
    matches the spectrum-level baseline exactly. Returns ``(F, col_wls,
    is_global)``: the feature matrix plus its per-column wavelength tags.
    """
    from spectrascope.SpectraScope import SpectraScope

    ss = SpectraScope(
        [X], y=None, specnames=["nir"], yname="br",
        random_seed=42, save_path="output/",
    )
    ss.add_x_axis("nir", xaxis)
    if feat == "pointwise_features":
        ss.add_spectral_feature("nir", feat, xrange=xaxis)
        col_wls, is_global = pointwise_feature_wls(xaxis)
    else:
        ss.add_spectral_feature("nir", feat)
        col_wls, is_global = polynomial_feature_wls(xaxis)
    F = getattr(ss, f"nir_{feat}")
    return F, col_wls, is_global


def fit_feature_matrix(F, br, feat, model):
    """Fit one fixed-param model (``"RF"`` or ``"LCEN"``) on a precomputed
    feature matrix ``F``; return test RMSE or NaN.

    Mirrors ``fit_model`` but skips featurization: ``F`` is set directly as the
    feature attribute. Uses the same fixed hyperparameters, ``random_seed`` and
    test split, so results are directly comparable to ``fit_model``'s baseline.
    """
    from spectrascope.SpectraScope import SpectraScope

    try:
        ss = SpectraScope(
            [F], y=br, specnames=["nir"], yname="br",
            random_seed=42, save_path="output/",
        )
        attr = "feat_matrix"
        setattr(ss, attr, F)
        ss.add_generated_feature(attr)
        if model == "RF":
            ss.do_sk(
                attr, "br", 0.25, use_pipeline=False, scale_X=False,
                estimator=RandomForestRegressor(**RF_PARAMS[feat]),
                metric=root_mean_squared_error,
            )
            return float(getattr(ss, f"{attr}_br_RandomForestRegressor")["score"])
        else:
            ss.do_LCEN(attr, "br", 0.25, **LCEN_PARAMS[feat])
            return float(getattr(ss, f"{attr}_br_LCEN")["score"])
    except Exception as exc:
        print(f"[WARN] {model} {feat} feature-fit failed: {exc}")
        return np.nan


def run_sweep(X, xaxis, br):
    """Sweep baseline + every band under both modes (noise-fill masking).

    Modes: ``exclude`` (band masked, rest kept) and ``only`` (band kept, rest
    masked). Masked columns are always replaced by N(0, 1) noise, so the
    spectrum keeps full dimensionality.

    Returns
    -------
    pd.DataFrame
        Long-format results with columns
        ``band, lo, hi, mode, model, feat, rmse``.
    """
    rng = np.random.default_rng(42)
    rows = []

    # baseline: full spectrum, no mask (shared reference per model/featurization)
    for model in MODELS:
        for feat in FEATS:
            rows.append(dict(
                band="baseline", lo=np.nan, hi=np.nan, mode="baseline",
                model=model, feat=feat, rmse=fit_model(X, xaxis, br, feat, model),
            ))

    # random-noise baseline: entire spectrum replaced by noise (no real signal),
    # i.e. the worst case masking can approach. One shared noise realization.
    X_noise = mask_noise(X, np.zeros(xaxis.shape[0], dtype=bool), rng=rng)
    for model in MODELS:
        for feat in FEATS:
            rows.append(dict(
                band="noise_baseline", lo=np.nan, hi=np.nan, mode="noise_baseline",
                model=model, feat=feat, rmse=fit_model(X_noise, xaxis, br, feat, model),
            ))

    for lo, hi in SLICES:
        exclude_keep = band_keep_mask(xaxis, lo, hi)  # True = keep (NOT in band)
        only_keep = ~exclude_keep                      # True = keep (IN band)
        for mode, keep in (("exclude", exclude_keep), ("only", only_keep)):
            Xm = mask_noise(X, keep, rng=rng)
            for model in MODELS:
                for feat in FEATS:
                    rows.append(dict(
                        band=f"{lo}-{hi}", lo=lo, hi=hi, mode=mode,
                        model=model, feat=feat,
                        rmse=fit_model(Xm, xaxis, br, feat, model),
                    ))
        print(f"band {lo}-{hi} done")
    return pd.DataFrame(rows)


def run_sweep_features(X, xaxis, br, slices=SLICES, models=MODELS, feats=FEATS):
    """Feature-level region sweep on baseline features (no re-featurization).

    Each featurization is generated **once** from the full spectrum; every band
    is then probed by slicing that baseline matrix by wavelength. Two modes
    (``exclude`` = obscure the band, keep the rest; ``only`` = keep the band,
    obscure the rest) are each run under two mechanisms (``drop`` = subset the
    matrix to the kept columns; ``noise`` = overwrite obscured columns with
    N(0, 1), preserving width). Global columns (e.g. the polynomial
    max-position) are always kept.

    Returns
    -------
    pd.DataFrame
        Long-format results with columns
        ``band, lo, hi, mode, mechanism, model, feat, rmse``.
    """
    rng = np.random.default_rng(42)
    rows = []

    # Generate each baseline feature matrix (+ its wavelength tags) once.
    baselines = {feat: baseline_feature_matrix(X, xaxis, feat) for feat in feats}

    # baseline: full feature matrix, no masking (shared reference).
    for feat in feats:
        F, _, _ = baselines[feat]
        for model in models:
            rows.append(dict(
                band="baseline", lo=np.nan, hi=np.nan, mode="baseline",
                mechanism="none", model=model, feat=feat,
                rmse=fit_feature_matrix(F, br, feat, model),
            ))

    # noise baseline: all non-global columns noised (noise mechanism only --
    # an all-columns-dropped matrix has no features to fit, so drop is omitted).
    for feat in feats:
        F, col_wls, is_global = baselines[feat]
        keep = is_global.copy()
        Fn = F.copy()
        Fn[:, ~keep] = rng.standard_normal((F.shape[0], int((~keep).sum())))
        for model in models:
            rows.append(dict(
                band="noise_baseline", lo=np.nan, hi=np.nan,
                mode="noise_baseline", mechanism="noise", model=model, feat=feat,
                rmse=fit_feature_matrix(Fn, br, feat, model),
            ))

    for lo, hi in slices:
        for feat in feats:
            F, col_wls, is_global = baselines[feat]
            for mode in ("exclude", "only"):
                for mechanism in ("drop", "noise"):
                    Fm = mask_features(
                        F, col_wls, is_global, lo, hi, mode, mechanism, rng=rng
                    )
                    for model in models:
                        rows.append(dict(
                            band=f"{lo}-{hi}", lo=lo, hi=hi, mode=mode,
                            mechanism=mechanism, model=model, feat=feat,
                            rmse=fit_feature_matrix(Fm, br, feat, model),
                        ))
        print(f"band {lo}-{hi} done (feature-level)")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_delta_prmse(df, br, outdir):
    """Single 2x2 Δ(%RMSE)-vs-region figure.

    Panels: rows are models (RF, LCEN), columns are featurizations (pointwise,
    polynomial) -- i.e. top-left RF/pointwise, top-right RF/polynomial,
    bottom-left LCEN/pointwise, bottom-right LCEN/polynomial. Within each panel
    both experiments are overlaid as scatter-with-connecting-lines:
    ``exclude`` (region masked) and ``only`` (only the region kept), where
    ``%RMSE = RMSE / max(br) * 100`` and Δ is measured against the unmasked
    baseline. A zero line marks the baseline; positive values mean prediction is
    worse than using the full spectrum.
    """
    os.makedirs(outdir, exist_ok=True)
    max_br = float(np.max(br))
    mode_styles = [
        ("exclude", "Region masked", "tab:blue"),
        ("only", "Region only", "tab:orange"),
    ]

    fig, axes = plt.subplots(len(MODELS), len(FEATS), figsize=(14, 9))
    for r, model in enumerate(MODELS):
        for c, feat in enumerate(FEATS):
            ax = axes[r, c]
            baseline = df[
                (df["model"] == model) & (df["feat"] == feat)
                & (df["mode"] == "baseline")
            ]["rmse"].iloc[0]
            for mode, label, color in mode_styles:
                msub = df[
                    (df["model"] == model) & (df["feat"] == feat)
                    & (df["mode"] == mode)
                ].sort_values("lo")
                bands = msub["band"].tolist()
                delta = (msub["rmse"].to_numpy() - baseline) / max_br * 100
                x = np.arange(len(bands))
                ax.plot(x, delta, marker="o", ls="-", color=color, label=label)
            ax.axhline(0, color="k", ls="--", lw=1)

            noise_base = df[
                (df["model"] == model) & (df["feat"] == feat)
                & (df["mode"] == "noise_baseline")
            ]["rmse"].iloc[0]
            delta_noise = (noise_base - baseline) / max_br * 100
            ax.axhline(delta_noise, color="red", ls=":", lw=1.5,
                       label="Random noise baseline")

            ax.set_xticks(x)
            ax.set_xticklabels(bands, rotation=90)
            ax.set_xlabel("Region (nm)")
            ax.set_ylabel(r"$\Delta$(%RMSE) from baseline")
            ax.set_title(f"{model} -- {FEAT_LABELS[feat]}")
            ax.legend()

            # Panel labels a., b., c., d. (left-to-right, top-to-bottom).
            panel = "abcd"[r * len(FEATS) + c]
            ax.text(-0.08, 1.08, f"{panel}.", transform=ax.transAxes,
                    fontsize=15, fontweight="bold", va="top", ha="right")

    fig.suptitle("Region importance via noise masking")
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(outdir, f"Grapes_Mask.{ext}"))
    plt.close(fig)


def plot_delta_prmse_features(df, br, outdir, mechanism):
    """2x2 Δ(%RMSE)-vs-region figure for one feature-level masking mechanism.

    Same panel layout as :func:`plot_delta_prmse` (rows: RF, LCEN; columns:
    pointwise, polynomial). Each panel overlays the two ``mode`` series for the
    given ``mechanism`` (``"drop"`` or ``"noise"``): ``exclude`` (region masked,
    blue) and ``only`` (region only, orange). Δ is measured against the unmasked
    baseline and the red dotted line is the all-noise baseline. ``drop`` and
    ``noise`` are plotted to separate files (``Grapes_Mask_features_{mechanism}``)
    rather than overlaid in one panel.
    """
    os.makedirs(outdir, exist_ok=True)
    max_br = float(np.max(br))
    mode_styles = [
        ("exclude", "Region masked", "tab:blue"),
        ("only", "Region only", "tab:orange"),
    ]

    fig, axes = plt.subplots(len(MODELS), len(FEATS), figsize=(14, 9))
    for r, model in enumerate(MODELS):
        for c, feat in enumerate(FEATS):
            ax = axes[r, c]
            baseline = df[
                (df["model"] == model) & (df["feat"] == feat)
                & (df["mode"] == "baseline")
            ]["rmse"].iloc[0]
            x = np.arange(0)
            for mode, label, color in mode_styles:
                msub = df[
                    (df["model"] == model) & (df["feat"] == feat)
                    & (df["mode"] == mode) & (df["mechanism"] == mechanism)
                ].sort_values("lo")
                bands = msub["band"].tolist()
                delta = (msub["rmse"].to_numpy() - baseline) / max_br * 100
                x = np.arange(len(bands))
                ax.plot(x, delta, marker="o", ls="-", color=color, label=label)
            ax.axhline(0, color="k", ls="--", lw=1)

            noise_base = df[
                (df["model"] == model) & (df["feat"] == feat)
                & (df["mode"] == "noise_baseline")
            ]["rmse"].iloc[0]
            delta_noise = (noise_base - baseline) / max_br * 100
            ax.axhline(delta_noise, color="red", ls=":", lw=1.5,
                       label="Random noise baseline")

            ax.set_xticks(x)
            ax.set_xticklabels(bands, rotation=90)
            ax.set_xlabel("Region (nm)")
            ax.set_ylabel(r"$\Delta$(%RMSE) from baseline")
            ax.set_title(f"{model} -- {FEAT_LABELS[feat]}")
            ax.legend()

            panel = "abcd"[r * len(FEATS) + c]
            ax.text(-0.08, 1.08, f"{panel}.", transform=ax.transAxes,
                    fontsize=15, fontweight="bold", va="top", ha="right")

    fig.suptitle(f"Region importance via {mechanism} masking")
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(outdir, f"Grapes_Mask_features_{mechanism}.{ext}"))
    plt.close(fig)


if __name__ == "__main__":
    X, xaxis, br = load_grapes_nir()
    outdir = os.path.join(os.path.dirname(__file__), "Figures")
    os.makedirs(outdir, exist_ok=True)

    # Spectrum-level masking (re-featurize a noise-masked spectrum).
    results = run_sweep(X, xaxis, br)
    results.to_csv(os.path.join(outdir, "grapes_masking.csv"), index=False)
    plot_delta_prmse(results, br, outdir)

    # Feature-level masking (slice the baseline feature matrix by wavelength).
    # drop and noise mechanisms are plotted to separate figures.
    results_feat = run_sweep_features(X, xaxis, br)
    results_feat.to_csv(os.path.join(outdir, "grapes_masking_features.csv"),
                        index=False)
    for mechanism in ("drop", "noise"):
        plot_delta_prmse_features(results_feat, br, outdir, mechanism)

    print(f"[DONE] outputs in {outdir}")
