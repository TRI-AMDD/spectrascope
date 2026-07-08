"""Tests for the feature-level region-masking helpers in overtone_analysis.py."""

import os
import sys

import numpy as np
import pytest

# overtone_analysis.py lives in examples/, not on the package path.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples"))
)
import overtone_analysis as oa  # noqa: E402


# --------------------------------------------------------------------------- #
# pointwise_feature_wls
# --------------------------------------------------------------------------- #
def test_pointwise_feature_wls_returns_axis_and_no_globals():
    xrange = np.linspace(400, 1300, 50)
    col_wls, is_global = oa.pointwise_feature_wls(xrange)

    assert col_wls.shape == (50,)
    assert is_global.shape == (50,)
    assert np.array_equal(col_wls, xrange)
    assert not is_global.any()


# --------------------------------------------------------------------------- #
# polynomial_feature_wls
# --------------------------------------------------------------------------- #
def test_polynomial_feature_wls_length_matches_polyfeatures_all():
    xaxis = np.linspace(400, 1300, 901)
    Narr = (4, 5, 10, 20)
    degree = 3
    col_wls, is_global = oa.polynomial_feature_wls(xaxis, Narr=Narr, degree=degree)

    expected = (degree + 1) * sum(Narr) + 1
    assert col_wls.shape == (expected,)
    assert is_global.shape == (expected,)


def test_polynomial_feature_wls_only_last_column_is_global():
    xaxis = np.linspace(400, 1300, 901)
    col_wls, is_global = oa.polynomial_feature_wls(xaxis)

    assert is_global[-1]
    assert not is_global[:-1].any()


def test_polynomial_feature_wls_bin_centers_within_axis_range():
    xaxis = np.linspace(400, 1300, 901)
    col_wls, is_global = oa.polynomial_feature_wls(xaxis, Narr=(4,), degree=3)

    # First N=4 block: 4 bins of width 225, centers at 512.5, 737.5, 962.5, 1187.5.
    # degree+1 = 4 columns per bin, all sharing the bin center.
    block = col_wls[:16]  # 4 bins * 4 coeffs, before the global column
    centers = np.array([512.5, 737.5, 962.5, 1187.5])
    expected = np.repeat(centers, 4)
    assert np.allclose(block, expected)


# --------------------------------------------------------------------------- #
# mask_features
# --------------------------------------------------------------------------- #
def _simple_setup():
    # 5 columns, wavelengths 400..800 step 100, no globals.
    F = np.arange(15, dtype=float).reshape(3, 5)
    col_wls = np.array([400.0, 500.0, 600.0, 700.0, 800.0])
    is_global = np.zeros(5, dtype=bool)
    return F, col_wls, is_global


def test_mask_features_only_drop_keeps_in_band_columns():
    F, col_wls, is_global = _simple_setup()
    out = oa.mask_features(F, col_wls, is_global, 500, 700, "only", "drop")
    # in-band: 500,600,700 -> columns 1,2,3
    assert out.shape == (3, 3)
    assert np.array_equal(out, F[:, [1, 2, 3]])


def test_mask_features_exclude_drop_keeps_out_of_band_columns():
    F, col_wls, is_global = _simple_setup()
    out = oa.mask_features(F, col_wls, is_global, 500, 700, "exclude", "drop")
    # out-of-band: 400,800 -> columns 0,4
    assert out.shape == (3, 2)
    assert np.array_equal(out, F[:, [0, 4]])


def test_mask_features_noise_preserves_shape_and_keeps_kept_columns():
    F, col_wls, is_global = _simple_setup()
    rng = np.random.default_rng(0)
    out = oa.mask_features(F, col_wls, is_global, 500, 700, "only", "noise", rng=rng)
    assert out.shape == F.shape
    # kept (in-band) columns unchanged
    assert np.array_equal(out[:, [1, 2, 3]], F[:, [1, 2, 3]])
    # obscured columns changed
    assert not np.array_equal(out[:, [0, 4]], F[:, [0, 4]])


def test_mask_features_global_column_always_kept():
    F = np.arange(12, dtype=float).reshape(3, 4)
    col_wls = np.array([400.0, 500.0, 600.0, np.nan])
    is_global = np.array([False, False, False, True])
    rng = np.random.default_rng(0)

    # 'only' on a band that excludes everything except the global column
    out_drop = oa.mask_features(F, col_wls, is_global, 1000, 1100, "only", "drop")
    # no real feature in band, but global column kept
    assert np.array_equal(out_drop, F[:, [3]])

    out_noise = oa.mask_features(
        F, col_wls, is_global, 1000, 1100, "exclude", "noise", rng=rng
    )
    # global column never noised, in either mode
    assert np.array_equal(out_noise[:, 3], F[:, 3])


# --------------------------------------------------------------------------- #
# fit_feature_matrix
# --------------------------------------------------------------------------- #
def test_fit_feature_matrix_rf_returns_finite_rmse():
    rng = np.random.default_rng(0)
    n = 60
    F = rng.standard_normal((n, 8))
    # learnable target: linear combination of two columns plus small noise
    br = 3.0 * F[:, 0] - 2.0 * F[:, 1] + 0.1 * rng.standard_normal(n) + 10.0

    rmse = oa.fit_feature_matrix(F, br, "pointwise_features", "RF")

    assert isinstance(rmse, float)
    assert np.isfinite(rmse)
    assert rmse >= 0


# --------------------------------------------------------------------------- #
# run_sweep_features (smoke)
# --------------------------------------------------------------------------- #
def test_run_sweep_features_smoke(tmp_path):
    X, xaxis, br = oa.load_grapes_nir()
    df = oa.run_sweep_features(
        X, xaxis, br,
        slices=[(700, 800)], models=["RF"], feats=["pointwise_features"],
    )

    assert set(df.columns) == {
        "band", "lo", "hi", "mode", "mechanism", "model", "feat", "rmse"
    }
    # baseline + noise_baseline + (1 band x 2 modes x 2 mechanisms) = 4 fitted rows
    assert (df["mode"] == "baseline").sum() == 1
    assert (df["mode"] == "noise_baseline").sum() == 1
    assert {"exclude", "only"} <= set(df["mode"])
    assert {"drop", "noise"} <= set(df["mechanism"])
    assert np.isfinite(df["rmse"]).all()


def _synthetic_feature_frame():
    """A full-shaped feature-sweep frame (all MODELS x FEATS x modes x mechs)."""
    rng = np.random.default_rng(0)
    rows = []
    for model in oa.MODELS:
        for feat in oa.FEATS:
            rows.append(dict(band="baseline", lo=np.nan, hi=np.nan,
                             mode="baseline", mechanism="none", model=model,
                             feat=feat, rmse=1.0))
            rows.append(dict(band="noise_baseline", lo=np.nan, hi=np.nan,
                             mode="noise_baseline", mechanism="noise",
                             model=model, feat=feat, rmse=2.0))
            for lo, hi in [(700, 800), (800, 900)]:
                for mode in ("exclude", "only"):
                    for mech in ("drop", "noise"):
                        rows.append(dict(
                            band=f"{lo}-{hi}", lo=lo, hi=hi, mode=mode,
                            mechanism=mech, model=model, feat=feat,
                            rmse=1.0 + rng.random()))
    import pandas as pd
    return pd.DataFrame(rows)


def test_plot_delta_prmse_features_writes_per_mechanism_figures(tmp_path):
    df = _synthetic_feature_frame()
    br = np.linspace(1, 25, 50)
    for mechanism in ("drop", "noise"):
        oa.plot_delta_prmse_features(df, br, str(tmp_path), mechanism)
        assert (tmp_path / f"Grapes_Mask_features_{mechanism}.pdf").exists()
        assert (tmp_path / f"Grapes_Mask_features_{mechanism}.svg").exists()
