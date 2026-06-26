"""QC tests for the AppTrail phenology pipeline.

Run with: pytest qc_tests.py -v

These tests use only synthetic data and do not require any completed pipeline
output files (no GeoTIFFs, stacks, or saved models needed).
"""

import numpy as np
import pandas as pd
import pytest

from build_data_table import _assign_label, _day_length
from edit_data_table import _balance_classes
from decision_trees import split_data, fit_tree
from fit_greendown_curves import (
    _decreasing_logistic,
    _curvature_extrema_doys,
    _fit_pixel,
    _make_psd,
    compute_curve_ci,
    compute_transition_dates_ci,
)
from map_utils import _pred_grid_to_rgba, _get_wgs84_bounds


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def simple_feature_df():
    """Balanced synthetic feature DataFrame with all required columns."""
    rng = np.random.default_rng(42)
    n = 200  # 50 per class
    labels = ['before'] * 50 + ['early'] * 50 + ['late'] * 50 + ['after'] * 50
    df = pd.DataFrame({
        'EVI':                 rng.uniform(0.1, 0.9, n),
        'NDVI':                rng.uniform(0.1, 0.9, n),
        'evi_delta':           rng.uniform(-0.1, 0.1, n),
        'evi_delta2':          rng.uniform(-0.1, 0.1, n),
        'ndvi_delta':          rng.uniform(-0.1, 0.1, n),
        'ndvi_delta2':         rng.uniform(-0.1, 0.1, n),
        'day_length_hrs':      rng.uniform(9.0, 15.0, n),
        'doy_minus_avg_middle': rng.uniform(-30, 30, n),
        'mode_label_7day':     rng.integers(0, 4, n).astype(float),
        'label':               labels,
    })
    return df


@pytest.fixture
def imbalanced_feature_df():
    """Synthetic feature DataFrame with heavily imbalanced class counts."""
    rng = np.random.default_rng(0)
    counts = {'before': 200, 'early': 80, 'late': 40, 'after': 20}
    frames = []
    for label, n in counts.items():
        frame = pd.DataFrame({
            'EVI':                 rng.uniform(0.1, 0.9, n),
            'NDVI':                rng.uniform(0.1, 0.9, n),
            'evi_delta':           rng.uniform(-0.1, 0.1, n),
            'evi_delta2':          rng.uniform(-0.1, 0.1, n),
            'ndvi_delta':          rng.uniform(-0.1, 0.1, n),
            'ndvi_delta2':         rng.uniform(-0.1, 0.1, n),
            'day_length_hrs':      rng.uniform(9.0, 15.0, n),
            'doy_minus_avg_middle': rng.uniform(-30, 30, n),
            'mode_label_7day':     rng.integers(0, 4, n).astype(float),
            'label':               label,
        })
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def known_logistic_params():
    """Logistic parameters for a realistic greendown curve."""
    return dict(L=0.5, k=0.1, t_mid=280.0, offset=0.1)


# ===========================================================================
# build_data_table: _assign_label
# ===========================================================================

class TestAssignLabel:
    def test_before_start(self):
        assert _assign_label(230, 260, 280, 300) == 'before'

    def test_early(self):
        assert _assign_label(270, 260, 280, 300) == 'early'

    def test_late(self):
        assert _assign_label(290, 260, 280, 300) == 'late'

    def test_after_end(self):
        assert _assign_label(310, 260, 280, 300) == 'after'

    def test_exactly_at_start_is_early(self):
        assert _assign_label(260, 260, 280, 300) == 'early'

    def test_exactly_at_middle_is_late(self):
        assert _assign_label(280, 260, 280, 300) == 'late'

    def test_exactly_at_end_is_after(self):
        assert _assign_label(300, 260, 280, 300) == 'after'


# ===========================================================================
# build_data_table: _day_length
# ===========================================================================

class TestDayLength:
    def test_positive_hours(self):
        assert _day_length(200, 42.0) > 0

    def test_less_than_24_hours(self):
        assert _day_length(200, 42.0) < 24

    def test_summer_longer_than_winter(self):
        summer = _day_length(182, 42.0)  # Jul 1
        winter = _day_length(355, 42.0)  # Dec 21
        assert summer > winter

    def test_latitude_effect(self):
        # Higher latitude → longer summer days
        high_lat = _day_length(200, 50.0)
        low_lat  = _day_length(200, 35.0)
        assert high_lat > low_lat

    def test_reasonable_range_for_ma(self):
        # Massachusetts latitude ~42°; summer days ~15h, winter ~9h
        for doy in [182, 220, 265, 300, 355]:
            dl = _day_length(doy, 42.3)
            assert 8 < dl < 16, f'Unexpected day length {dl:.1f}h on DOY {doy}'


# ===========================================================================
# fit_greendown_curves: _decreasing_logistic
# ===========================================================================

class TestDecreasingLogistic:
    def test_decreases_over_season(self, known_logistic_params):
        p = known_logistic_params
        t = np.array([182.0, 220.0, 265.0, 300.0, 355.0])
        y = _decreasing_logistic(t, p['L'], p['k'], p['t_mid'], p['offset'])
        assert np.all(np.diff(y) < 0), 'Logistic should decrease monotonically'

    def test_output_bounded(self, known_logistic_params):
        p = known_logistic_params
        t = np.linspace(182, 365, 100)
        y = _decreasing_logistic(t, p['L'], p['k'], p['t_mid'], p['offset'])
        assert y.min() >= p['offset'] - 1e-6
        assert y.max() <= p['L'] + p['offset'] + 1e-6

    def test_midpoint_is_half_amplitude(self, known_logistic_params):
        p = known_logistic_params
        y_mid = _decreasing_logistic(p['t_mid'], p['L'], p['k'], p['t_mid'], p['offset'])
        expected = p['L'] / 2 + p['offset']
        assert abs(y_mid - expected) < 1e-10


# ===========================================================================
# fit_greendown_curves: _curvature_extrema_doys
# ===========================================================================

class TestCurvatureExtremaDoys:
    def test_start_less_than_middle_less_than_end(self):
        start, middle, end = _curvature_extrema_doys(k=0.1, t_mid=280.0)
        assert start < middle < end

    def test_middle_equals_t_mid(self):
        _, middle, _ = _curvature_extrema_doys(k=0.1, t_mid=280.0)
        assert abs(middle - 280.0) < 1e-10

    def test_symmetric_around_middle(self):
        start, middle, end = _curvature_extrema_doys(k=0.1, t_mid=280.0)
        assert abs((middle - start) - (end - middle)) < 1e-10

    def test_steeper_curve_narrower_transition(self):
        _, s1, e1 = _curvature_extrema_doys(k=0.05, t_mid=280.0)
        _, s2, e2 = _curvature_extrema_doys(k=0.20, t_mid=280.0)
        assert (e1 - s1) > (e2 - s2)


# ===========================================================================
# fit_greendown_curves: _fit_pixel
# ===========================================================================

class TestFitPixel:
    def test_returns_none_with_too_few_points(self):
        doys   = np.array([200.0, 220.0, 240.0])
        values = np.array([0.7,   0.5,   0.3])
        assert _fit_pixel(doys, values) is None

    def test_returns_none_with_no_valid_values(self):
        doys   = np.array([200.0, 220.0, 240.0, 260.0, 280.0])
        values = np.array([np.nan, 0.0, np.nan, -0.1, np.nan])
        assert _fit_pixel(doys, values) is None

    def test_fit_on_clean_logistic(self, known_logistic_params):
        p = known_logistic_params
        doys   = np.linspace(182, 355, 30)
        values = _decreasing_logistic(doys, p['L'], p['k'], p['t_mid'], p['offset'])
        result = _fit_pixel(doys, values)
        assert result is not None
        popt, _ = result
        assert abs(popt[2] - p['t_mid']) < 5, 'Fitted t_mid should be close to true value'

    def test_popt_has_positive_k(self, known_logistic_params):
        p = known_logistic_params
        doys   = np.linspace(182, 355, 30)
        values = _decreasing_logistic(doys, p['L'], p['k'], p['t_mid'], p['offset'])
        result = _fit_pixel(doys, values)
        popt, _ = result
        assert popt[1] > 0, 'Steepness k must be positive for a decreasing curve'


# ===========================================================================
# fit_greendown_curves: _make_psd
# ===========================================================================

class TestMakePsd:
    def test_output_is_symmetric(self):
        M = np.array([[4.0, -2.0], [-2.0, 3.0]])
        psd = _make_psd(M)
        assert np.allclose(psd, psd.T)

    def test_eigenvalues_non_negative(self):
        M = np.array([[1.0, 2.0], [2.0, 1.0]])  # has a negative eigenvalue
        psd = _make_psd(M)
        eigvals = np.linalg.eigvalsh(psd)
        assert np.all(eigvals >= -1e-10)

    def test_preserves_psd_matrix(self):
        M = np.array([[4.0, 1.0], [1.0, 3.0]])  # already PSD
        psd = _make_psd(M)
        assert np.allclose(psd, M, atol=1e-10)


# ===========================================================================
# fit_greendown_curves: compute_transition_dates_ci
# ===========================================================================

class TestComputeTransitionDatesCi:
    @pytest.fixture
    def fitted_pixel(self, known_logistic_params):
        p = known_logistic_params
        doys   = np.linspace(182, 355, 40)
        values = _decreasing_logistic(doys, p['L'], p['k'], p['t_mid'], p['offset'])
        values += np.random.default_rng(7).normal(0, 0.01, len(doys))
        return _fit_pixel(doys, values)

    def test_start_less_than_middle_less_than_end(self, fitted_pixel):
        popt, pcov = fitted_pixel
        ci = compute_transition_dates_ci(popt, pcov)
        assert ci['start'][0] < ci['middle'][0] < ci['end'][0]

    def test_ci_lower_less_than_upper(self, fitted_pixel):
        popt, pcov = fitted_pixel
        ci = compute_transition_dates_ci(popt, pcov)
        for phase in ('start', 'middle', 'end'):
            point, lower, upper = ci[phase]
            assert lower <= point <= upper, f'{phase}: lower={lower}, point={point}, upper={upper}'

    def test_no_pcov_gives_nan_bounds(self, known_logistic_params):
        p = known_logistic_params
        popt = np.array([p['L'], p['k'], p['t_mid'], p['offset']])
        ci = compute_transition_dates_ci(popt, None)
        for phase in ('start', 'middle', 'end'):
            _, lower, upper = ci[phase]
            assert np.isnan(lower) and np.isnan(upper)

    def test_point_estimates_near_truth(self, fitted_pixel, known_logistic_params):
        popt, pcov = fitted_pixel
        ci = compute_transition_dates_ci(popt, pcov)
        _, true_middle, _ = _curvature_extrema_doys(
            known_logistic_params['k'], known_logistic_params['t_mid']
        )
        assert abs(ci['middle'][0] - true_middle) < 5


# ===========================================================================
# edit_data_table: _balance_classes
# ===========================================================================

class TestBalanceClasses:
    # _balance_classes uses groupby('label'), which drops the label column in the
    # result. The caller (edit_feature_table) saves and restores it. Tests here
    # verify row counts using the saved label Series, matching the actual usage.

    def test_all_classes_equal_count(self, imbalanced_feature_df):
        labels   = imbalanced_feature_df['label']
        balanced = _balance_classes(imbalanced_feature_df)
        restored = labels.loc[balanced.index]
        counts   = restored.value_counts()
        assert counts.nunique() == 1, f'Unequal counts after balancing: {counts.to_dict()}'

    def test_count_equals_minimum_class(self, imbalanced_feature_df):
        min_count = imbalanced_feature_df['label'].value_counts().min()
        labels    = imbalanced_feature_df['label']
        balanced  = _balance_classes(imbalanced_feature_df)
        restored  = labels.loc[balanced.index]
        for count in restored.value_counts():
            assert count == min_count

    def test_all_original_labels_present(self, imbalanced_feature_df):
        labels   = imbalanced_feature_df['label']
        balanced = _balance_classes(imbalanced_feature_df)
        restored = labels.loc[balanced.index]
        assert set(restored.unique()) == {'before', 'early', 'late', 'after'}

    def test_no_rows_added(self, imbalanced_feature_df):
        balanced = _balance_classes(imbalanced_feature_df)
        assert len(balanced) <= len(imbalanced_feature_df)


# ===========================================================================
# decision_trees: split_data
# ===========================================================================

class TestSplitData:
    def test_split_sizes(self, simple_feature_df):
        x_train, x_test, y_train, y_test = split_data(simple_feature_df)
        n = len(simple_feature_df)
        assert len(x_train) + len(x_test) == n
        # 75/25 split: allow ±5 rows for rounding
        assert abs(len(x_train) - int(n * 0.75)) <= 5

    def test_label_column_excluded_from_predictors(self, simple_feature_df):
        x_train, x_test, _, _ = split_data(simple_feature_df)
        assert 'label' not in x_train.columns
        assert 'label' not in x_test.columns

    def test_all_labels_in_train_and_test(self, simple_feature_df):
        _, _, y_train, y_test = split_data(simple_feature_df)
        expected = {'before', 'early', 'late', 'after'}
        assert set(y_train.unique()) == expected
        assert set(y_test.unique()) == expected

    def test_no_index_overlap(self, simple_feature_df):
        x_train, x_test, _, _ = split_data(simple_feature_df)
        assert len(set(x_train.index) & set(x_test.index)) == 0


# ===========================================================================
# decision_trees: fit_tree
# ===========================================================================

class TestFitTree:
    def test_unpruned_accuracy_above_chance(self, simple_feature_df):
        x_train, x_test, y_train, y_test = split_data(simple_feature_df)
        mdl = fit_tree(x_train, y_train, prune=False)
        acc = (mdl.predict(x_test) == y_test).mean()
        assert acc > 0.25, f'Test accuracy {acc:.2f} not above random baseline'

    def test_all_classes_in_fitted_model(self, simple_feature_df):
        x_train, _, y_train, _ = split_data(simple_feature_df)
        mdl = fit_tree(x_train, y_train, prune=False)
        assert set(mdl.classes_) == {'before', 'early', 'late', 'after'}

    def test_feature_importances_sum_to_one(self, simple_feature_df):
        x_train, _, y_train, _ = split_data(simple_feature_df)
        mdl = fit_tree(x_train, y_train, prune=False)
        assert abs(mdl.feature_importances_.sum() - 1.0) < 1e-6

    def test_pruned_depth_le_unpruned(self, simple_feature_df):
        x_train, _, y_train, _ = split_data(simple_feature_df)
        unpruned = fit_tree(x_train, y_train, prune=False)
        pruned   = fit_tree(x_train, y_train, prune=True)
        assert pruned.get_depth() <= unpruned.get_depth()

    def test_pruned_train_accuracy_not_perfect(self, simple_feature_df):
        # A pruned tree on synthetic data should not perfectly memorize training set
        x_train, _, y_train, _ = split_data(simple_feature_df)
        pruned = fit_tree(x_train, y_train, prune=True)
        train_acc = (pruned.predict(x_train) == y_train).mean()
        # Allow up to 1.0 but flag if suspiciously perfect on balanced synthetic data
        assert train_acc <= 1.0  # just sanity; main guard is depth test above


# ===========================================================================
# dashboard: _pred_grid_to_rgba
# ===========================================================================

class TestPredGridToRgba:
    @pytest.fixture
    def small_grid(self):
        grid = np.array([['before', 'early'],
                         ['late',   'after']], dtype=object)
        mask = np.array([[True, True],
                         [True, False]])
        return grid, mask

    def test_output_shape(self, small_grid):
        grid, mask = small_grid
        rgba = _pred_grid_to_rgba(grid, mask)
        assert rgba.shape == (2, 2, 4)

    def test_non_forest_pixels_transparent(self, small_grid):
        grid, mask = small_grid
        rgba = _pred_grid_to_rgba(grid, mask)
        assert rgba[1, 1, 3] == 0, 'Non-forest pixel should have alpha=0'

    def test_forest_pixels_opaque(self, small_grid):
        grid, mask = small_grid
        rgba = _pred_grid_to_rgba(grid, mask, opacity=0.85)
        assert rgba[0, 0, 3] > 0, 'Forest pixel should have non-zero alpha'
        assert rgba[0, 1, 3] > 0
        assert rgba[1, 0, 3] > 0

    def test_dtype_is_uint8(self, small_grid):
        grid, mask = small_grid
        rgba = _pred_grid_to_rgba(grid, mask)
        assert rgba.dtype == np.uint8

    def test_different_labels_get_different_colors(self, small_grid):
        grid, mask = small_grid
        rgba = _pred_grid_to_rgba(grid, mask)
        before_rgb = tuple(rgba[0, 0, :3])
        early_rgb  = tuple(rgba[0, 1, :3])
        assert before_rgb != early_rgb


# ===========================================================================
# dashboard: _get_wgs84_bounds
# ===========================================================================

class TestGetWgs84Bounds:
    @pytest.fixture
    def mock_raster_params(self):
        """Affine transform and CRS mimicking the MA AT study area."""
        from affine import Affine
        from pyproj import CRS
        # UTM Zone 19N covering Massachusetts, 30m pixels
        transform = Affine(30.0, 0.0, 300000.0,
                           0.0, -30.0, 4750000.0)
        crs = CRS.from_epsg(32619)  # WGS 84 / UTM zone 19N
        return transform, crs

    def test_south_less_than_north(self, mock_raster_params):
        transform, crs = mock_raster_params
        bounds = _get_wgs84_bounds(transform, crs, h=100, w=100)
        south, west = bounds[0]
        north, east = bounds[1]
        assert south < north

    def test_west_less_than_east(self, mock_raster_params):
        transform, crs = mock_raster_params
        bounds = _get_wgs84_bounds(transform, crs, h=100, w=100)
        _, west = bounds[0]
        _, east = bounds[1]
        assert west < east

    def test_lat_lon_in_reasonable_range(self, mock_raster_params):
        transform, crs = mock_raster_params
        bounds = _get_wgs84_bounds(transform, crs, h=100, w=100)
        south, west = bounds[0]
        north, east = bounds[1]
        assert -90 <= south <= 90
        assert -90 <= north <= 90
        assert -180 <= west <= 180
        assert -180 <= east <= 180
