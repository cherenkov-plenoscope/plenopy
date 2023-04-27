import numpy as np
import plenopy as pl

CHANNELS = 137
TIME_SLICES = 100


def test_sequence_peak_detection_zeros():
    seq = np.zeros(shape=(TIME_SLICES, CHANNELS))
    peak_slice = pl.sequence.time_slice_with_max_intensity(seq)
    assert peak_slice == 0


def test_sequence_peak_detection_ones():
    seq = np.ones(shape=(TIME_SLICES, CHANNELS))
    peak_slice = pl.sequence.time_slice_with_max_intensity(seq)
    assert peak_slice == 0


def test_sequence_peak_detection_one_peak():
    seq = np.zeros(shape=(TIME_SLICES, CHANNELS))
    seq[55, :] = np.ones(CHANNELS)
    peak_slice = pl.sequence.time_slice_with_max_intensity(seq)
    assert peak_slice == 55


def test_sequence_peak_detection_several_peaks():
    seq = np.zeros(shape=(TIME_SLICES, CHANNELS))
    seq[12, :] = np.ones(CHANNELS)
    seq[23, :] = np.ones(CHANNELS)
    seq[55, :] = 2 * np.ones(CHANNELS)
    seq[60, :] = np.ones(CHANNELS)
    seq[91, :] = np.ones(CHANNELS)
    peak_slice = pl.sequence.time_slice_with_max_intensity(seq)
    assert peak_slice == 55


def test_sequence_integration_return_dict():
    seq = np.zeros(shape=(TIME_SLICES, CHANNELS))
    integral = pl.sequence.integrate_around_arrival_peak(seq)
    assert "integral" in integral
    assert "peak_slice" in integral
    assert "start_slice" in integral
    assert "stop_slice" in integral


def test_sequence_integration_zeros():
    seq = np.zeros(shape=(TIME_SLICES, CHANNELS))
    integral = pl.sequence.integrate_around_arrival_peak(seq)
    np.testing.assert_array_equal(integral["integral"], np.zeros(CHANNELS))


def test_sequence_integration_several_distinct_peaks():
    seq = np.zeros(shape=(TIME_SLICES, CHANNELS))
    seq[12, :] = np.ones(CHANNELS)
    seq[23, :] = np.ones(CHANNELS)
    seq[55, :] = 2 * np.ones(CHANNELS)
    seq[60, :] = np.ones(CHANNELS)
    seq[91, :] = np.ones(CHANNELS)
    integral = pl.sequence.integrate_around_arrival_peak(seq)
    np.testing.assert_array_equal(integral["integral"], 2 * np.ones(CHANNELS))


def test_sequence_integration_several_close_peaks():
    seq = np.zeros(shape=(TIME_SLICES, CHANNELS))
    seq[12, :] = np.ones(CHANNELS)
    seq[23, :] = np.ones(CHANNELS)
    seq[54, :] = 1 * np.ones(CHANNELS)
    seq[55, :] = 2 * np.ones(CHANNELS)
    seq[56, :] = 1 * np.ones(CHANNELS)
    seq[60, :] = np.ones(CHANNELS)
    seq[91, :] = np.ones(CHANNELS)
    integral = pl.sequence.integrate_around_arrival_peak(seq)
    np.testing.assert_array_equal(integral["integral"], 4 * np.ones(CHANNELS))
