import numpy as np
import plenopy as pl


def test_benchmark_of_perfect_detector():
    number_all_photons = 1000 * 1000
    number_cherenkov_photons = 100
    pulse_origins = -100 * np.ones(number_all_photons, dtype=np.int)
    pulse_origins[0:number_cherenkov_photons] = np.arange(
        number_cherenkov_photons
    )

    photon_ids_cherenkov = np.arange(number_cherenkov_photons)

    result = pl.classify.benchmark(
        pulse_origins=pulse_origins, photon_ids_cherenkov=photon_ids_cherenkov
    )

    assert result["num_true_positives"] == number_cherenkov_photons
    assert (
        result["num_true_negatives"]
        == number_all_photons - number_cherenkov_photons
    )
    assert result["num_false_negatives"] == 0
    assert result["num_false_positives"] == 0
    assert (
        number_all_photons
        - result["num_true_negatives"]
        - result["num_true_positives"]
        - result["num_false_positives"]
        - result["num_false_negatives"]
    ) == 0


def test_benchmark_of_normal_detector():
    number_all_photons = 1000 * 1000
    number_cherenkov_photons = 100
    pulse_origins = -100 * np.ones(number_all_photons, dtype=np.int)
    pulse_origins[0:number_cherenkov_photons] = np.arange(
        number_cherenkov_photons
    )

    photon_ids_cherenkov = np.arange(
        int(number_cherenkov_photons / 2),
        int(number_cherenkov_photons * 1.5),
        dtype=np.int,
    )

    result = pl.classify.benchmark(
        pulse_origins=pulse_origins, photon_ids_cherenkov=photon_ids_cherenkov
    )

    assert result["num_true_positives"] == number_cherenkov_photons / 2
    assert (
        result["num_true_negatives"]
        == number_all_photons - number_cherenkov_photons * 1.5
    )
    assert result["num_false_negatives"] == number_cherenkov_photons / 2
    assert result["num_false_positives"] == number_cherenkov_photons / 2
    assert (
        number_all_photons
        - result["num_true_negatives"]
        - result["num_true_positives"]
        - result["num_false_positives"]
        - result["num_false_negatives"]
    ) == 0
