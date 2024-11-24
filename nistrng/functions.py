# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
# Inspired by the work of David Johnston (C) 2017: https://github.com/dj-on-github/sp800_22_tests
#
# NistRng is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.

# Import packages

import numpy
from tqdm import tqdm  # Importation de tqdm pour les barres de progression

# Import required src

from nistrng import Test
from nistrng.sp800_22r1a import *

# Define default NIST battery constant

SP800_22R1A_BATTERY: dict = {
    "monobit": MonobitTest(),
    "frequency_within_block": FrequencyWithinBlockTest(),
    "runs": RunsTest(),
    "longest_run_ones_in_a_block": LongestRunOnesInABlockTest(),
    "binary_matrix_rank": BinaryMatrixRankTest(),
    "dft": DiscreteFourierTransformTest(),
    "non_overlapping_template_matching": NonOverlappingTemplateMatchingTest(),
    "overlapping_template_matching": OverlappingTemplateMatchingTest(),
    "maurers_universal": MaurersUniversalTest(),
    "linear_complexity": LinearComplexityTest(),
    "serial": SerialTest(),
    "approximate_entropy": ApproximateEntropyTest(),
    "cumulative_sums": CumulativeSumsTest(),
    "random_excursion": RandomExcursionTest(),
    "random_excursion_variant": RandomExcursionVariantTest()
}

# Define cache global variables
# Note: each test is defined by a tuple name and instance

_cached_tests: list = []

# Define functions

def run_all_battery(bits: numpy.ndarray, battery: dict,
                    check_eligibility: bool = True) -> list:
    """
    Run all the given tests in the battery with the given bits as input.
    E.g., a battery of tests is the sp800-22r1a test battery.

    :param bits: the sequence (ndarray) of bits encoding the sequence of integers
    :param battery: the battery of tests (dict with keys as names and values as Test instances) to run on the sequence
    :param check_eligibility: whether to check or not for eligibility. If checked and failed, the associated test returns None
    :return: a list of Result objects or Nones for each not eligible test (if check is required)
    """
    results: list = []
    test_names = list(battery.keys())
    with tqdm(total=len(test_names), desc="Running NIST Tests") as pbar:
        for name in test_names:
            result = run_by_name_battery(name, bits, battery, check_eligibility)
            results.append(result)
            pbar.update(1)
    return results

def run_in_order_battery(bits: numpy.ndarray, battery: dict,
                         check_eligibility: bool = True) -> list:
    """
    Run all the given tests in the battery in order, stopping if a test fails.
    E.g., a battery of tests is the sp800-22r1a test battery.

    :param bits: the sequence (ndarray) of bits encoding the sequence of integers
    :param battery: the battery of tests (dict with keys as names and values as Test instances) to run on the sequence
    :param check_eligibility: whether to check or not for eligibility. If checked and failed, the associated test returns None
    :return: a list of Result objects up to the point of failure or Nones for each not eligible test (if check is required)
    """
    results: list = []
    test_names = list(battery.keys())
    with tqdm(total=len(test_names), desc="Sequentially Running NIST Tests") as pbar:
        for name in test_names:
            result = run_by_name_battery(name, bits, battery, check_eligibility)
            results.append(result)
            pbar.update(1)
            if result is not None:
                test_result, _ = result
                if not test_result.passed:
                    tqdm.write(f"Test Failed: {test_result.name}")
                    break
        else:
            tqdm.write("All tests passed successfully.")
    return results

def run_by_name_battery(test_name: str,
                        bits: numpy.ndarray, battery: dict,
                        check_eligibility: bool = True) -> tuple or None:
    """
    Run the given test in the battery by name with the given bits as input.
    E.g., a battery of tests is the sp800-22r1a test battery.

    :param test_name: the name of the test to run
    :param bits: the sequence (ndarray) of bits encoding the sequence of integers
    :param battery: the battery of tests (dict with keys as names and values as Test instances) to run on the sequence
    :param check_eligibility: whether to check or not for eligibility. If checked and failed, return None
    :return: a tuple of Result object and elapsed time if eligible, None otherwise (if check is required)
    """
    # Retrieve the test from cache or initialize it
    test: Test or None = None
    for cached_name, instance in _cached_tests:
        if cached_name == test_name:
            test = instance
            break
    if test is None:
        test = battery[test_name]
        _cached_tests.append((test_name, test))
    
    # Check for eligibility if required
    if check_eligibility:
        if not test.is_eligible(bits):
            return None
    
    # Display the name of the test being executed
    tqdm.write(f"Executing Test: {test_name}")
    
    # Run the test and return the result
    return test.run(bits)

def check_eligibility_all_battery(bits: numpy.ndarray, battery: dict) -> dict:
    """
    Check the eligibility for all the given tests in the battery with the given bits as input.
    E.g., a battery of tests is the sp800-22r1a test battery.

    :param bits: the sequence (ndarray) of bits encoding the sequence of integers
    :param battery: the battery of tests (dict with keys as names and values as Test instances) to run on the sequence
    :return: a dict with names and corresponding Test instances for eligible tests
    """
    results: dict = {}
    for name in battery.keys():
        if check_eligibility_by_name_battery(name, bits, battery):
            results[name] = battery[name]
    return results

def check_eligibility_by_name_battery(test_name: str,
                                      bits: numpy.ndarray, battery: dict) -> bool:
    """
    Check the eligibility of a specific test by name.

    :param test_name: the name of the test to check
    :param bits: the sequence (ndarray) of bits encoding the sequence of integers
    :param battery: the battery of tests (dict with keys as names and values as Test instances) to run on the sequence
    :return: True if eligible, False otherwise
    """
    # Retrieve the test from cache or initialize it
    test: Test or None = None
    for cached_name, instance in _cached_tests:
        if cached_name == test_name:
            test = instance
            break
    if test is None:
        test = battery[test_name]
        _cached_tests.append((test_name, test))
    
    # Check eligibility
    return test.is_eligible(bits)

def pack_sequence(sequence: numpy.ndarray) -> numpy.ndarray:
    """
    Pack a sequence of signed integers to its binary 8-bit representation using numpy.

    :param sequence: the integer sequence to pack (in the form of a numpy array, ndarray)
    :return: the sequence packed in 8-bit integers as a numpy array (ndarray)
    """
    return numpy.unpackbits(numpy.array(sequence, dtype=numpy.uint8)).astype(numpy.int8)

def unpack_sequence(sequence_binary_encoded: numpy.ndarray) -> numpy.ndarray:
    """
    Unpack a sequence of numbers represented with 8-bits to its signed integer representation using numpy.

    :param sequence_binary_encoded: the 8-bit numbers sequence to unpack (in the form of a numpy array, ndarray)
    :return: the sequence unpacked into signed integers as a numpy array (ndarray)
    """
    return numpy.packbits(numpy.array(sequence_binary_encoded)).astype(numpy.int8)
