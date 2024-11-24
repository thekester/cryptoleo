#
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
import math

# Import required src

from nistrng import Test, Result



class CumulativeSumsTest(Test):
    """
    Cumulative sums test as described in NIST paper: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
    The focus of this test is the maximal excursion (from zero) of the random walk defined by the cumulative sum of adjusted (-1, +1) digits in the sequence.
    The purpose of the test is to determine whether the cumulative sum of the partial sequences occurring in the tested sequence is too large or too small
    relative to the expected behavior of that cumulative sum for random sequences.
    This cumulative sum may be considered as a random walk. For a random sequence, the excursions of the random walk should
    be near zero. For certain types of non-random sequences, the excursions of this random walk from zero will be large.

    The significance value of the test is 0.01.
    """

    def __init__(self):
        # Generate base Test class
        super(CumulativeSumsTest, self).__init__("Cumulative Sums", 0.01)

    def _execute(self, bits: numpy.ndarray) -> Result:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # Convert bits to a signed integer type and replace 0 with -1
        bits_copy: numpy.ndarray = bits.astype(numpy.int64).copy()
        bits_copy[bits_copy == 0] = -1

        # Compute forward cumulative sums
        forward_cumsum = numpy.cumsum(bits_copy)
        forward_max = numpy.max(numpy.abs(forward_cumsum))

        # Compute backward cumulative sums
        backward_cumsum = numpy.cumsum(bits_copy[::-1])
        backward_max = numpy.max(numpy.abs(backward_cumsum))

        # Compute the scores (P-Values)
        score_1: float = self._compute_p_value(bits_copy.size, forward_max)
        score_2: float = self._compute_p_value(bits_copy.size, backward_max)

        # Return result based on significance values
        if score_1 >= self.significance_value and score_2 >= self.significance_value:
            return Result(self.name, True, numpy.array([score_1, score_2]))
        return Result(self.name, False, numpy.array([score_1, score_2]))

    def is_eligible(self, bits: numpy.ndarray) -> bool:
        """
        Overridden method of Test class: check its docstring for further information.
        """
        # This test is always eligible for any sequence
        return True

    @staticmethod
    def _compute_p_value(sequence_size: int, max_excursion: int) -> float:
        """
        Compute P-Value given the sequence size and the max excursion.

        :param sequence_size: the length of the sequence of bits
        :param max_excursion: the max excursion backward or forward
        :return: the computed float P-Value
        """
        # Execute first sum
        sum_a: float = 0.0
        start_k: int = int(math.floor((((float(-sequence_size) / max_excursion) + 1.0) / 4.0)))
        end_k: int = int(math.floor((((float(sequence_size) / max_excursion) - 1.0) / 4.0)))
        for k in range(start_k, end_k + 1):
            c: float = 0.5 * math.erfc(-(((4.0 * k) + 1.0) * max_excursion) / math.sqrt(sequence_size) * math.sqrt(0.5))
            d: float = 0.5 * math.erfc(-(((4.0 * k) - 1.0) * max_excursion) / math.sqrt(sequence_size) * math.sqrt(0.5))
            sum_a = sum_a + c - d
        # Execute second sum
        sum_b: float = 0.0
        start_k = int(math.floor((((float(-sequence_size) / max_excursion) - 3.0) / 4.0)))
        end_k = int(math.floor((((float(sequence_size) / max_excursion) - 1.0) / 4.0)))
        for k in range(start_k, end_k + 1):
            c: float = 0.5 * math.erfc(-(((4.0 * k) + 3.0) * max_excursion) / math.sqrt(sequence_size) * math.sqrt(0.5))
            d: float = 0.5 * math.erfc(-(((4.0 * k) + 1.0) * max_excursion) / math.sqrt(sequence_size) * math.sqrt(0.5))
            sum_b = sum_b + c - d
        # Return value
        return 1.0 - sum_a + sum_b