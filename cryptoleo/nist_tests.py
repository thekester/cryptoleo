# nist_tests.py
import math
import numpy as np

def monobit_test(bits):
    n = len(bits)
    s_obs = bits.count('1') - bits.count('0')
    s = abs(s_obs) / math.sqrt(n)
    p_value = math.erfc(s / math.sqrt(2))
    return p_value

def runs_test(bits):
    n = len(bits)
    pi = bits.count('1') / n
    tau = 2 / math.sqrt(n)
    if abs(pi - 0.5) >= tau:
        return 0.0  # Échec du test
    v_obs = 1
    for i in range(1, n):
        if bits[i] != bits[i-1]:
            v_obs += 1
    p_value = math.erfc(abs(v_obs - (2 * n * pi * (1 - pi))) / (2 * math.sqrt(2 * n) * pi * (1 - pi)))
    return p_value

def poker_test(bits, m=4):
    n = len(bits)
    if n % m != 0:
        bits = bits[:-(n % m)]
        n = len(bits)
    k = n // m
    counts = {}
    for i in range(k):
        block = bits[i*m:(i+1)*m]
        counts[block] = counts.get(block, 0) + 1
    sum_val = sum(count**2 for count in counts.values())
    x = ((16 / k) * sum_val) - k
    p_value = math.exp(-x / 2)
    return p_value

def convert_keystream_to_bits(input_file):
    bits = ''
    with open(input_file, 'rb') as f:
        byte = f.read(1)
        while byte:
            bits += bin(ord(byte))[2:].zfill(8)
            byte = f.read(1)
    return bits

if __name__ == "__main__":
    # Lire le keystream depuis le fichier
    bits = convert_keystream_to_bits('keystream.bin')

    # Limiter la longueur pour les tests (par exemple, 1 million de bits)
    bits = bits[:1000000]

    # Exécuter les tests NIST
    p_value_monobit = monobit_test(bits)
    print(f"Monobit Test p-value: {p_value_monobit}")

    p_value_runs = runs_test(bits)
    print(f"Runs Test p-value: {p_value_runs}")

    p_value_poker = poker_test(bits)
    print(f"Poker Test p-value: {p_value_poker}")
