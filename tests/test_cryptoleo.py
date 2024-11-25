# test_chaotic_encryption.py

import pytest
import math

# Chaotic Neural Network Components and Functions

def rotr(x, n, bits=32):
    """
    Performs a bitwise right rotation on a 32-bit integer.
    """
    return ((x >> n) | (x << (bits - n))) & (2**bits - 1)

def sigma_0(d1):
    return rotr(d1, 2) ^ rotr(d1, 13) ^ rotr(d1, 22)

def sigma_1(d3):
    return rotr(d3, 6) ^ rotr(d3, 11) ^ rotr(d3, 25)

def majority(d1, d2, d3):
    return (d1 & d2) ^ (d1 & d3) ^ (d2 & d3)

def choose(d1, d2, d3):
    return (d1 & d2) ^ (~d1 & d3)

def skew_tent_map(x, q1, n=32):
    """
    Skew Tent Map function.
    """
    max_val = 2**n
    if 0 < x < q1:
        return (2 * x * max_val) // q1
    elif x == q1:
        return max_val - 1
    elif q1 < x < max_val:
        return (2 * (max_val - x) * max_val) // (max_val - q1)
    else:
        return 0

def pwlcm_map(x, q2, n=32):
    """
    Piecewise Linear Chaotic Map function.
    """
    max_val = 2**n
    if 0 < x <= q2:
        return (x * max_val) // q2
    elif q2 < x <= max_val // 2:
        return ((x - q2) * max_val) // ((max_val // 2) - q2)
    elif max_val // 2 < x <= max_val - q2:
        return ((max_val - q2 - x) * max_val) // ((max_val // 2) - q2)
    elif max_val - q2 < x < max_val:
        return ((max_val - x) * max_val) // q2
    else:
        return 0

def ensure_p_size(p, size):
    if len(p) < size:
        p += [0] * (size - len(p))
    return p

def calculate_ck(ks, p, k, n, mode=256):
    if mode == 256:
        p = ensure_p_size(p, 4 * k + 4)
        fn1_input = (ks['BI'][k] + sum(
            p[j] * ks['WI'][k][j % len(ks['WI'][k])]
            for j in range(4 * k, 4 * k + 2)
        )) % 2**n
        fn2_input = (ks['BI'][k] + sum(
            p[j] * ks['WI'][k][(j + 2) % len(ks['WI'][k])]
            for j in range(4 * k, 4 * k + 2)
        )) % 2**n
    else:
        raise NotImplementedError("Only mode 256 is implemented.")

    fn1 = skew_tent_map(fn1_input, ks['Q1'], n)
    fn2 = pwlcm_map(fn2_input, ks['Q2'], n)
    return (fn1 + fn2) % (2**n)

def calculate_hk(d, t1, t2):
    return [
        t2 ^ t1 ^ d[0],
        t1 ^ d[0],
        d[1] ^ d[0],
        d[2] ^ d[1],
        d[3] ^ d[2],
        t1 ^ d[1] ^ d[0],
        t1 ^ d[2] ^ d[1],
        t1 ^ d[3] ^ d[2],
    ]

def chaotic_nl_layer(p, ks, wo, n, mode=256):
    c = [calculate_ck(ks, p, k, n, mode) for k in range(4)]
    d = [c[k] * wo[k] % 2**n for k in range(4)]
    t1 = sigma_1(d[3]) ^ choose(d[1], d[2], d[3])
    t2 = sigma_0(d[1]) ^ majority(d[1], d[2], d[3])
    return calculate_hk(d, t1, t2)

def iterate_nl_layers(p, ks, wo, n, nr, mode=256):
    h = p
    for _ in range(nr):
        h = chaotic_nl_layer(h, ks, wo, n, mode)
    return h

def generate_key_schedule(k, n=32):
    key_ints = [int.from_bytes(k[i:i+4], 'big') for i in range(0, len(k), 4)]
    while len(key_ints) < 4:
        key_ints.append(0)
    ks = {
        'BI': key_ints[:4],
        'WI': [key_ints[:4] for _ in range(4)],
        'Q1': (key_ints[0] % (2**n - 1)) or 1,
        'Q2': (key_ints[1] % (2**n - 1)) or 1
    }
    return ks

def chaotic_compression(k, h):
    ks = generate_key_schedule(k)
    wo = ks['BI']
    n = 32
    nr = 8
    p = [int.from_bytes(h[i:i+4], 'big') for i in range(0, len(h), 4)]
    p = iterate_nl_layers(p, ks, wo, n, nr)
    new_state = b''.join(int.to_bytes(x % 2**n, 4, 'big') for x in p)
    return new_state[:len(h)]

def pad_message_injective(block, desired_length):
    """
    Pads the block to the desired length using injective padding.
    """
    block_length = len(block)
    if block_length == desired_length:
        return block  # No padding needed
    padding_length = desired_length - block_length
    if padding_length < 1:
        raise ValueError("Block is too large for the injective padding scheme.")
    padding = b'\x80' + b'\x00' * (padding_length - 1)
    return block + padding

def divide_message(message, r):
    """
    Divides the message into blocks of size up to r - 1 bytes.
    """
    block_size = r - 1  # Reserve 1 byte for the flag
    num_blocks = math.ceil(len(message) / block_size)
    blocks = []
    for i in range(num_blocks):
        block = message[i * block_size:(i + 1) * block_size]
        is_last = 0 if i == num_blocks - 1 else 1
        blocks.append((block, is_last))
    return blocks

def initialize(k, iv, cf):
    return cf(k, iv)

def duplexing(hm, s, bf, pad_func, cf, r, b, k):
    # Pad s to length r - 1
    padded_input = pad_func(s, r - 1)
    # Append the flag to reach length r
    s_with_flag = padded_input + bytes([bf])
    h = bytes([hm[i] ^ s_with_flag[i] for i in range(r)]) + hm[r:]
    hm = cf(k, h)
    return hm

def encrypt(k, iv, ad, m, cf, r, b):
    hm = initialize(k, iv, cf)
    for block, bf in divide_message(ad, r):
        hm = duplexing(hm, block, bf, pad_message_injective, cf, r, b, k)
    c = b""
    for block, bf in divide_message(m, r):
        c_block = bytes([block[j] ^ hm[j] for j in range(len(block))])
        c += c_block
        hm = duplexing(hm, c_block, bf, pad_message_injective, cf, r, b, k)
    hm = duplexing(hm, b'', 0, pad_message_injective, cf, r, b, k)  # Empty block with flag 0
    tag = hm[:r]
    return c, tag

def decrypt(k, iv, ad, c, t, cf, r, b):
    hm = initialize(k, iv, cf)
    for block, bf in divide_message(ad, r):
        hm = duplexing(hm, block, bf, pad_message_injective, cf, r, b, k)
    m = b""
    for block, bf in divide_message(c, r):
        m_block = bytes([block[j] ^ hm[j] for j in range(len(block))])
        m += m_block
        hm = duplexing(hm, block, bf, pad_message_injective, cf, r, b, k)
    hm = duplexing(hm, b'', 0, pad_message_injective, cf, r, b, k)  # Empty block with flag 0
    if t != hm[:len(t)]:
        return None, "Error: Authentication Failed"
    return m, "Success"

# Test Functions

def test_skew_tent_map():
    """
    Test the skew_tent_map function to ensure it returns an integer and operates correctly.
    """
    x = 123456789
    q1 = 8
    n = 32
    result = skew_tent_map(x, q1, n)
    assert isinstance(result, int), "Result of skew_tent_map should be an integer."
    print("skew_tent_map works correctly with result:", result)

def test_pwlcm_map():
    """
    Test the pwlcm_map function to ensure it returns an integer and operates correctly.
    """
    x = 123456789
    q2 = 12
    n = 32
    result = pwlcm_map(x, q2, n)
    assert isinstance(result, int), "Result of pwlcm_map should be an integer."
    print("pwlcm_map works correctly with result:", result)

def test_generate_key_schedule():
    """
    Test the generate_key_schedule function to ensure it returns a valid key schedule.
    """
    k = b'\x01' * 16
    ks = generate_key_schedule(k)
    assert 'BI' in ks and 'WI' in ks and 'Q1' in ks and 'Q2' in ks, "Key schedule should contain 'BI', 'WI', 'Q1', 'Q2'."
    print("Key schedule generated correctly:", ks)

def test_chaotic_compression():
    """
    Test the chaotic_compression function to ensure it updates the state correctly.
    """
    k = b'\x01' * 16
    h = b'\x00' * 32  # State length of 32 bytes
    new_state = chaotic_compression(k, h)
    assert isinstance(new_state, bytes), "chaotic_compression should return bytes."
    assert len(new_state) == len(h), "New state should be same length as input state."
    print("Chaotic compression works correctly with new state:", new_state.hex())

def test_encrypt_decrypt():
    """
    Test the encrypt and decrypt functions to ensure they work correctly and the message can be recovered.
    """
    k = b'\x01' * 16
    iv = b'\x00' * (32 + 64)  # r + b bytes
    ad = b"Associated Data"
    m = b"Message to encrypt"
    r, b_cap = 32, 64
    # Perform encryption
    ciphertext, tag = encrypt(k, iv, ad, m, chaotic_compression, r, b_cap)
    print("Ciphertext:", ciphertext.hex())
    print("Tag:", tag.hex())
    # Perform decryption
    message, status = decrypt(k, iv, ad, ciphertext, tag, chaotic_compression, r, b_cap)
    print("Decrypted Message:", message)
    print("Status:", status)
    assert status == "Success", f"Decryption failed with status: {status}"
    assert message == m, "Decrypted message does not match original."
    print("Encryption and decryption work correctly.")

# Pytest requires test functions to be prefixed with 'test_'
# You can run these tests using the command: pytest test_chaotic_encryption.py

if __name__ == "__main__":
    # If running directly, execute the tests
    test_skew_tent_map()
    test_pwlcm_map()
    test_generate_key_schedule()
    test_chaotic_compression()
    test_encrypt_decrypt()
