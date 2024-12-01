import math
import os

# Padding Function
def pad_message_injective(block, desired_length):
    """
    Applies injective padding to the input block to make its length equal to desired_length.

    Parameters:
    - block (bytes): The input block to be padded.
    - desired_length (int): The desired block length after padding.

    Returns:
    - bytes: The padded block.

    Padding Scheme:
    - If the block is shorter than desired_length, append a 0x80 byte followed by zeros.
    - If the block is exactly length desired_length, return it unchanged.
    - If the block is longer than desired_length, raise an error.
    """
    block_length = len(block)
    if block_length == desired_length:
        return block  # No padding needed
    padding_length = desired_length - block_length
    if padding_length < 1:
        raise ValueError("Block is too large for the injective padding scheme.")
    # Create padding: start with 0x80, then zeros
    padding = b'\x80' + b'\x00' * (padding_length - 1)
    return block + padding

# Divide Message into Blocks
def divide_message(message, r):
    """
    Divides the message into blocks of length up to r - 1 bytes, marking each block
    with a flag indicating whether it's the last block.

    Parameters:
    - message (bytes): The message to be divided.
    - r (int): The rate parameter, which determines block size.

    Returns:
    - List[Tuple[bytes, int]]: A list of tuples where each tuple contains:
        - block (bytes): A segment of the message.
        - is_last (int): A flag (0 if it's the last block, 1 if not).
    """
    block_size = r - 1  # Reserve 1 byte for flag
    num_blocks = math.ceil(len(message) / block_size)
    blocks = []
    for i in range(num_blocks):
        # Extract block of size block_size
        block = message[i * block_size:(i + 1) * block_size]
        # Determine if it's the last block
        is_last = 0 if i == num_blocks - 1 else 1
        blocks.append((block, is_last))
    return blocks

# Chaotic Neural Network Components

def rotr(x, n, bits=32):
    """
    Performs a bitwise right rotation on a 32-bit integer.

    Parameters:
    - x (int): The integer to rotate.
    - n (int): The number of bits to rotate.
    - bits (int): The bit width of the integer (default is 32).

    Returns:
    - int: The result after rotation.
    """
    return ((x >> n) | (x << (bits - n))) & (2**bits - 1)

def sigma_0(d1):
    """
    Sigma_0 function as defined in cryptographic hash functions.

    Parameters:
    - d1 (int): Input integer.

    Returns:
    - int: The result of the sigma_0 function.
    """
    return rotr(d1, 2) ^ rotr(d1, 13) ^ rotr(d1, 22)

def sigma_1(d3):
    """
    Sigma_1 function as defined in cryptographic hash functions.

    Parameters:
    - d3 (int): Input integer.

    Returns:
    - int: The result of the sigma_1 function.
    """
    return rotr(d3, 6) ^ rotr(d3, 11) ^ rotr(d3, 25)

def majority(d1, d2, d3):
    """
    Majority function used in cryptographic algorithms.

    Parameters:
    - d1, d2, d3 (int): Input integers.

    Returns:
    - int: The majority value of the inputs.
    """
    return (d1 & d2) ^ (d1 & d3) ^ (d2 & d3)

def choose(d1, d2, d3):
    """
    Choose function used in cryptographic algorithms.

    Parameters:
    - d1, d2, d3 (int): Input integers.

    Returns:
    - int: The result of the choose function.
    """
    return (d1 & d2) ^ (~d1 & d3)

# Chaotic Maps

def skew_tent_map(x, q1, n=32):
    """
    Skew Tent Map function for generating chaotic sequences.

    Parameters:
    - x (int): Current state.
    - q1 (int): Control parameter (must be in range 1 to 2^n - 1).
    - n (int): Bit width (default is 32).

    Returns:
    - int: Next state in the chaotic sequence.
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
    Piecewise Linear Chaotic Map (PWLCM) function.

    Parameters:
    - x (int): Current state.
    - q2 (int): Control parameter (must be in range 1 to 2^n - 1).
    - n (int): Bit width (default is 32).

    Returns:
    - int: Next state in the chaotic sequence.
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

# Ensure p has enough size
def ensure_p_size(p, size):
    """
    Ensures that the parameter list p has at least 'size' elements.

    Parameters:
    - p (List[int]): The parameter list.
    - size (int): The desired minimum size.

    Returns:
    - List[int]: The parameter list, padded with zeros if necessary.
    """
    if len(p) < size:
        p += [0] * (size - len(p))
    return p

# Calculate c_k
def calculate_ck(ks, p, k, n, mode=256):
    """
    Calculates the c_k value used in the chaotic neural network layer.

    Parameters:
    - ks (Dict): Key schedule containing 'BI', 'WI', 'Q1', and 'Q2'.
    - p (List[int]): Current state parameters.
    - k (int): Index for the c_k calculation.
    - n (int): Bit width for calculations.
    - mode (int): Mode of operation (default is 256).

    Returns:
    - int: The calculated c_k value.
    """
    if mode == 256:
        p = ensure_p_size(p, 4 * k + 4)
        # Calculate input for skew_tent_map and pwlcm_map
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

    # Apply chaotic maps
    fn1 = skew_tent_map(fn1_input, ks['Q1'], n)
    fn2 = pwlcm_map(fn2_input, ks['Q2'], n)
    # Combine results modulo 2^n
    return (fn1 + fn2) % (2**n)

# Calculate h_k
def calculate_hk(d, t1, t2):
    """
    Calculates the h_k vector for the next layer in the chaotic neural network.

    Parameters:
    - d (List[int]): The list of d_k values from the current layer.
    - t1, t2 (int): Intermediate values calculated from sigma and other functions.

    Returns:
    - List[int]: The h_k values for the next layer.
    """
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

# Perform a single Non-Linear (NL) Layer
def chaotic_nl_layer(p, ks, wo, n, mode=256):
    """
    Performs a single non-linear layer of the chaotic neural network.

    Parameters:
    - p (List[int]): Current state parameters.
    - ks (Dict): Key schedule.
    - wo (List[int]): Output weights.
    - n (int): Bit width.
    - mode (int): Mode of operation.

    Returns:
    - List[int]: Updated state parameters after the NL layer.
    """
    # Calculate c_k values
    c = [calculate_ck(ks, p, k, n, mode) for k in range(4)]
    # Calculate d_k values by multiplying c_k with output weights
    d = [c[k] * wo[k] % 2**n for k in range(4)]
    # Compute intermediate values t1 and t2
    t1 = sigma_1(d[3]) ^ choose(d[1], d[2], d[3])
    t2 = sigma_0(d[1]) ^ majority(d[1], d[2], d[3])
    # Calculate h_k for next layer
    return calculate_hk(d, t1, t2)

# Iterate NL Layers
def iterate_nl_layers(p, ks, wo, n, nr, mode=256):
    """
    Iteratively applies the non-linear layer to the parameters.

    Parameters:
    - p (List[int]): Initial state parameters.
    - ks (Dict): Key schedule.
    - wo (List[int]): Output weights.
    - n (int): Bit width.
    - nr (int): Number of rounds (iterations).
    - mode (int): Mode of operation.

    Returns:
    - List[int]: Final state parameters after nr iterations.
    """
    h = p
    for _ in range(nr):
        h = chaotic_nl_layer(h, ks, wo, n, mode)
    return h

# Key Schedule Generation
def generate_key_schedule(k, n=32):
    """
    Generates the key schedule from the secret key.

    Parameters:
    - k (bytes): The secret key.
    - n (int): Bit width for calculations.

    Returns:
    - Dict: Key schedule containing 'BI', 'WI', 'Q1', and 'Q2'.
    """
    # Convert key into list of integers
    key_ints = [int.from_bytes(k[i:i+4], 'big') for i in range(0, len(k), 4)]
    # Pad with zeros if necessary
    while len(key_ints) < 4:
        key_ints.append(0)
    # Generate key schedule
    ks = {
        'BI': key_ints[:4],  # Biases for input neurons
        'WI': [key_ints[:4] for _ in range(4)],  # Weights for input neurons
        'Q1': (key_ints[0] % (2**n - 1)) or 1,  # Control parameter for skew tent map (avoid zero)
        'Q2': (key_ints[1] % (2**n - 1)) or 1   # Control parameter for PWLCM (avoid zero)
    }
    return ks

# Compression function using chaotic neural network layers
def chaotic_compression(k, h):
    """
    Compression function that applies chaotic neural network layers to update the state.

    Parameters:
    - k (bytes): The secret key.
    - h (bytes): Current state.

    Returns:
    - bytes: Updated state after applying the chaotic neural network.
    """
    # Generate key schedule
    ks = generate_key_schedule(k)
    wo = ks['BI']  # Use biases as output weights for simplicity
    n = 32  # Bit width
    nr = 8  # Number of rounds
    # Convert state to list of integers
    p = [int.from_bytes(h[i:i+4], 'big') for i in range(0, len(h), 4)]
    # Iterate the non-linear layers
    p = iterate_nl_layers(p, ks, wo, n, nr)
    # Convert back to bytes
    new_state = b''.join(int.to_bytes(x % 2**n, 4, 'big') for x in p)
    return new_state[:len(h)]

# CNN-Duplex Initialization
def initialize(k, iv, cf):
    """
    Initializes the state for encryption/decryption.

    Parameters:
    - k (bytes): Secret key.
    - iv (bytes): Initialization vector.
    - cf (function): Compression function.

    Returns:
    - bytes: Initial state after applying the compression function.
    """
    return cf(k, iv)

# Duplexing Phase
def duplexing(hm, s, bf, pad_func, cf, r, b, k):
    """
    Performs the duplexing operation to update the state with input data.

    Parameters:
    - hm (bytes): Current state.
    - s (bytes): Input data.
    - bf (int): Flag indicating whether more data follows (1) or not (0).
    - pad_func (function): Function to pad input data.
    - cf (function): Compression function.
    - r (int): Rate parameter.
    - b (int): Block size.
    - k (bytes): Secret key.

    Returns:
    - bytes: Updated state after duplexing.
    """
    # Pad input data to length r - 1
    padded_input = pad_func(s, r - 1)
    # Append the flag byte to reach length r
    s_with_flag = padded_input + bytes([bf])
    # XOR with the first 'r' bytes of hm
    h = bytes([hm[i] ^ s_with_flag[i] for i in range(r)]) + hm[r:]
    # Update state using compression function
    hm = cf(k, h)
    return hm

# AEAD Encryption
def encrypt(k, iv, ad, m, cf, r, b):
    """
    Authenticated Encryption with Associated Data (AEAD) encryption function.

    Parameters:
    - k (bytes): Secret key.
    - iv (bytes): Initialization vector.
    - ad (bytes): Associated data to be authenticated but not encrypted.
    - m (bytes): Plaintext message to be encrypted.
    - cf (function): Compression function.
    - r (int): Rate parameter.
    - b (int): Block size.

    Returns:
    - Tuple[bytes, bytes]: The ciphertext and the authentication tag.
    """
    # Initialize state
    hm = initialize(k, iv, cf)
    # Absorb associated data
    for block, bf in divide_message(ad, r):
        hm = duplexing(hm, block, bf, pad_message_injective, cf, r, b, k)
    # Encrypt message
    c = b""
    for block, bf in divide_message(m, r):
        # Generate ciphertext block
        c_block = bytes([block[j] ^ hm[j] for j in range(len(block))])
        c += c_block
        # Update state using ciphertext block
        hm = duplexing(hm, c_block, bf, pad_message_injective, cf, r, b, k)
    # Final duplexing with empty block and flag 0 to generate tag
    hm = duplexing(hm, b'', 0, pad_message_injective, cf, r, b, k)
    tag = hm[:r]  # Extract authentication tag
    return c, tag

# AEAD Decryption
def decrypt(k, iv, ad, c, t, cf, r, b):
    """
    Authenticated Encryption with Associated Data (AEAD) decryption function.

    Parameters:
    - k (bytes): Secret key.
    - iv (bytes): Initialization vector.
    - ad (bytes): Associated data to be authenticated.
    - c (bytes): Ciphertext to be decrypted.
    - t (bytes): Authentication tag to be verified.
    - cf (function): Compression function.
    - r (int): Rate parameter.
    - b (int): Block size.

    Returns:
    - Tuple[Optional[bytes], str]: The decrypted plaintext and a status message.
    """
    # Initialize state
    hm = initialize(k, iv, cf)
    # Absorb associated data
    for block, bf in divide_message(ad, r):
        hm = duplexing(hm, block, bf, pad_message_injective, cf, r, b, k)
    # Decrypt ciphertext
    m = b""
    for block, bf in divide_message(c, r):
        # Update state using ciphertext block
        hm = duplexing(hm, block, bf, pad_message_injective, cf, r, b, k)
        # Generate plaintext block
        m_block = bytes([block[j] ^ hm[j] for j in range(len(block))])
        m += m_block
    # Final duplexing with empty block and flag 0 for tag verification
    hm = duplexing(hm, b'', 0, pad_message_injective, cf, r, b, k)
    # Verify authentication tag
    if t != hm[:len(t)]:
        return None, "Error: Authentication Failed"
    return m, "Success"

def generate_keystream(k, iv, length, cf, r, b):
    """
    Generates a keystream of the specified length.

    Parameters:
    - k (bytes): Secret key.
    - iv (bytes): Initialization vector.
    - length (int): The number of bytes to generate.
    - cf (function): Compression function.
    - r (int): Rate parameter.
    - b (int): Block size.

    Returns:
    - bytes: The generated keystream.
    """
    hm = initialize(k, iv, cf)
    keystream = b''

    while len(keystream) < length:
        # Extract 'r' bytes from the state as keystream
        ks_part = hm[:r]
        keystream += ks_part
        # Update the state with an empty input and flag 0
        hm = duplexing(hm, b'', 0, pad_message_injective, cf, r, b, k)
    return keystream[:length]


# Example Parameters
if __name__ == "__main__":
    # Rate and Block sizes
    r, b = 32, 64  # 'r' is the rate (in bytes), 'b' is the capacity of the duplex construction (in bytes)

    # Secret key and Initialization Vector
    k = b'\x01' * 16  # Secret key of 16 bytes (128 bits)
    iv = b'\x00' * (r + b)  # Initialization vector, length depends on 'r' and 'b'

    # Associated Data and Message
    ad = b"Associated Data"  # Data to be authenticated but not encrypted
    m = b"Message to encrypt"  # Plaintext message to be encrypted

    # Perform encryption
    ciphertext, tag = encrypt(k, iv, ad, m, chaotic_compression, r, b)

    # Perform decryption
    message, status = decrypt(k, iv, ad, ciphertext, tag, chaotic_compression, r, b)

    # Output the results
    print("Ciphertext:", ciphertext)
    print("Tag:", tag)
    print("Decrypted Message:", message)
    print("Status:", status)

    # Generate a keystream of desired length, e.g., 1 MB
    keystream_length = 1024 * 1024  # 1 MB in bytes
    keystream = generate_keystream(k, iv, keystream_length, chaotic_compression, r, b)

    # Save the keystream to a binary file
    with open('keystream.bin', 'wb') as f:
        f.write(keystream)

    print("Keystream generated and saved to 'keystream.bin'.")

    # Generate 100 keystream files for NIST STS
    num_files = 100
    bits_per_file = 1_000_000
    bytes_per_file = bits_per_file // 8
    if bits_per_file % 8 != 0:
        bytes_per_file += 1

    data_filenames = []

    for i in range(1, num_files + 1):
        # Use a unique IV for each keystream file
        iv = os.urandom(r + b)  # Generates a random IV

        # Generate the keystream
        keystream = generate_keystream(k, iv, bytes_per_file, chaotic_compression, r, b)

        # Convert keystream to a string of '0's and '1's
        keystream_bits = ''.join(format(byte, '08b') for byte in keystream)

        # Ensure the keystream_bits has exactly bits_per_file bits
        keystream_bits = keystream_bits[:bits_per_file]

        # Save the keystream bits to a file named 'dataXXX'
        filename = f"data{str(i).zfill(3)}"
        with open(filename, 'w') as f:
            f.write(keystream_bits)

        print(f"File '{filename}' generated with {len(keystream_bits)} bits.")

        # Optional: Print bit frequencies
        zeros = keystream_bits.count('0')
        ones = keystream_bits.count('1')
        print(f"Keystream '{filename}' contains {zeros} zeros and {ones} ones.")

        data_filenames.append(filename)

    # Assemble the files into one
    assembled_filename = 'keystream_all.txt'
    with open(assembled_filename, 'w') as outfile:
        for fname in data_filenames:
            with open(fname, 'r') as infile:
                outfile.write(infile.read())

    print(f"All keystream files assembled into '{assembled_filename}'.")

    # Delete the individual data files
    for fname in data_filenames:
        os.remove(fname)
        print(f"File '{fname}' deleted.")

    print("All individual keystream files deleted.")