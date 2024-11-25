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

import numpy as np
from hashlib import sha256
from dotenv import load_dotenv
import os
from tqdm import tqdm  # Importation of tqdm for the progress bar

class ChaoticSystem:
    def __init__(self, key):
        self.key = key
        # Initialize state with a hash of the key
        self.state = int.from_bytes(sha256(key).digest(), 'big') % (2**32)
        print(f"ChaoticSystem initialized with state: {self.state}")

    def skew_tent_map(self, x, q):
        N = 32
        if 0 < x < q:
            result = (2**N * x) // q
        elif x == q:
            result = 2**N - 1
        elif q < x < 2**N:
            result = (2**N * (2**N - x)) // (2**N - q)
        else:
            result = 0
        # print(f"skew_tent_map: x={x}, q={q}, result={result}")
        return result % (2**32)  # Limit the size of the result

    def dpwlcm_map(self, x, q):
        N = 32
        if 0 < x <= q:
            result = (2**N * x) // q
        elif q < x <= 2**(N - 1):
            result = (2**N * (x - q)) // (2**(N - 1) - q)
        elif 2**(N - 1) < x <= 2**N - q:
            result = (2**N * (2**N - x - q)) // (2**(N - 1) - q)
        elif 2**N - q < x <= 2**N - 1:
            result = (2**N * (2**N - x)) // q
        else:
            result = 2**N - 1 - q
        # print(f"dpwlcm_map: x={x}, q={q}, result={result}")
        return result % (2**32)  # Limit the size of the result

    def generate_keystream(self, length):
        keystream = []
        x = self.state
        # Extract q1 and q2 parameters from the key
        q1 = (int.from_bytes(self.key[:4], 'big') % (2**32 - 1)) + 1
        q2 = (int.from_bytes(self.key[4:8], 'big') % (2**32 - 1)) + 1
        print(f"Generating keystream with q1={q1}, q2={q2}")
        # Display the progress bar if the length is significant
        if length >= 1000:
            iterator = tqdm(range(length), desc="Generating ChaoticSystem keystream")
        else:
            iterator = range(length)
        for i in iterator:
            # Apply chaotic functions
            x = self.skew_tent_map(x, q1)
            x = self.dpwlcm_map(x, q2)
            keystream_value = x & 0xFFFFFFFF
            keystream.append(keystream_value)
            # You can uncomment the following line to display periodic information
            # if i % 100000 == 0:
            #     print(f"Keystream[{i}] = {keystream_value}")
        self.state = x  # Update the internal state
        print(f"ChaoticSystem state updated to: {self.state}")
        return keystream

class ChaoticNeuralNetwork:
    def __init__(self, key):
        print("Initializing Chaotic Neural Network...")
        self.chaotic_system = ChaoticSystem(key)
        # Generate weights and biases for the neural network
        self.input_weights = self.chaotic_system.generate_keystream(16)
        print(f"Input weights: {self.input_weights}")
        self.output_weights = self.chaotic_system.generate_keystream(4)
        print(f"Output weights: {self.output_weights}")
        self.biases = self.chaotic_system.generate_keystream(4)
        print(f"Biases: {self.biases}")
        self.q_parameters = self.chaotic_system.generate_keystream(8)
        print(f"Q parameters: {self.q_parameters}")
        print("Chaotic Neural Network initialized.")

    def activation_function(self, inputs, q1, q2, bias):
        x = (sum(inputs) + bias) % (2**32)
        print(f"Activation function inputs sum (mod 2^32): {x}")
        y1 = self.chaotic_system.skew_tent_map(x, q1)
        y2 = self.chaotic_system.dpwlcm_map(x, q2)
        result = y1 ^ y2
        print(f"Activation function result: {result}")
        return result

    def process(self, data_block):
        print(f"Processing data block: {data_block}")
        outputs = []
        for i in range(0, len(data_block), 4):
            inputs = data_block[i:i+4]
            if len(inputs) < 4:
                inputs += [0] * (4 - len(inputs))
            print(f"Processing inputs: {inputs}")
            q1 = self.q_parameters[(i // 4) % 4]
            q2 = self.q_parameters[((i // 4) % 4) + 4]
            bias = self.biases[(i // 4) % 4]
            # Limit weights to avoid too large numbers
            weighted_inputs = [((inp * w) % (2**32)) for inp, w in zip(inputs, self.input_weights[i:i+4])]
            print(f"Weighted inputs: {weighted_inputs}")
            neuron_output = self.activation_function(weighted_inputs, q1, q2, bias)
            outputs.append(neuron_output)
        final_output = self.non_linear_processing(outputs)
        print(f"Final output after non-linear processing: {final_output}")
        return final_output

    def non_linear_processing(self, outputs):
        h = []
        for i in range(len(outputs)):
            t1 = ((outputs[i] >> 6) | (outputs[i] << (32 - 6))) & 0xFFFFFFFF
            t2 = ((outputs[(i+1)%len(outputs)] >> 11) | (outputs[(i+1)%len(outputs)] << (32 - 11))) & 0xFFFFFFFF
            h_value = outputs[i] ^ t1 ^ t2
            print(f"Non-linear processing [{i}]: h = {h_value}")
            h.append(h_value)
        return h

class CNN_Duplex:
    def __init__(self, key, iv, r=256, c=256):
        print("Initializing CNN_Duplex...")
        self.key = key
        self.iv = iv
        self.state = iv
        self.r = r
        self.c = c
        self.cnn = ChaoticNeuralNetwork(key)
        print("CNN_Duplex initialized.")

    def absorb(self, data):
        print(f"Absorbing data: {data}")
        padded_data = self.pad(data)
        print(f"Padded data: {padded_data}")
        for block in self.chunk_data(padded_data, self.r // 8):
            block_ints = [int.from_bytes(block[i:i+4], 'big') for i in range(0, len(block), 4)]
            print(f"Block ints: {block_ints}")
            processed_block = self.cnn.process(block_ints)
            self.state = self.state_update(self.state, processed_block)
            print(f"State updated to: {self.state}")

    def squeeze(self, length, save_keystream=False):
        print(f"Squeezing keystream of length: {length}")
        output = b''
        state_length = len(self.state)
        total_blocks = (length + state_length - 1) // state_length
        with tqdm(total=total_blocks, desc="Generating CNN_Duplex keystream") as pbar:
            while len(output) < length:
                state_ints = [int.from_bytes(self.state[i:i+4], 'big') for i in range(0, len(self.state), 4)]
                print(f"State ints: {state_ints}")
                processed_block = self.cnn.process(state_ints)
                output_block = b''.join([int.to_bytes(word, 4, 'big') for word in processed_block])
                output += output_block
                if save_keystream:
                    with open('keystream.bin', 'ab') as f:
                        f.write(output_block)
                self.state = self.state_update(self.state, processed_block)
                print(f"Keystream generated: {output_block}")
                print(f"State updated to: {self.state}")
                pbar.update(1)
        return output[:length]

    def pad(self, data):
        pad_len = (-len(data) - 1) % (self.r // 8)
        padded_data = data + b'\x80' + b'\x00' * pad_len
        print(f"Data after padding: {padded_data}")
        return padded_data

    def chunk_data(self, data, chunk_size):
        for i in range(0, len(data), chunk_size):
            yield data[i:i+chunk_size]

    def state_update(self, state, processed_block):
        state_ints = [int.from_bytes(state[i:i+4], 'big') for i in range(0, len(state), 4)]
        new_state_ints = [(s ^ p) & 0xFFFFFFFF for s, p in zip(state_ints, processed_block)]
        new_state = b''.join([int.to_bytes(word, 4, 'big') for word in new_state_ints])
        return new_state

    def encrypt(self, plaintext, associated_data):
        """
        Encrypt the plaintext with associated data.

        :param plaintext: The data to encrypt (bytes)
        :param associated_data: The associated data (bytes)
        :return: Tuple containing ciphertext and tag
        """
        print("Starting encryption...")
        # Absorb associated data only
        self.absorb(associated_data)
        
        # Generate keystream
        keystream = self.squeeze(len(plaintext))
        print(f"Generated keystream of length {len(keystream)}")
        
        # Perform XOR between plaintext and keystream to get ciphertext
        ciphertext = bytes([p ^ k for p, k in zip(plaintext, keystream)])
        print(f"Ciphertext: {ciphertext}")
        
        # Generate a simple tag (e.g., SHA-256 hash of ciphertext and associated data)
        tag_input = ciphertext + associated_data
        tag = sha256(tag_input).digest()[:16]  # Using first 16 bytes as tag
        print(f"Generated tag: {tag}")
        
        return ciphertext, tag

    def decrypt(self, ciphertext, associated_data, tag):
        """
        Decrypt the ciphertext with associated data and verify the tag.

        :param ciphertext: The data to decrypt (bytes)
        :param associated_data: The associated data (bytes)
        :param tag: The authentication tag (bytes)
        :return: Decrypted plaintext (bytes)
        :raises ValueError: If tag verification fails
        """
        print("Starting decryption...")
        # Absorb associated data only
        self.absorb(associated_data)
        
        # Generate keystream
        keystream = self.squeeze(len(ciphertext))
        print(f"Generated keystream of length {len(keystream)}")
        
        # Perform XOR between ciphertext and keystream to get plaintext
        plaintext = bytes([c ^ k for c, k in zip(ciphertext, keystream)])
        print(f"Decrypted plaintext: {plaintext}")
        
        # Verify tag
        tag_input = ciphertext + associated_data
        expected_tag = sha256(tag_input).digest()[:16]
        print(f"Expected tag: {expected_tag}")
        if tag != expected_tag:
            raise ValueError("Invalid authentication tag!")
        print("Tag verification successful.")
        
        return plaintext

# Example usage
if __name__ == "__main__":
    # Import dotenv to load environment variables
    from dotenv import load_dotenv
    import os

    # Load environment variables from .env file
    load_dotenv()

    # Secret key of 16 bytes (128 bits) loaded from .env
    key_env = os.getenv('KEY')
    iv_env = os.getenv('IV')

    if key_env is None or iv_env is None:
        raise ValueError("Key or IV not found in the .env file")

    key = key_env.encode('utf-8')
    iv = iv_env.encode('utf-8')

    # Initialize the encryption scheme
    cnn_duplex = CNN_Duplex(key, iv)

    # Generate a 1MB keystream and save it to 'keystream.bin'
    keystream_length = 1 * 1024 * 1024  # 1 megabyte
    # Before generating, ensure that 'keystream.bin' is empty or delete it if it exists
    if os.path.exists('keystream.bin'):
        os.remove('keystream.bin')

    print("Generating keystream...")
    # The keystream is generated via the squeeze method of CNN_Duplex
    cnn_duplex.squeeze(keystream_length, save_keystream=True)
    print("Keystream generated and saved to 'keystream.bin'")