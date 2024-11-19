# Cryptoleo: Authenticated Encryption Based on Chaotic Neural Networks and Duplex Construction

> **Note**: Cryptoleo is the name for this repository, and the underlying algorithm is based on the research article: [Authenticated Encryption Based on Chaotic Neural Networks and Duplex Construction](https://www.mdpi.com/2073-8994/13/12/2432) by Nabil Abdoun, Safwan El Assad, Thang Manh Hoang, Olivier Deforges, Rima Assaf, and Mohamad Khalil.

## Overview
Cryptoleo is a novel implementation of Authenticated Encryption with Associated Data (AEAD) based on the Modified Duplex Construction (MDC) that utilizes chaotic neural networks (CNN). This approach aims to simultaneously ensure confidentiality, integrity, and authenticity of data transmitted over insecure channels.

Unlike traditional AEAD schemes, Cryptoleo leverages chaotic maps (Skew-Tent Map and Discrete Piecewise Linear Chaotic Map) within its chaotic compression function, providing strong randomness and non-linearity. This implementation is designed for enhanced security, especially against attacks such as differential cryptanalysis, while being efficient for applications requiring authenticated encryption.

## Features
- **Authenticated Encryption with Associated Data (AEAD):** Provides both confidentiality and authentication in a single cryptographic operation.
- **Chaotic Neural Network:** Incorporates a single-layer CNN to generate pseudo-random weights and biases, ensuring unpredictable outputs.
- **Modified Duplex Construction (MDC):** Employs a unique initialization and duplexing phase, with parameter tuning to ensure high resistance to cryptanalytic attacks.
- **Variable Key and Tag Lengths:** Supports flexible key lengths up to 160 bits and tag lengths of up to 256 bits.
- **Robust Security Performance:** Tested against various attacks (e.g., brute force, collision attacks, chosen plaintext attacks) and subjected to different statistical randomness tests.

## Components
- **ChaoticSystem Class:** Implements the chaotic functions Skew-Tent Map and DPWLCM. It generates the pseudo-random keystream used by the neural network to create chaotic weights.
- **ChaoticNeuralNetwork Class:** Uses the chaotic keystream to initialize weights, biases, and other parameters to build a lightweight chaotic neural network.
- **CNN_Duplex Class:** Implements the Modified Duplex Construction to provide authenticated encryption and decryption functionalities.

## Security Analysis
Cryptoleo's security relies on the chaotic properties of its core, ensuring:
- **High Sensitivity to Input and Key:** A single-bit change in the key or plaintext results in a significant change in ciphertext.
- **Statistical Uniformity:** Ciphertexts produced by Cryptoleo show a near-uniform distribution in histograms and pass randomness tests, such as NIST and chi-square tests.
- **Diffusion Effect:** The Strict Avalanche Criterion (SAC) is achieved, meaning each bit in the plaintext affects the output significantly.

## Requirements
- Python 3.7+
- `numpy`: Used for numerical operations.
- `hashlib`: Provides SHA-256 for key hashing.

To install the requirements:
```sh
pip install numpy
pip install python-dotenv
pip install pytest
```

Create a `.env` file with the following content:
```dotenv
KEY=<Your_Secret_Key>
IV=<Your_initial_vector>
```

## Getting Started
1. **Clone the Repository**:
   ```sh
   git clone https://github.com/yourusername/cryptoleo.git
   cd cryptoleo
   ```

2. **Run the Example**:
   ```sh
   python cryptoleo.py
   ```

3. **Encryption and Decryption Example**:
   - Replace the `key`, `iv`, `associated_data`, and `plaintext` variables in `cryptoleo.py` to test different inputs.

## Usage
```python
from cryptoleo import CNN_Duplex

# Define the key and IV
key = b"clefsecrete123456"
iv = b"vecteurinitialiv1"

# Initialize the Duplex Scheme
cnn_duplex = CNN_Duplex(key, iv)

# Associated Data and Plaintext
associated_data = b"Header Information"
plaintext = b"Confidential Message"

# Encryption
ciphertext, tag = cnn_duplex.encrypt(plaintext, associated_data)
print("Ciphertext:", ciphertext)
print("Tag:", tag)

# Decryption
try:
    cnn_duplex_dec = CNN_Duplex(key, iv)
    decrypted_plaintext = cnn_duplex_dec.decrypt(ciphertext, associated_data, tag)
    print("Decrypted Plaintext:", decrypted_plaintext.decode('utf-8'))
except ValueError:
    print("Authentication Failed: Tag does not match.")
```

## Testing
- **Key Sensitivity Test**: Change one bit in the key and observe changes in the ciphertext.
- **Message Sensitivity Test**: Change one bit in the plaintext and verify the generated ciphertext.

## Limitations and Future Work
- **Performance**: The current implementation is computationally intensive due to the chaotic maps and neural network.
- **Hardware Implementation**: The next step involves deploying the system on FPGA and analyzing the hardware performance.

## References
- Nabil Abdoun et al., "Authenticated Encryption Based on Chaotic Neural Networks and Duplex Construction," *Symmetry*, 2021. [Link to Article](https://www.mdpi.com/2073-8994/13/12/2432)
- P. Rogaway, "Authenticated Encryption with Associated Data," *Proceedings of the 9th ACM Conference on Computer and Communications Security*, 2002.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
