import pytest
from cryptoleo.cryptoleo import ChaoticSystem, ChaoticNeuralNetwork, CNN_Duplex
from dotenv import load_dotenv
import os

import sys
print("PYTHONPATH:", sys.path)
print("Current Directory:", os.getcwd())


# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

def test_chaotic_system_initialization():
    key = os.getenv('KEY').encode('utf-8')
    chaotic_system = ChaoticSystem(key)
    assert chaotic_system.state is not None
    print("ChaoticSystem initialized correctly with state:", chaotic_system.state)

def test_skew_tent_map():
    key = os.getenv('KEY').encode('utf-8')
    chaotic_system = ChaoticSystem(key)
    x = 123456789
    q = 987654321
    result = chaotic_system.skew_tent_map(x, q)
    assert isinstance(result, int)
    print("skew_tent_map function works correctly with result:", result)

def test_dpwlcm_map():
    key = os.getenv('KEY').encode('utf-8')
    chaotic_system = ChaoticSystem(key)
    x = 123456789
    q = 987654321
    result = chaotic_system.dpwlcm_map(x, q)
    assert isinstance(result, int)
    print("dpwlcm_map function works correctly with result:", result)

def test_generate_keystream():
    key = os.getenv('KEY').encode('utf-8')
    chaotic_system = ChaoticSystem(key)
    keystream = chaotic_system.generate_keystream(10)
    assert len(keystream) == 10
    assert all(isinstance(k, int) for k in keystream)
    print("Keystream generated correctly:", keystream)

def test_chaotic_neural_network_initialization():
    key = os.getenv('KEY').encode('utf-8')
    cnn = ChaoticNeuralNetwork(key)
    assert len(cnn.input_weights) == 16
    assert len(cnn.output_weights) == 4
    assert len(cnn.biases) == 4
    assert len(cnn.q_parameters) == 8
    print("ChaoticNeuralNetwork initialized correctly.")

def test_activation_function():
    key = os.getenv('KEY').encode('utf-8')
    cnn = ChaoticNeuralNetwork(key)
    inputs = [1, 2, 3, 4]
    q1 = cnn.q_parameters[0]
    q2 = cnn.q_parameters[1]
    bias = cnn.biases[0]
    result = cnn.activation_function(inputs, q1, q2, bias)
    assert isinstance(result, int)
    print("Activation function works correctly with result:", result)

def test_cnn_duplex_encryption_decryption():
    key = os.getenv('KEY').encode('utf-8')
    iv = os.getenv('IV').encode('utf-8')
    cnn_duplex = CNN_Duplex(key, iv)
    associated_data = "Informations d'en-tête".encode('utf-8')
    plaintext = "Message confidentiel à chiffrer.".encode('utf-8')
    ciphertext, tag = cnn_duplex.encrypt(plaintext, associated_data)

    # Réinitialiser l'état pour le déchiffrement
    cnn_duplex_dec = CNN_Duplex(key, iv)
    decrypted_plaintext = cnn_duplex_dec.decrypt(ciphertext, associated_data, tag)

    assert plaintext == decrypted_plaintext
    print("Encryption and decryption work correctly.")
    print("Plaintext:", plaintext)
    print("Ciphertext:", ciphertext)
    print("Decrypted Plaintext:", decrypted_plaintext)

if __name__ == "__main__":
    pytest.main()
