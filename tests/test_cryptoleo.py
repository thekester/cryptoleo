import pytest
from cryptoleo.cryptoleoretry import ChaoticSystem, ChaoticNeuralNetwork, CNN_Duplex, generate_and_overwrite_env
from dotenv import load_dotenv
import os
from pathlib import Path

@pytest.fixture(scope='module')
def setup_env():
    """
    Fixture pour charger les variables d'environnement KEY et IV et les convertir en bytes.
    Si le fichier .env n'existe pas ou contient des valeurs invalides, génère un nouveau fichier .env.
    """
    dotenv_path = Path(__file__).resolve().parent.parent / 'cryptoleo' / '.env'
    
    # Vérifier si le fichier .env existe
    if not dotenv_path.exists():
        print(f"Fichier .env non trouvé à {dotenv_path}. Génération d'un nouveau fichier .env.")
        key, iv = generate_and_overwrite_env(env_path=dotenv_path)
    else:
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Chargement des variables d'environnement depuis {dotenv_path}")
        
        key_hex = os.getenv('KEY')
        iv_hex = os.getenv('IV')
        print(f"KEY hex: {key_hex}")
        print(f"IV hex: {iv_hex}")
        
        # Vérifier si KEY et IV sont valides hex
        valid = True
        if key_hex is None or iv_hex is None:
            valid = False
            print("KEY ou IV manquant dans le fichier .env.")
        else:
            try:
                key = bytes.fromhex(key_hex)
                iv = bytes.fromhex(iv_hex)
            except ValueError as e:
                valid = False
                print(f"Erreur de conversion hex dans le fichier .env : {e}")
        
        if not valid:
            print("Génération de nouvelles clés et IV car les existantes sont invalides.")
            key, iv = generate_and_overwrite_env(env_path=dotenv_path)
        else:
            print("Clé et IV valides dans le fichier .env.")
    
    # Charger les variables d'environnement après éventuelle génération
    load_dotenv(dotenv_path=dotenv_path)
    key_hex = os.getenv('KEY')
    iv_hex = os.getenv('IV')
    key = bytes.fromhex(key_hex)
    iv = bytes.fromhex(iv_hex)
    
    return key, iv

def test_chaotic_system_initialization(setup_env):
    key, iv = setup_env
    chaotic_system = ChaoticSystem(key, iv)

    assert chaotic_system.state_a is not None
    assert chaotic_system.state_b is not None
    assert chaotic_system.state_c is not None
    assert chaotic_system.state_d is not None
    assert chaotic_system.state_e is not None
    assert chaotic_system.state_f is not None
    assert chaotic_system.state_g is not None
    assert chaotic_system.state_h is not None
    print("ChaoticSystem initialisé correctement avec les états :", 
          chaotic_system.state_a, chaotic_system.state_b, chaotic_system.state_c,
          chaotic_system.state_d, chaotic_system.state_e, chaotic_system.state_f,
          chaotic_system.state_g, chaotic_system.state_h)

def test_skew_tent_map(setup_env):
    key, iv = setup_env
    chaotic_system = ChaoticSystem(key, iv)

    x = 0.123456789  # Valeur normalisée entre 0 et 1
    result = chaotic_system.skew_tent_map(x)
    assert isinstance(result, float), "Le résultat de skew_tent_map doit être un float."
    print("skew_tent_map fonctionne correctement avec le résultat :", result)

def test_dpwlcm_map(setup_env):
    key, iv = setup_env
    chaotic_system = ChaoticSystem(key, iv)

    x = 0.123456789  # Valeur normalisée entre 0 et 1
    result = chaotic_system.dpwlcm_map(x)
    assert isinstance(result, float), "Le résultat de dpwlcm_map doit être un float."
    print("dpwlcm_map fonctionne correctement avec le résultat :", result)

def test_generate_keystream(setup_env):
    key, iv = setup_env
    chaotic_system = ChaoticSystem(key, iv)

    keystream_length = 10
    keystream = chaotic_system.generate_keystream(keystream_length)
    assert isinstance(keystream, bytes), "generate_keystream doit retourner un objet bytes."
    assert len(keystream) == keystream_length, f"Le keystream doit avoir une longueur de {keystream_length} bytes."
    print("Keystream généré correctement :", keystream.hex())

def test_chaotic_neural_network_initialization(setup_env):
    key, iv = setup_env
    chaotic_system = ChaoticSystem(key, iv)

    cnn = ChaoticNeuralNetwork(chaotic_system)

    assert len(cnn.input_weights) == 32 * 8, f"input_weights doit avoir une longueur de {32 * 8} bytes."
    assert len(cnn.output_weights) == 8 * 8, f"output_weights doit avoir une longueur de {8 * 8} bytes."
    assert len(cnn.biases) == 8 * 8, f"biases doit avoir une longueur de {8 * 8} bytes."
    assert len(cnn.q_parameters) == 16 * 8, f"q_parameters doit avoir une longueur de {16 * 8} bytes."
    print("ChaoticNeuralNetwork initialisé correctement.")

def test_activation_function(setup_env):
    key, iv = setup_env
    chaotic_system = ChaoticSystem(key, iv)
    cnn = ChaoticNeuralNetwork(chaotic_system)

    inputs = [1, 2, 3, 4]
    # Extraire les premiers paramètres Q et bias
    q1 = int.from_bytes(cnn.q_parameters[:8], 'big')
    q2 = int.from_bytes(cnn.q_parameters[8:16], 'big')
    bias = int.from_bytes(cnn.biases[:8], 'big')

    result = cnn.activation_function(inputs, q1, q2, bias)
    assert isinstance(result, int), "Le résultat de activation_function doit être un int."
    print("Activation function fonctionne correctement avec le résultat :", result)

def test_cnn_duplex_encryption_decryption(setup_env):
    key, iv = setup_env
    cnn_duplex_enc = CNN_Duplex(key, iv)

    associated_data = "Header Information".encode('utf-8')
    plaintext = "Confidential message to encrypt.".encode('utf-8')

    print("\n--- Encryption ---")
    ciphertext, tag = cnn_duplex_enc.encrypt(plaintext, associated_data)
    print("Ciphertext hex:", ciphertext.hex())
    print("Tag hex:", tag.hex())
    print("État interne après encryption:", cnn_duplex_enc.state)

    # Initialiser une nouvelle instance pour le déchiffrement avec la même clé et IV
    cnn_duplex_dec = CNN_Duplex(key, iv)

    print("\n--- Decryption ---")
    decrypted_plaintext = cnn_duplex_dec.decrypt(ciphertext, associated_data, tag)
    print("Decrypted Plaintext:", decrypted_plaintext)
    print("État interne après décryption:", cnn_duplex_dec.state)

    # Comparer les états internes pour s'assurer qu'ils sont synchronisés
    assert cnn_duplex_enc.state == cnn_duplex_dec.state, "Les états internes ne sont pas synchronisés entre chiffrement et déchiffrement."

    assert plaintext == decrypted_plaintext, "Le message décrypté ne correspond pas au message original."
    print("Chiffrement et déchiffrement fonctionnent correctement.")
    print("Plaintext :", plaintext)
    print("Ciphertext :", ciphertext.hex())
    print("Tag :", tag.hex())
    print("Decrypted Plaintext :", decrypted_plaintext)

if __name__ == "__main__":
    pytest.main()
