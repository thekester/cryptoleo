import numpy as np
from hashlib import sha256
from dotenv import load_dotenv
import os
from tqdm import tqdm  # Importation de tqdm pour la barre de progression

class ChaoticSystem:
    def __init__(self, key):
        self.key = key
        # Initialisation de l'état avec un haché de la clé
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
        return result % (2**32)  # Limiter la taille du résultat

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
        return result % (2**32)  # Limiter la taille du résultat

    def generate_keystream(self, length):
        keystream = []
        x = self.state
        # Extraction des paramètres q1 et q2 à partir de la clé
        q1 = (int.from_bytes(self.key[:4], 'big') % (2**32 - 1)) + 1
        q2 = (int.from_bytes(self.key[4:8], 'big') % (2**32 - 1)) + 1
        print(f"Generating keystream with q1={q1}, q2={q2}")
        # Afficher la barre de progression si la longueur est significative
        if length >= 1000:
            iterator = tqdm(range(length), desc="Génération du keystream ChaoticSystem")
        else:
            iterator = range(length)
        for i in iterator:
            # Application des fonctions chaotiques
            x = self.skew_tent_map(x, q1)
            x = self.dpwlcm_map(x, q2)
            keystream_value = x & 0xFFFFFFFF
            keystream.append(keystream_value)
            # Vous pouvez décommenter la ligne suivante pour afficher des informations périodiques
            # if i % 100000 == 0:
            #     print(f"Keystream[{i}] = {keystream_value}")
        self.state = x  # Mise à jour de l'état interne
        print(f"ChaoticSystem state updated to: {self.state}")
        return keystream

class ChaoticNeuralNetwork:
    def __init__(self, key):
        print("Initializing Chaotic Neural Network...")
        self.chaotic_system = ChaoticSystem(key)
        # Génération des poids et des biais pour le réseau de neurones
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
            # Limiter les poids pour éviter les nombres trop grands
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
        with tqdm(total=total_blocks, desc="Génération du keystream CNN_Duplex") as pbar:
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

# Exemple d'utilisation
if __name__ == "__main__":
    # Importer dotenv pour charger les variables d'environnement
    from dotenv import load_dotenv
    import os

    # Charger les variables d'environnement depuis le fichier .env
    load_dotenv()

    # Clé secrète de 16 octets (128 bits) chargée depuis .env
    key_env = os.getenv('KEY')
    iv_env = os.getenv('IV')

    if key_env is None or iv_env is None:
        raise ValueError("La clé ou l'IV n'a pas été trouvé dans le fichier .env")

    key = key_env.encode('utf-8')
    iv = iv_env.encode('utf-8')

    # Initialisation du schéma de chiffrement
    cnn_duplex = CNN_Duplex(key, iv)

    # Générer un keystream de 1 Mo et le sauvegarder dans 'keystream.bin'
    keystream_length = 1 * 1024 * 1024  # 1 mégaoctet
    # Avant de générer, on s'assure que le fichier 'keystream.bin' est vide ou le supprime s'il existe
    if os.path.exists('keystream.bin'):
        os.remove('keystream.bin')

    print("Génération du keystream...")
    # Le keystream est généré via la méthode squeeze de CNN_Duplex
    cnn_duplex.squeeze(keystream_length, save_keystream=True)
    print("Keystream generated and saved to 'keystream.bin'")
