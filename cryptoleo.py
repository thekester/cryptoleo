import numpy as np
from hashlib import sha256

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
        print(f"skew_tent_map: x={x}, q={q}, result={result}")
        return result

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
        print(f"dpwlcm_map: x={x}, q={q}, result={result}")
        return result

    def generate_keystream(self, length):
        keystream = []
        x = self.state
        # Extraction des paramètres q1 et q2 à partir de la clé
        q1 = (int.from_bytes(self.key[:4], 'big') % (2**32 - 1)) + 1  # Assure que q1 ∈ [1, 2^32 - 1]
        q2 = (int.from_bytes(self.key[4:8], 'big') % (2**32 - 1)) + 1  # Assure que q2 ∈ [1, 2^32 - 1]
        print(f"Generating keystream with q1={q1}, q2={q2}")
        for i in range(length):
            # Application des fonctions chaotiques
            x = self.skew_tent_map(x, q1)
            x = self.dpwlcm_map(x, q2)
            keystream_value = x & 0xFFFFFFFF  # On garde les 32 bits de poids faible
            keystream.append(keystream_value)
            print(f"Keystream[{i}] = {keystream_value}")
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
        # Fonction d'activation utilisant les cartes chaotiques
        x = sum(inputs) + bias
        print(f"Activation function inputs sum: {x}")
        y1 = self.chaotic_system.skew_tent_map(x, q1)
        y2 = self.chaotic_system.dpwlcm_map(x, q2)
        result = y1 ^ y2  # Opération XOR des deux sorties chaotiques
        print(f"Activation function result: {result}")
        return result

    def process(self, data_block):
        print(f"Processing data block: {data_block}")
        # Traitement d'un bloc de données par le réseau de neurones
        outputs = []
        for i in range(0, len(data_block), 4):
            inputs = data_block[i:i+4]
            if len(inputs) < 4:
                inputs += [0] * (4 - len(inputs))  # Remplissage si le bloc est incomplet
            print(f"Processing inputs: {inputs}")
            # Sélection des paramètres q et du biais pour le neurone courant
            q1 = self.q_parameters[(i // 4) % 4]
            q2 = self.q_parameters[((i // 4) % 4) + 4]
            bias = self.biases[(i // 4) % 4]
            # Pondération des entrées
            weighted_inputs = [inp * w for inp, w in zip(inputs, self.input_weights[i:i+4])]
            print(f"Weighted inputs: {weighted_inputs}")
            # Calcul de la sortie du neurone
            neuron_output = self.activation_function(weighted_inputs, q1, q2, bias)
            outputs.append(neuron_output)
        # Traitement avec des fonctions non linéaires
        final_output = self.non_linear_processing(outputs)
        print(f"Final output after non-linear processing: {final_output}")
        return final_output

    def non_linear_processing(self, outputs):
        # Implémentation simplifiée des fonctions non linéaires pour mélanger les sorties
        h = []
        for i in range(len(outputs)):
            # Opérations de rotation et XOR pour augmenter la diffusion
            t1 = ((outputs[i] >> 6) | (outputs[i] << (32 - 6))) & 0xFFFFFFFF
            t2 = ((outputs[(i+1)%len(outputs)] >> 11) | (outputs[(i+1)%len(outputs)] << (32 - 11))) & 0xFFFFFFFF
            h_value = outputs[i] ^ t1 ^ t2
            h.append(h_value)
            print(f"Non-linear processing [{i}]: h = {h_value}")
        return h

class CNN_Duplex:
    def __init__(self, key, iv, r=256, c=256):
        print("Initializing CNN_Duplex...")
        self.key = key
        self.iv = iv
        self.state = iv  # État interne initialisé avec l'IV
        self.r = r  # Taille du bitrate
        self.c = c  # Taille de la capacité
        self.cnn = ChaoticNeuralNetwork(key)
        print("CNN_Duplex initialized.")

    def absorb(self, data):
        print(f"Absorbing data: {data}")
        # Absorption des données dans l'état interne
        padded_data = self.pad(data)
        print(f"Padded data: {padded_data}")
        for block in self.chunk_data(padded_data, self.r // 8):
            # Conversion du bloc en entiers de 32 bits
            block_ints = [int.from_bytes(block[i:i+4], 'big') for i in range(0, len(block), 4)]
            print(f"Block ints: {block_ints}")
            processed_block = self.cnn.process(block_ints)
            # Mise à jour de l'état interne
            self.state = self.state_update(self.state, processed_block)
            print(f"State updated to: {self.state}")

    def squeeze(self, length):
        print(f"Squeezing keystream of length: {length}")
        # Extraction du keystream à partir de l'état interne
        output = b''
        while len(output) < length:
            # Traitement de l'état interne pour générer le keystream
            state_ints = [int.from_bytes(self.state[i:i+4], 'big') for i in range(0, len(self.state), 4)]
            print(f"State ints: {state_ints}")
            block = self.cnn.process(state_ints)
            output_block = b''.join([int.to_bytes(word, 4, 'big') for word in block])
            output += output_block
            # Mise à jour de l'état interne
            self.state = self.state_update(self.state, block)
            print(f"Keystream generated: {output_block}")
            print(f"State updated to: {self.state}")
        return output[:length]

    def encrypt(self, plaintext, associated_data):
        print("Starting encryption...")
        # Absorption des données associées
        self.absorb(associated_data)
        # Chiffrement du texte en clair
        ciphertext = b''
        for block in self.chunk_data(plaintext, self.r // 8):
            keystream = self.squeeze(len(block))
            ciphertext_block = bytes(a ^ b for a, b in zip(block, keystream))
            ciphertext += ciphertext_block
            print(f"Plaintext block: {block}")
            print(f"Keystream: {keystream}")
            print(f"Ciphertext block: {ciphertext_block}")
            # Absorption du bloc chiffré pour l'authentification
            self.absorb(ciphertext_block)
        # Génération du tag d'authentification
        tag = self.squeeze(32)  # Tag de 256 bits (32 octets)
        print(f"Authentication tag generated: {tag}")
        print("Encryption completed.")
        return ciphertext, tag

    def decrypt(self, ciphertext, associated_data, tag):
        print("Starting decryption...")
        # Absorption des données associées
        self.absorb(associated_data)
        # Déchiffrement du texte chiffré
        plaintext = b''
        for block in self.chunk_data(ciphertext, self.r // 8):
            keystream = self.squeeze(len(block))
            plaintext_block = bytes(a ^ b for a, b in zip(block, keystream))
            plaintext += plaintext_block
            print(f"Ciphertext block: {block}")
            print(f"Keystream: {keystream}")
            print(f"Plaintext block: {plaintext_block}")
            # Absorption du bloc chiffré pour l'authentification
            self.absorb(block)
        # Vérification du tag d'authentification
        computed_tag = self.squeeze(32)
        print(f"Computed authentication tag: {computed_tag}")
        if computed_tag != tag:
            print("Authentication failed: tags do not match.")
            raise ValueError("Échec de l'authentification : le tag ne correspond pas.")
        print("Decryption completed successfully.")
        return plaintext

    def pad(self, data):
        # Padding avec un bit '1' suivi de bits '0' pour aligner sur la taille du bloc
        pad_len = (-len(data) - 1) % (self.r // 8)
        padded_data = data + b'\x80' + b'\x00' * pad_len
        print(f"Data after padding: {padded_data}")
        return padded_data

    def chunk_data(self, data, chunk_size):
        # Divise les données en morceaux de taille chunk_size
        for i in range(0, len(data), chunk_size):
            yield data[i:i+chunk_size]

    def state_update(self, state, processed_block):
        # Mise à jour de l'état interne en combinant avec le bloc traité
        state_ints = [int.from_bytes(state[i:i+4], 'big') for i in range(0, len(state), 4)]
        new_state_ints = [(s ^ p) & 0xFFFFFFFF for s, p in zip(state_ints, processed_block)]
        new_state = b''.join([int.to_bytes(word, 4, 'big') for word in new_state_ints])
        return new_state

# Exemple d'utilisation
if __name__ == "__main__":
    # Clé secrète de 16 octets (128 bits)
    key = "clefsecrete123456".encode('utf-8')
    # Vecteur d'initialisation (IV) de 16 octets
    iv = "vecteurinitialiv1".encode('utf-8')
    # Initialisation du schéma de chiffrement
    cnn_duplex = CNN_Duplex(key, iv)
    # Données associées (données supplémentaires à authentifier mais non chiffrées)
    associated_data = "Informations d'en-tête".encode('utf-8')
    # Texte en clair à chiffrer
    plaintext = "Message confidentiel à chiffrer.".encode('utf-8')

    # Chiffrement
    ciphertext, tag = cnn_duplex.encrypt(plaintext, associated_data)
    print("\nRésultats du chiffrement:")
    print("Texte chiffré :", ciphertext)
    print("Tag d'authentification :", tag)

    # Déchiffrement
    try:
        # Réinitialisation du schéma pour le déchiffrement (nécessaire car l'état a changé)
        cnn_duplex_dec = CNN_Duplex(key, iv)
        decrypted_plaintext = cnn_duplex_dec.decrypt(ciphertext, associated_data, tag)
        print("\nRésultats du déchiffrement:")
        print("Texte déchiffré :", decrypted_plaintext.decode('utf-8'))
    except ValueError as e:
        print(e)
