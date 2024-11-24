import numpy as np
from hashlib import sha256, sha3_512
import hmac
from dotenv import load_dotenv, set_key
import os
from tqdm import tqdm
import secrets  # Pour générer des octets aléatoires sécurisés
from Crypto.Cipher import AES  # Pour des fonctions de chiffrement robustes
from Crypto.Util import Counter

class ChaoticSystem:
    def __init__(self, key, iv):
        self.key = key
        self.iv = iv
        if len(self.key) < 16:
            raise ValueError("La clé doit contenir au moins 16 octets.")
        # Initialiser les états comme des entiers basés sur SHA-3-512 pour un état plus large (512 bits)
        key_hash = sha3_512(key).digest()
        iv_hash = sha3_512(iv).digest()
        self.state_a = int.from_bytes(key_hash[:8], 'big') % (1 << 64)
        self.state_b = int.from_bytes(key_hash[8:16], 'big') % (1 << 64)
        self.state_c = int.from_bytes(iv_hash[:8], 'big') % (1 << 64)
        self.state_d = int.from_bytes(iv_hash[8:16], 'big') % (1 << 64)
        self.state_e = int.from_bytes(iv_hash[16:24], 'big') % (1 << 64)
        self.state_f = int.from_bytes(iv_hash[24:32], 'big') % (1 << 64)
        self.state_g = int.from_bytes(iv_hash[32:40], 'big') % (1 << 64)
        self.state_h = int.from_bytes(iv_hash[40:48], 'big') % (1 << 64)
        print(f"ChaoticSystem initialisé avec les états : a={self.state_a}, b={self.state_b}, c={self.state_c}, d={self.state_d}, e={self.state_e}, f={self.state_f}, g={self.state_g}, h={self.state_h}")
        self.boost_entropy()
        # Générer une clé supplémentaire pour HMAC
        self.mixer_key = secrets.token_bytes(32)  # 256 bits
        print(f"Mixer key générée : {self.mixer_key.hex()}")

    def boost_entropy(self):
        additional_entropy = secrets.token_bytes(32)  # Augmentation de l'entropie
        mixed_entropy = sha3_512(additional_entropy).digest()
        self.state_a ^= int.from_bytes(mixed_entropy[:8], 'big') & 0xFFFFFFFFFFFFFFFF
        self.state_b ^= int.from_bytes(mixed_entropy[8:16], 'big') & 0xFFFFFFFFFFFFFFFF
        self.state_c ^= int.from_bytes(mixed_entropy[16:24], 'big') & 0xFFFFFFFFFFFFFFFF
        self.state_d ^= int.from_bytes(mixed_entropy[24:32], 'big') & 0xFFFFFFFFFFFFFFFF
        self.state_e ^= int.from_bytes(mixed_entropy[32:40], 'big') & 0xFFFFFFFFFFFFFFFF
        self.state_f ^= int.from_bytes(mixed_entropy[40:48], 'big') & 0xFFFFFFFFFFFFFFFF
        self.state_g ^= int.from_bytes(mixed_entropy[48:56], 'big') & 0xFFFFFFFFFFFFFFFF
        self.state_h ^= int.from_bytes(mixed_entropy[56:64], 'big') & 0xFFFFFFFFFFFFFFFF
        print("Entropie boostée.")

    def lorenz_map(self, x, y, z, sigma=10.0, rho=28.0, beta=8/3):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return x + dx * 0.01, y + dy * 0.01, z + dz * 0.01  # 0.01 est le pas de temps

    def henon_map(self, x, y, a=1.4, b=0.3):
        new_x = 1 - a * x**2 + y
        new_y = b * x
        return new_x, new_y

    def rosenbrock_map(self, x, y, a=1.0, b=100.0):
        new_x = (1 - x) + a * x * (y - x**2)
        new_y = b * (x - 1)**2
        return new_x, new_y

    def logistic_map(self, x, r=3.99):
        return r * x * (1 - x)

    def skew_tent_map(self, x, mu=1.99):
        if x < 0.5:
            return mu * x
        else:
            return mu * (1 - x)

    def dpwlcm_map(self, x, a=3.99):
        return a * x * (1 - x)

    def nonlinear_transform(self, value):
        # Rotation gauche de 13 bits
        value = ((value << 13) | (value >> (64 - 13))) & 0xFFFFFFFFFFFFFFFF
        value ^= (value >> 7)
        value ^= (value << 17) & 0xFFFFFFFFFFFFFFFF
        # Transformation supplémentaire
        value = ((value * 0xDEADBEEFDEADBEEF) + 0xCAFEBABECAFEBABE) & 0xFFFFFFFFFFFFFFFF
        # Ajout de permutations de bits
        value = ((value & 0xF0F0F0F0F0F0F0F0) >> 4) | ((value & 0x0F0F0F0F0F0F0F0F) << 4)
        value = ((value & 0xCCCCCCCCCCCCCCCC) >> 2) | ((value & 0x3333333333333333) << 2)
        return value

    def complex_nonlinear_transform(self, value):
        # Rotation gauche de 13 bits
        value = ((value << 13) | (value >> (64 - 13))) & 0xFFFFFFFFFFFFFFFF
        value ^= (value >> 7)
        value ^= (value << 17) & 0xFFFFFFFFFFFFFFFF
        return value

    def normalize(self, value, bits=64):
        # Normaliser la valeur entre 0 et 1
        return value / float(1 << bits)

    def secure_mix(self, combined_state):
        # Utiliser HMAC-SHA256 pour le mélange sécurisé
        return hmac.new(self.mixer_key, combined_state, sha256).digest()

    def secure_mix_with_AES(self, combined_state):
        # Utiliser AES en mode CTR pour le mélange sécurisé
        ctr = Counter.new(128)
        cipher = AES.new(self.key, AES.MODE_CTR, counter=ctr)
        # Assurez-vous que combined_state est un multiple de 16 bytes pour AES
        if len(combined_state) % 16 != 0:
            combined_state = combined_state.ljust((len(combined_state) // 16 + 1) * 16, b'\0')
        return cipher.encrypt(combined_state)

    def generate_keystream(self, length, iterations=512, feedback_interval=100000):
        keystream = bytearray()
        a = self.normalize(self.state_a, 64)
        b = self.normalize(self.state_b, 64)
        c = self.normalize(self.state_c, 64)
        d = self.normalize(self.state_d, 64)
        e = self.normalize(self.state_e, 64)
        f = self.normalize(self.state_f, 64)
        g = self.normalize(self.state_g, 64)
        h = self.normalize(self.state_h, 64)
        # Générer q1 et q2 comme entiers dans [0, 2^64)
        q1 = int.from_bytes(self.key[:8], 'big') % (1 << 64)
        q2 = int.from_bytes(self.key[8:16], 'big') % (1 << 64)
        # Incorporer l'IV
        iv_int = int.from_bytes(self.iv, 'big') % (1 << 64)
        print(f"Génération du keystream avec q1={q1}, q2={q2}, IV={iv_int}")
        iterator = tqdm(range(length), desc="Génération du keystream ChaoticSystem") if length >= 1000 else range(length)
        
        for count, _ in enumerate(iterator, 1):
            for _ in range(iterations):
                a, b, c = self.lorenz_map(a, b, c)
                d, e = self.henon_map(d, e)
                f, g = self.rosenbrock_map(f, g)
                h = self.skew_tent_map(h)
                a = self.logistic_map(a)  # Interaction plus profonde
                # Mettre à jour les états avec IV_int pour plus de diffusion
                a = (a + self.normalize(iv_int, 64)) % 1.0
                b = (b + self.normalize(iv_int, 64)) % 1.0
                c = (c + self.normalize(iv_int, 64)) % 1.0
                d = (d + self.normalize(iv_int, 64)) % 1.0
                e = (e + self.normalize(iv_int, 64)) % 1.0
                f = (f + self.normalize(iv_int, 64)) % 1.0
                g = (g + self.normalize(iv_int, 64)) % 1.0
                h = (h + self.normalize(iv_int, 64)) % 1.0
            # Mélanger l'état chaotique avec HMAC-SHA256 pour augmenter l'aléatoire
            a_int = int(a * (1 << 64)) % (1 << 64)
            b_int = int(b * (1 << 64)) % (1 << 64)
            c_int = int(c * (1 << 64)) % (1 << 64)
            d_int = int(d * (1 << 64)) % (1 << 64)
            e_int = int(e * (1 << 64)) % (1 << 64)
            f_int = int(f * (1 << 64)) % (1 << 64)
            g_int = int(g * (1 << 64)) % (1 << 64)
            h_int = int(h * (1 << 64)) % (1 << 64)
            combined_state = (
                a_int.to_bytes(8, 'big') +
                b_int.to_bytes(8, 'big') +
                c_int.to_bytes(8, 'big') +
                d_int.to_bytes(8, 'big') +
                e_int.to_bytes(8, 'big') +
                f_int.to_bytes(8, 'big') +
                g_int.to_bytes(8, 'big') +
                h_int.to_bytes(8, 'big')
            )
            # Utiliser HMAC-SHA256 pour le mélange sécurisé
            mixed = self.secure_mix(combined_state)
            keystream += mixed
            # Feedback: Incorporate part of the keystream into the state
            if count % feedback_interval == 0:
                feedback = int.from_bytes(mixed[:16], 'big') % (1 << 64)  # Utiliser 16 octets pour plus d'entropie
                feedback = self.nonlinear_transform(feedback)
                self.state_a = (self.state_a ^ feedback) & 0xFFFFFFFFFFFFFFFF
                self.state_b = (self.state_b ^ feedback) & 0xFFFFFFFFFFFFFFFF
                self.state_c = (self.state_c ^ feedback) & 0xFFFFFFFFFFFFFFFF
                self.state_d = (self.state_d ^ feedback) & 0xFFFFFFFFFFFFFFFF
                self.state_e = (self.state_e ^ feedback) & 0xFFFFFFFFFFFFFFFF
                self.state_f = (self.state_f ^ feedback) & 0xFFFFFFFFFFFFFFFF
                self.state_g = (self.state_g ^ feedback) & 0xFFFFFFFFFFFFFFFF
                self.state_h = (self.state_h ^ feedback) & 0xFFFFFFFFFFFFFFFF
                print("États mis à jour avec le feedback du keystream.")
        # Mettre à jour les états internes avec les nouvelles valeurs chaotiques
        self.state_a = int(a * (1 << 64)) % (1 << 64)
        self.state_b = int(b * (1 << 64)) % (1 << 64)
        self.state_c = int(c * (1 << 64)) % (1 << 64)
        self.state_d = int(d * (1 << 64)) % (1 << 64)
        self.state_e = int(e * (1 << 64)) % (1 << 64)
        self.state_f = int(f * (1 << 64)) % (1 << 64)
        self.state_g = int(g * (1 << 64)) % (1 << 64)
        self.state_h = int(h * (1 << 64)) % (1 << 64)
        print(f"États de ChaoticSystem mis à jour : a={self.state_a}, b={self.state_b}, c={self.state_c}, d={self.state_d}, e={self.state_e}, f={self.state_f}, g={self.state_g}, h={self.state_h}")
        return bytes(keystream[:length])

class ChaoticNeuralNetwork:
    def __init__(self, chaotic_system):
        print("Initialisation du Chaotic Neural Network...")
        self.chaotic_system = chaotic_system
        # Générer les poids, biais et paramètres Q comme listes d'entiers dans [0, 2^64)
        self.input_weights = self.chaotic_system.generate_keystream(32 * 8)  # 32 poids de 64 bits
        print(f"Poids d'entrée générés.")
        self.output_weights = self.chaotic_system.generate_keystream(8 * 8)  # 8 poids de sortie de 64 bits
        print(f"Poids de sortie générés.")
        self.biases = self.chaotic_system.generate_keystream(8 * 8)  # 8 biais de 64 bits
        print(f"Biais générés.")
        self.q_parameters = self.chaotic_system.generate_keystream(16 * 8)  # 16 paramètres Q de 64 bits
        print(f"Paramètres Q générés.")
        print("Chaotic Neural Network initialisé.")
    
    def activation_function(self, inputs, q1, q2, bias):
        """
        Fonction d'activation combinant les cartes chaotiques.
        """
        x = (sum(inputs) + bias) % (1 << 64)  # Inputs et biais sont des entiers dans [0, 2^64)
        print(f"Somme des inputs (mod 2^64) : {x}")
        # Normaliser x pour les cartes chaotiques
        x_norm = self.chaotic_system.normalize(x, 64)
        y1 = self.chaotic_system.skew_tent_map(x_norm)
        y2 = self.chaotic_system.dpwlcm_map(x_norm)
        # Combiner y1 et y2 en addition modulo 2^64
        result = int((y1 + y2) * (1 << 64)) % (1 << 64)
        result = self.chaotic_system.nonlinear_transform(result)
        print(f"Résultat de la fonction d'activation : {result}")
        return result
    
    def process(self, data_block):
        """
        Traiter un bloc de données en appliquant les fonctions d'activation.
        """
        print(f"Traitement du bloc de données : {data_block}")
        outputs = []
        data_list = list(data_block)  # Supposons que data_block est une liste d'entiers dans [0, 2^64)
        for i in range(0, len(data_list), 4):
            inputs = data_list[i:i+4]
            if len(inputs) < 4:
                inputs += [0] * (4 - len(inputs))  # Remplir avec des zéros si nécessaire
            print(f"Traitement des inputs : {inputs}")
            # Extraire les paramètres Q dynamiques
            q1 = int.from_bytes(
                self.q_parameters[i % len(self.q_parameters)*8:(i % len(self.q_parameters)*8)+8], 
                'big'
            )
            q2 = int.from_bytes(
                self.q_parameters[(i+8) % len(self.q_parameters)*8:(i+8) % len(self.q_parameters)*8+8], 
                'big'
            )
            bias = int.from_bytes(
                self.biases[i % len(self.biases)*8:(i % len(self.biases)*8)+8], 
                'big'
            )
            # Appliquer les poids
            weighted_inputs = [
                (inp * int.from_bytes(self.input_weights[i + j*8:i + (j+1)*8], 'big')) % (1 << 64)
                for j, inp in enumerate(inputs)
            ]
            print(f"Inputs pondérés : {weighted_inputs}")
            neuron_output = self.activation_function(weighted_inputs, q1, q2, bias)
            outputs.append(neuron_output)
        final_output = self.non_linear_processing(outputs)
        print(f"Sortie finale après traitement non-linéaire : {final_output}")
        return final_output
    
    def non_linear_processing(self, outputs):
        """
        Appliquer des fonctions non-linéaires supplémentaires pour augmenter la diffusion.
        """
        h = []
        for i in range(len(outputs)):
            # Utiliser des constantes plus grandes et variées
            t1 = (outputs[i] + 0x1A2B3C4D5E6F7081) % (1 << 64)
            t2 = (outputs[(i+1)%len(outputs)] + 0x5E6F7A8B9C0D1E2F) % (1 << 64)
            h_value = (outputs[i] ^ t1 ^ t2) % (1 << 64)
            h_value = self.chaotic_system.nonlinear_transform(h_value)
            print(f"Traitement non-linéaire [{i}] : h = {h_value}")
            h.append(h_value)
        return h

class CNN_Duplex:
    def __init__(self, key, iv, r=256, c=256):
        print("Initialisation de CNN_Duplex...")
        self.key = key
        self.iv = iv
        self.r = r
        self.c = c
        # Initialiser ChaoticSystem et le passer à ChaoticNeuralNetwork
        self.chaotic_system = ChaoticSystem(key, iv)
        self.cnn = ChaoticNeuralNetwork(self.chaotic_system)
        # Initialiser l'état interne
        self.state = iv
        print("CNN_Duplex initialisé.")
    
    def absorb(self, data):
        """
        Absorber les données associées (AD) ou le message (M) dans l'état interne.
        """
        print(f"Absorption des données : {data}")
        padded_data = self.pad(data)
        print(f"Données après padding : {padded_data}")
        for block in self.chunk_data(padded_data, self.r // 8):
            # Convertir les octets du bloc en entiers dans [0, 2^64)
            block_ints = [int.from_bytes(block[i:i+8], 'big') for i in range(0, len(block), 8)]
            print(f"Bloc en entiers : {block_ints}")
            processed_block = self.cnn.process(block_ints)
            # Mettre à jour l'état ; pour simplifier, convertir les entiers en octets
            processed_bytes = b''.join([word.to_bytes(8, 'big') for word in processed_block])
            self.state = self.state_update(self.state, processed_bytes)
            print(f"État mis à jour.")
    
    def squeeze(self, length, save_keystream=False):
        """
        Générer un keystream de la longueur spécifiée.
        """
        print(f"Génération du keystream de longueur : {length}")
        output = bytearray()
        state_length = len(self.state)
        total_blocks = (length + state_length - 1) // state_length
        iterator = tqdm(range(total_blocks), desc="Génération du keystream CNN_Duplex") if length >= 1000 else range(total_blocks)
        
        for _ in iterator:
            # Convertir les octets de l'état en entiers dans [0, 2^64)
            state_ints = [int.from_bytes(self.state[i:i+8], 'big') for i in range(0, len(self.state), 8)]
            print(f"État en entiers : {state_ints}")
            processed_block = self.cnn.process(state_ints)
            # Convertir les entiers traités en octets
            output_block = b''.join([word.to_bytes(8, 'big') for word in processed_block])
            output += output_block
            if save_keystream:
                with open('keystream.bin', 'ab') as f:
                    f.write(output_block)
            # Mettre à jour l'état
            self.state = self.state_update(self.state, output_block)
            print(f"Keystream généré.")
            if len(output) >= length:
                break
        if isinstance(iterator, tqdm):
            iterator.close()
        # Post-processing pour éliminer les motifs restants
        keystream_final = self.post_process_keystream(bytes(output[:length]))
        return keystream_final
    
    def post_process_keystream(self, keystream):
        """
        Appliquer un post-processing sécurisé au keystream.
        """
        # Utiliser HMAC-SHA256 pour le post-processing
        post_processed = hmac.new(self.key, keystream, sha256).digest()
        print("Keystream post-traité avec HMAC-SHA256.")
        return post_processed
    
    def pad(self, data):
        """
        Appliquer un padding conforme aux spécifications (PKCS7).
        """
        pad_len = (self.r // 8) - (len(data) % (self.r // 8))
        padded_data = data + bytes([pad_len] * pad_len)
        print(f"Données après padding : {padded_data}")
        return padded_data
    
    def chunk_data(self, data, chunk_size):
        """
        Diviser les données en chunks de taille spécifiée.
        """
        for i in range(0, len(data), chunk_size):
            yield data[i:i+chunk_size]
    
    def state_update(self, state, processed_block):
        """
        Mettre à jour l'état interne en appliquant une opération XOR avec le bloc traité.
        """
        state_ints = [int.from_bytes(state[i:i+8], 'big') for i in range(0, len(state), 8)]
        processed_ints = [int.from_bytes(processed_block[i:i+8], 'big') for i in range(0, len(processed_block), 8)]
        new_state_ints = [(s ^ p) & 0xFFFFFFFFFFFFFFFF for s, p in zip(state_ints, processed_ints)]
        new_state = b''.join([word.to_bytes(8, 'big') for word in new_state_ints])
        return new_state
    
    def generate_tag(self, ciphertext, associated_data):
        """
        Générer un tag d'authentification sécurisé en utilisant HMAC-SHA256.
        """
        tag_input = ciphertext + associated_data
        tag = hmac.new(self.key, tag_input, sha256).digest()[:16]  # Tag de 128 bits
        return tag
    
    def encrypt(self, plaintext, associated_data):
        """
        Crypter le plaintext avec les données associées et générer un tag d'authentification.
        """
        print("Début du chiffrement...")
        self.absorb(associated_data)
        keystream = self.squeeze(len(plaintext))
        ciphertext = bytes([p ^ k for p, k in zip(plaintext, keystream)])
        print(f"Ciphertext : {ciphertext.hex()}")
        tag = self.generate_tag(ciphertext, associated_data)
        print(f"Tag généré : {tag.hex()}")
        return ciphertext, tag
    
    def decrypt(self, ciphertext, associated_data, tag):
        """
        Décrypter le ciphertext et vérifier le tag d'authentification.
        """
        print("Début du déchiffrement...")
        self.absorb(associated_data)
        keystream = self.squeeze(len(ciphertext))
        plaintext = bytes([c ^ k for c, k in zip(ciphertext, keystream)])
        print(f"Plaintext décrypté : {plaintext}")
        expected_tag = self.generate_tag(ciphertext, associated_data)
        print(f"Tag attendu : {expected_tag.hex()}")
        if not hmac.compare_digest(tag, expected_tag):
            raise ValueError("Tag d'authentification invalide !")
        print("Vérification du tag réussie.")
        return plaintext

def generate_and_overwrite_env():
    """
    Génère des clés et IV sécurisés et les écrit dans le fichier .env, écrasant tout contenu existant.
    """
    # Générer une clé et un IV sécurisés
    key = secrets.token_bytes(32)  # 256 bits
    iv = secrets.token_bytes(16)   # 128 bits
    print(f"Clé générée : {key.hex()}")
    print(f"IV généré : {iv.hex()}")
    
    # Définir les chemins
    env_path = '.env'
    
    # Écrire ou écraser les clés dans .env
    set_key(env_path, 'KEY', key.hex())
    set_key(env_path, 'IV', iv.hex())
    print(f"Fichier .env mis à jour avec les nouvelles clés.")
    
    return key, iv

# Exemple d'utilisation
if __name__ == "__main__":
    # Générer de nouvelles clés et IV, écrasant les anciens dans .env
    key, iv = generate_and_overwrite_env()

    # Initialiser le schéma de chiffrement avec les nouvelles clés
    cnn_duplex_enc = CNN_Duplex(key, iv)

    # Générer un keystream de 1 Mo et le sauvegarder dans 'keystream.bin'
    keystream_length = 1 * 1024 * 1024  # 1 mégaoctet
    if os.path.exists('keystream.bin'):
        os.remove('keystream.bin')

    print("Génération du keystream...")
    keystream = cnn_duplex_enc.squeeze(keystream_length, save_keystream=True)
    print("Keystream généré et sauvegardé dans 'keystream.bin'")

    # Exemple de chiffrement
    message = "Message secret à chiffrer.".encode('utf-8')
    associated_data = "Données associées non chiffrées.".encode('utf-8')
    print("\n--- Chiffrement ---")
    ciphertext, tag = cnn_duplex_enc.encrypt(message, associated_data)
    print(f"Ciphertext : {ciphertext.hex()}")
    print(f"Tag : {tag.hex()}")

    # Initialiser une nouvelle instance pour le déchiffrement
    cnn_duplex_dec = CNN_Duplex(key, iv)

    # Exemple de déchiffrement
    print("\n--- Déchiffrement ---")
    try:
        decrypted_message = cnn_duplex_dec.decrypt(ciphertext, associated_data, tag)
        # Vérifiez si le décryptage est correct avant de décoder
        if decrypted_message == message:
            print(f"Decrypted Message : {decrypted_message.decode('utf-8')}")
        else:
            print("Le message décrypté ne correspond pas au message original.")
    except ValueError as e:
        print(e)
