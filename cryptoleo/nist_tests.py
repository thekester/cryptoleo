import numpy as np
import sys
import os

# Ajouter le dossier parent au chemin de recherche des modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from nistrng import *
from tqdm import tqdm  # Importation de tqdm pour les barres de progression

# Lire le keystream depuis le fichier 'keystream.bin'
with open('keystream.bin', 'rb') as f:
    keystream_bytes = f.read()

# Convertir le keystream en une séquence de bits signés
keystream_bits = np.unpackbits(np.frombuffer(keystream_bytes, dtype=np.uint8)).astype(np.int8)
print(f"Nombre total de bits dans le keystream : {len(keystream_bits)}")

# Vérifier l'éligibilité des tests pour la séquence donnée
eligible_battery = check_eligibility_all_battery(keystream_bits, SP800_22R1A_BATTERY)

# Afficher les tests éligibles
print("Tests éligibles du NIST SP800-22r1a :")
for test_name in eligible_battery.keys():
    print(f"- {test_name}")

# Exécuter tous les tests éligibles avec barre de progression
results = run_all_battery(keystream_bits, eligible_battery)

# Afficher les résultats des tests
print("\nRésultats des tests :")
for result, elapsed_time in results:
    if result is not None:
        if result.passed:
            print(f"- PASSED - score: {np.round(result.score, 3)} - {result.name} - elapsed time: {elapsed_time} ms")
        else:
            print(f"- FAILED - score: {np.round(result.score, 3)} - {result.name} - elapsed time: {elapsed_time} ms")
    else:
        print("- Test non éligible ou échoué.")
