from setuptools import setup, find_packages

setup(
    name='cryptoleo',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'python-dotenv',
        'numpy',
        'scipy',
        'tqdm',
        'pycryptodome'
    ],
    extras_require={
        'dev': [
            'pytest',
        ],
    },
    author='Théophile',
    author_email='theophile.avanel@gmail.com',
    description='Authenticated Encryption Based on Chaotic Neural Networks and Duplex Construction',
    url='https://github.com/thekester/cryptoleo',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
