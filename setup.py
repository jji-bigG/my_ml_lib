from setuptools import setup, find_packages

setup(
    name='my_ml_',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    entry_points={
        'console_scripts': [
            'main=main:main',
        ],
    },
)
