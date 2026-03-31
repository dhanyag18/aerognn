from setuptools import setup, find_packages

setup(
    name='aerognn',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch', 'torch-geometric', 'numpy', 'scipy',
        'matplotlib', 'meshio', 'click', 'xgboost',
    ],
    entry_points={
        'console_scripts': [
            'aerognn=aerognn.cli:cli',
        ],
    },
)