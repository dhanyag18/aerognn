from setuptools import setup, find_packages

setup(
    name='aerognn',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        'torch', 'torch-geometric', 'numpy', 'scipy','matplotlib', 'click', 'xgboost', 'shapely', 'trimesh', 'pandas', 'scikit-learn', 'pyvista'
    ],
    entry_points={
        'console_scripts': [
            'aerognn=aerognn.cli:cli',
        ],
    },
)