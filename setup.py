from setuptools import setup, find_packages

setup(
    name='MultiomicsLM',
    version='0.1.0',
    description='Ensemble inference for multiomics BERT disease classification models',
    author='Shizheng Qiu',
    author_email='qiushizheng@hit.edu.cn',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
    ],
    entry_points={
        'console_scripts': [
            'MultiomicsLM=MultiomicsLM.cli:main'
        ]
    },
)