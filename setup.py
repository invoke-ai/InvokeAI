from setuptools import setup, find_packages

setup(
    name='invoke-ai',
    version='2.1.3',
    description='InvokeAI text to image generation toolkit',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)

