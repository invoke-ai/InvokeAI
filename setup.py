from setuptools import setup, find_packages
import os

def frontend_files(directory):
     paths = []
     for (path, directories, filenames) in os.walk(directory):
         for filename in filenames:
             paths.append(os.path.join(path, filename))
     return paths

frontend_files = frontend_files('frontend/dist')
print(f'DEBUG: {frontend_files}')


setup(
    name='invoke-ai',
    version='2.1.4',
    description='InvokeAI: A Stable Diffusion text to image generation toolkit',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
    scripts = ['scripts/invoke.py','scripts/load_models.py','scripts/sd-metadata.py'],
    data_files=[('frontend',frontend_files)],
)

