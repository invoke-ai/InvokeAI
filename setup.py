import os
import re
from setuptools import setup, find_packages

def list_files(directory):
    return [os.path.join(directory,x) for x in os.listdir(directory) if os.path.isfile(os.path.join(directory,x))]

VERSION = '2.2.4'
DESCRIPTION = ('An implementation of Stable Diffusion which provides various new features'
               ' and options to aid the image generation process')
LONG_DESCRIPTION = ('This version of Stable Diffusion features a slick WebGUI, an'
                    ' interactive command-line script that combines text2img and img2img'
                    ' functionality in a "dream bot" style interface, and multiple features'
                    ' and other enhancements.')
HOMEPAGE = 'https://github.com/invoke-ai/InvokeAI'
REQUIREMENTS=[
    'accelerate',
    'albumentations',
    'diffusers[torch]==0.10.*',
    'einops',
    'eventlet',
    'facexlib',
    'flask==2.1.3',
    'flask_cors==3.0.10',
    'flask_socketio==5.3.0',
    'flaskwebgui==1.0.3',
    'getpass_asterisk',
    'gfpgan==1.3.8',
    'imageio-ffmpeg',
    'kornia',
    'numpy==1.23.*',
    'omegaconf',
    'opencv-python',
    'picklescan',
    'pillow',
    'pudb',
    'pyreadline3',
    'pytorch-lightning==1.7.7',
    'realesrgan',
    'requests==2.25.1',
    'scikit-image>=0.19',
    'send2trash',
    'streamlit',
    'taming-transformers-rom1504',
    'test-tube>=0.7.5',
    'torch',
    'torch-fidelity',
    'torchvision',
    'torchmetrics',
    'transformers==4.25.*',
    'clip @ https://github.com/openai/CLIP/archive/eaa22acb90a5876642d0507623e859909230a52d.zip',
    'clipseg @ https://github.com/invoke-ai/clipseg/archive/relaxed-python-requirement.zip',
    'k-diffusion @ https://github.com/Birch-san/k-diffusion/archive/refs/heads/mps.zip',
    'pypatchmatch @ https://github.com/mauwii/PyPatchMatch/archive/refs/heads/master.zip'
]

setup(
    name='InvokeAI',
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author='The InvokeAI Project',
    author_email='lincoln.stein@gmail.com',
    url=HOMEPAGE,
    license='MIT',
    packages=find_packages(exclude=['tests.*']),
    install_requires=REQUIREMENTS,
    dependency_links=['https://download.pytorch.org/whl/torch_stable.html'],
    python_requires='>=3.9, <3.11',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: GPU',
        'Environment :: GPU :: NVIDIA CUDA',
        'Environment :: MacOS X',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3 :: Only,'
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Artistic Software',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Application',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Server',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    scripts = ['scripts/invoke.py','scripts/configure_invokeai.py', 'scripts/sd-metadata.py',
               'scripts/preload_models.py', 'scripts/images2prompt.py','scripts/merge_embeddings.py'
    ],
    data_files=[('frontend/dist',list_files('frontend/dist')),
                ('frontend/dist/assets',list_files('frontend/dist/assets')),
                ('assets',['assets/caution.png']),
    ],
)
