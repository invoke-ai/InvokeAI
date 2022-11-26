import os
import re

from setuptools import setup, find_packages


def frontend_files(directory):
    paths = []
    for (path, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths


def _get_requirements(path):
    try:
        with open(path) as f:
            packages = f.read().splitlines()
    except (IOError, OSError) as ex:
        raise RuntimeError("Can't open file with requirements: %s", repr(ex))

    # Drop option lines
    packages = [package for package in packages if not re.match(r'^--', package)]
    print(f'Packages found for "install_requires":\n{packages}')
    return packages


frontend_files = frontend_files('frontend/dist')
print(f'DEBUG: {frontend_files}')

VERSION = '2.1.4'
DESCRIPTION = ('An implementation of Stable Diffusion which provides various new features'
               ' and options to aid the image generation process')
LONG_DESCRIPTION = ('This version of Stable Diffusion features a slick WebGUI, an'
                    ' interactive command-line script that combines text2img and img2img'
                    ' functionality in a "dream bot" style interface, and multiple features'
                    ' and other enhancements.')
HOMEPAGE = 'https://github.com/invoke-ai/InvokeAI'

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
    install_requires=_get_requirements('installer/requirements.in'),
    dependency_links=['https://download.pytorch.org/whl/torch_stable.html'],
    python_requires='>=3.9, <4',
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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Artistic Software',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Application',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Server',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    scripts = ['scripts/invoke.py','scripts/configure_invokeai.py', 'scripts/preload_models.py', 'scripts/sd-metadata.py'],
    data_files=[('frontend',frontend_files)],
)
