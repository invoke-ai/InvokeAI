import re

from setuptools import setup, find_packages


def _get_requirements(path):
    try:
        with open(path) as f:
            p_str = f.read().splitlines()
    except (IOError, OSError) as ex:
        raise RuntimeError("Can't open file with requirements: %s", repr(ex))

    # Drop comment lines; delete ' ' and '\\'; strip lines
    packages = [re.sub(r"[\s\\]", '', package).strip() for package in p_str
                if not re.match(r"^\s*#", package)
                if package]
    
    # Drop option and hash lines
    packages = [package for package in packages if not re.match(r"^--", package)]
    return packages


DESCRIPTION = ('An implementation of Stable Diffusion which provides various new features'
               ' and options to aid the image generation process')
LONG_DESCRIPTION = ('This version of Stable Diffusion features a slick WebGUI, an'
                     ' interactive command-line script that combines text2img and img2img'
                     ' functionality in a "dream bot" style interface, and multiple features'
                     ' and other enhancements.')
HOMEPAGE = 'https://github.com/invoke-ai/InvokeAI'

setup(
    name='InvokeAI',
    version='2.0.2',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author='The InvokeAI Project',
    author_email='lincoln.stein@gmail.com',
    url=HOMEPAGE,
    license='MIT',
    packages=find_packages(exclude=['tests.*']),
    # Some day...
    # entry_points={
    #       'console_scripts': [],
    # },
    install_requires=_get_requirements('requirements.txt'),
    python_requires='>=3.8, <4',
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
)
 