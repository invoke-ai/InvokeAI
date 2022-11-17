from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        print('Will now try loading a module (develop)')
        import ldm.generate
        print('ldm.generate loaded ok')

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        print('Will now try loading a module (install)')
        import ldm.generate
        print('ldm.generate loaded ok')

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
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)

