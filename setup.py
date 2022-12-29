"""
To be compatible with legacy builds
"""
import os
from setuptools import setup


def list_files(directory):
    """
    returns all files in a provided directory
    """
    return [
        os.path.join(directory, x)
        for x in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, x))
    ]


setup(
    data_files=[
        ("frontend/dist", list_files("frontend/dist")),
        ("frontend/dist/assets", list_files("frontend/dist/assets")),
        ("assets", ["assets/caution.png"]),
        ("configs", list_files("configs")),
        ("configs/stable-diffusion", list_files("configs/stable-diffusion")),
        ("scripts", list_files("scripts")),
        # ("environments-and-requirements",[
        #     "environments-and-requirements/requirements-base.txt"
        #     ],
        # ),
    ],
)
