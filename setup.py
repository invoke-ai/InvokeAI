"""
To be compatible with legacy builds
"""
from setuptools import setup

setup()


# def list_files(directory):
#     """
#     returns all files in a provided directory
#     """
#     return [
#         os.path.join(directory, x)
#         for x in os.listdir(directory)
#         if os.path.isfile(os.path.join(directory, x))
#     ]


# setup(
#     data_files=[
#         ("frontend/dist", "./frontend/dist"),
#         ("assets", ["assets/caution.png"]),
#         ("configs", list_files("configs")),
#         ("configs/stable-diffusion", list_files("configs/stable-diffusion")),
#         ("scripts", list_files("scripts")),
#     ],
# )
