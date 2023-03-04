'''
This is a modularized version of the sd-metadata.py script,
which retrieves and prints the metadata from a series of generated png files.
'''
import sys
import json
from invokeai.backend.image_util import retrieve_metadata


def print_metadata():
    if len(sys.argv) < 2:
        print("Usage: file2prompt.py <file1.png> <file2.png> <file3.png>...")
        print("This script opens up the indicated invoke.py-generated PNG file(s) and prints out their metadata.")
        exit(-1)

    filenames = sys.argv[1:]
    for f in filenames:
        try:
            metadata = retrieve_metadata(f)
            print(f'{f}:\n',json.dumps(metadata['sd-metadata'], indent=4))
        except FileNotFoundError:
            sys.stderr.write(f'{f} not found\n')
            continue
        except PermissionError:
            sys.stderr.write(f'{f} could not be opened due to inadequate permissions\n')
            continue

if __name__== '__main__':
    print_metadata()

