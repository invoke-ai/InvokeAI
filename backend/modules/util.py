import os
import re

"""
Generates a unique filename in the format this project uses:
f`{unique prefix}.{seed}.{postprocessing}.png`

With logic from `ldm.dream.pngwriter.unique_prefix`.
"""


def get_filename(seed, path, postprocessing):
    # Generating a prefix
    # sort reverse alphabetically until we find max+1
    dirlist = sorted(os.listdir(path), reverse=True)
    # find the first filename that matches our pattern or return 000000.0.png
    existing_name = next(
        (f for f in dirlist if re.match('^(\d+)\..*\.png', f)),
        '0000000.0.png',
    )

    basecount = int(existing_name.split('.', 1)[0]) + 1
    prefix = f'{basecount: 06}'

    if postprocessing:
        return f'{prefix}.{seed}.{postprocessing}.png'
    else:
        return f'{prefix}.{seed}.png'
