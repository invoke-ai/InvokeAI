#!/usr/bin/env python

'''
Simple script to generate a file of InvokeAI prompts and settings
that scan across steps and other parameters.
'''

from omegaconf import OmegaConf, listconfig
import re
import sys

INSTRUCTIONS='''
To use, create a file named "template.yaml" (or similar) formatted like this
>>> cut here <<<
steps: "30:50:1"
seed: 50
cfg:
  - 7
  - 8
  - 12
sampler:
  - ddim
  - k_lms
prompt:
  - a sunny meadow in the mountains
  - a gathering storm in the mountains
>>> cut here <<<

Create sections named "steps", "seed", "cfg", "sampler" and "prompt".
- Each section can have a constant value such as this:
     steps: 50
- Or a range of numeric values in the format:
     steps: "<start>:<stop>:<step>"
- Or a list of values in the format:
     - value1
     - value2
     - value3

Be careful to: 1) put quotation marks around numeric ranges; 2) put a 
space between the "-" and the value in a list of values; and 3) use spaces,
not tabs, at the beginnings of indented lines.

When you run this script, capture the output into a text file like this:

    python generate_param_scan.py template.yaml > output_prompts.txt

"output_prompts.txt" will now contain an expansion of all the list
values you provided. You can examine it in a text editor such as
Notepad.

Now start the CLI, and feed the expanded prompt file to it using the
"!replay" command:

   !replay output_prompts.txt

Alternatively, you can directly feed the output of this script
by issuing a command like this from the developer's console:

   python generate_param_scan.py template.yaml | invokeai

You can use the web interface to view the resulting images and their
metadata.
'''

def main():
    if len(sys.argv)<2:
        print(f'Usage: {__file__} template_file.yaml')
        print('Outputs a series of prompts expanded from the provided template.')
        print(INSTRUCTIONS)
        sys.exit(-1)
        
    conf_file = sys.argv[1]
    conf = OmegaConf.load(conf_file)

    steps = expand_values(conf.get('steps')) or [30]
    cfg = expand_values(conf.get('cfg')) or [7.5]
    sampler = expand_values(conf.get('sampler')) or ['ddim']
    prompt = expand_values(conf.get('prompt')) or ['banana sushi']
    seed = expand_values(conf.get('seed'))

    for seed in seed:
        for p in prompt:
            for s in sampler:
                for c in cfg:
                    for step in steps:
                        print(f'"{p}" -s{step} {f"-S{seed}" if seed else ""} -A{s} -C{c}')

def expand_values(stanza: str|dict|listconfig.ListConfig)->list|range:
    if not stanza:
        return None
    if isinstance(stanza,listconfig.ListConfig):
        return stanza
    elif match := re.match('^(\d+):(\d+)(:(\d+))?',str(stanza)):
        return range(int(match.group(1)), int(match.group(2)), int(match.group(4)) or 1)
    else:
        return [stanza]
    
if __name__ == '__main__':
    main()
