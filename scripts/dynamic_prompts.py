#!/usr/bin/env python

"""
Simple script to generate a file of InvokeAI prompts and settings
that scan across steps and other parameters.
"""

import argparse
import io
import json
import pydoc
import re
import shutil
import sys
from dataclasses import dataclass
from io import TextIOBase
from itertools import product
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Iterable, List, Union

import yaml
from omegaconf import OmegaConf, dictconfig, listconfig


def expand_prompts(
    template_file: Path,
    run_invoke: bool = False,
    invoke_model: str = None,
    invoke_outdir: Path = None,
):
    """
    :param template_file: A YAML file containing templated prompts and args
    :param run_invoke: A boolean which if True will pass expanded prompts to invokeai CLI
    :param invoke_model: Name of the model to load when run_invoke is true; otherwise uses default
    :param invoke_outdir: Directory for outputs when run_invoke is true; otherwise uses default
    """
    if template_file.name.endswith(".json"):
        with open(template_file, "r") as file:
            with io.StringIO(yaml.dump(json.load(file))) as fh:
                conf = OmegaConf.load(fh)
    else:
        conf = OmegaConf.load(template_file)
    try:
        if run_invoke:
            invokeai_args = [shutil.which("invokeai")]
            if invoke_model:
                invokeai_args.extend(("--model", invoke_model))
            if invoke_outdir:
                invokeai_args.extend(("--outdir", invoke_outdir))
            print(f"Calling invokeai with arguments {invokeai_args}", file=sys.stderr)
            process = Popen(invokeai_args, stdin=PIPE, text=True)
            with process.stdin as fh:
                _do_expand(conf, file=fh)
            process.wait()
        else:
            _do_expand(conf)
    except KeyboardInterrupt:
        process.kill()


def main():
    parser = argparse.ArgumentParser(
        description=HELP,
    )
    parser.add_argument(
        "template_file",
        type=Path,
        nargs="?",
        help="path to a template file, use --example to generate an example file",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        default=False,
        help=f'Print an example template file in YAML format. Use "{sys.argv[0]} --example > example.yaml" to save output to a file',
    )
    parser.add_argument(
        "--json-example",
        action="store_true",
        default=False,
        help=f'Print an example template file in json format. Use "{sys.argv[0]} --json-example > example.json" to save output to a file',
    )
    parser.add_argument(
        "--instructions",
        "-i",
        dest="instructions",
        action="store_true",
        default=False,
        help=f"Print verbose instructions.",
    )
    parser.add_argument(
        "--invoke",
        action="store_true",
        help="Execute invokeai using specified optional --model and --outdir",
    )
    parser.add_argument(
        "--model",
        help="Feed the generated prompts to the invokeai CLI using the indicated model. Will be overriden by a model: section in template file.",
    )
    parser.add_argument(
        "--outdir", type=Path, help="Write images and log into indicated directory"
    )
    opt = parser.parse_args()

    if opt.example:
        print(EXAMPLE_TEMPLATE_FILE)
        sys.exit(0)

    if opt.json_example:
        print(_yaml_to_json(EXAMPLE_TEMPLATE_FILE))
        sys.exit(0)

    if opt.instructions:
        pydoc.pager(INSTRUCTIONS)
        sys.exit(0)

    if not opt.template_file:
        parser.print_help()
        sys.exit(-1)

    expand_prompts(
        template_file=opt.template_file,
        run_invoke=opt.invoke,
        invoke_model=opt.model,
        invoke_outdir=opt.outdir,
    )


def _do_expand(conf: OmegaConf, file: TextIOBase = sys.stdout):
    models = expand_values(conf.get("model"))
    steps = expand_values(conf.get("steps")) or [30]
    cfgs = expand_values(conf.get("cfg")) or [7.5]
    samplers = expand_values(conf.get("sampler")) or ["ddim"]
    seeds = expand_values(conf.get("seed")) or [0]
    prompts = expand_prompt(conf.get("prompt")) or ["banana sushi"]
    dimensions = expand_prompt(conf.get("dimensions")) or ["512x512"]

    cross_product = product(
        *[models, seeds, prompts, samplers, cfgs, steps, dimensions]
    )
    previous_model = None
    for p in cross_product:
        (model, seed, prompt, sampler, cfg, step, dimensions) = tuple(p)
        (width, height) = dimensions.split("x")
        if previous_model != model:
            previous_model = model
            print(f"!switch {model}", file=file)
        print(
            f'"{prompt}" -S{seed} -A{sampler} -C{cfg} -s{step} -W{width} -H{height}',
            file=file,
        )


def expand_prompt(
    stanza: str | dict | listconfig.ListConfig | dictconfig.DictConfig,
) -> list | range:
    if not stanza:
        return None
    if isinstance(stanza, listconfig.ListConfig):
        return stanza
    if isinstance(stanza, str):
        return [stanza]
    if not isinstance(stanza, dictconfig.DictConfig):
        raise ValueError(f"Unrecognized template: {stanza}")

    if not (template := stanza.get("template")):
        raise KeyError('"prompt" section must contain a "template" definition')

    fragment_labels = re.findall("{([^{}]+?)}", template)
    if len(fragment_labels) == 0:
        return [template]
    fragments = [[{x: y} for y in stanza.get(x)] for x in fragment_labels]
    dicts = merge(product(*fragments))
    return [template.format(**x) for x in dicts]


def merge(dicts: Iterable) -> List[dict]:
    result = list()
    for x in dicts:
        to_merge = dict()
        for item in x:
            to_merge = to_merge | item
        result.append(to_merge)
    return result


def expand_values(stanza: str | dict | listconfig.ListConfig) -> list | range:
    if not stanza:
        return None
    if isinstance(stanza, listconfig.ListConfig):
        return stanza
    elif match := re.match("^(\d+);(\d+)(;(\d+))?", str(stanza)):
        return range(
            int(match.group(1)), 1 + int(match.group(2)), int(match.group(4)) or 1
        )
    else:
        return [stanza]


def _yaml_to_json(yaml_input: str) -> str:
    """
    Converts a yaml string into a json string. Used internally
    to generate the example template file.
    """
    with io.StringIO(yaml_input) as yaml_in:
        data = yaml.safe_load(yaml_in)
    return json.dumps(data, indent=2)


HELP = f"""
This script takes a prompt template file that contains multiple
alternative values for the prompt and its generation arguments (such
as steps). It then expands out the prompts using all combinations of
arguments and either prints them to the terminal's standard output, or
feeds the prompts directly to the invokeai command-line interface.

Call this script again with --instructions (-i) for verbose instructions.
"""

INSTRUCTIONS = f"""
== INTRODUCTION ==
This script takes a prompt template file that contains multiple
alternative values for the prompt and its generation arguments (such
as steps). It then expands out the prompts using all combinations of
arguments and either prints them to the terminal's standard output, or
feeds the prompts directly to the invokeai command-line interface.

If the optional --invoke argument is provided, then the generated
prompts will be fed directly to invokeai for image generation. You
will likely want to add the --outdir option in order to save the image
files to their own folder.

   {sys.argv[0]} --invoke --outdir=/tmp/outputs my_template.yaml

If --invoke isn't specified, the expanded prompts will be printed to
output. You can capture them to a file for inspection and editing this
way:

   {sys.argv[0]} my_template.yaml > prompts.txt

And then feed them to invokeai this way:

  invokeai --outdir=/tmp/outputs < prompts.txt

Note that after invokeai finishes processing the list of prompts, the
output directory will contain a markdown file named `log.md`
containing annotated images. You can open this file using an e-book
reader such as the cross-platform Calibre eBook reader
(https://calibre-ebook.com/).

== FORMAT OF THE TEMPLATES FILE ==

This will generate an example template file that you can get
started with:

  {sys.argv[0]} --example > example.yaml

An excerpt from the top of this file looks like this:

 model:
   - stable-diffusion-1.5
   - stable-diffusion-2.1-base
 steps: 30:50:1  # start steps at 30 and go up to 50, incrementing by 1 each time
 seed: 50        # fixed constant, seed=50
 cfg:            # list of CFG values to try
   - 7
   - 8
   - 12
 prompt: a walk in the park   # constant value

In more detail, the template file can have any of the
following sections:
 - model:
 - steps:
 - seed:
 - cfg:
 - sampler:
 - prompt:

- Each section can have a constant value such as this:
     steps: 50
- Or a range of numeric values in the format:
     steps: <start>;<stop>;<step>      (note semicolon, not colon!)
- Or a list of values in the format:
     - value1
     - value2
     - value3

The "prompt:" section is special. It can accept a constant value:

   prompt: a walk in the woods in the style of donatello

Or it can accept a list of prompts:

   prompt:
      - a walk in the woods
      - a walk on the beach

Or it can accept a templated list of prompts. These allow you to
define a series of phrases, each of which is a list. You then combine
them together into a prompt template in this way:

  prompt:
    style:
         - greg rutkowski
         - gustav klimt
         - renoir
         - donetello
    subject:
         - sunny meadow in the mountains
         - gathering storm in the mountains
    template: a {{subject}} in the style of {{style}}

In the example above, the phrase names "style" and "subject" are
examples only. You can use whatever you like. However, the "template:"
field is required. The output will be:

  "a sunny meadow in the mountains in the style of greg rutkowski"
  "a sunny meadow in the mountains in the style of gustav klimt"
  ...
  "a gathering storm in the mountains in the style of donetello"

== SUPPORT FOR JSON FORMAT ==

For those who prefer the JSON format, this script will accept JSON
template files as well. Please run "{sys.argv[1]} --json-example"
to print out a version of the example template file in json format.
You may save it to disk and use it as a starting point for your own
template this way:

   {sys.argv[1]} --json-example > template.json 
"""

EXAMPLE_TEMPLATE_FILE = """
model: stable-diffusion-1.5
steps: 30;50;10
seed: 50
dimensions: 512x512
cfg:
  - 7
  - 12
sampler:
  - k_euler_a
  - k_lms
prompt:
  style:
       - greg rutkowski
       - gustav klimt
  location:
       - the mountains
       - a desert
  object:
       - luxurious dwelling
       - crude tent
  template: a {object} in {location}, in the style of {style}
"""

if __name__ == "__main__":
    main()
