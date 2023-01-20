"""Helper class for dealing with image generation arguments.

The Args class parses both the command line (shell) arguments, as well as the
command string passed at the invoke> prompt. It serves as the definitive repository
of all the arguments used by Generate and their default values, and implements the
preliminary metadata standards discussed here:

https://github.com/lstein/stable-diffusion/issues/266

To use:
  opt = Args()

  # Read in the command line options:
  # this returns a namespace object like the underlying argparse library)
  # You do not have to use the return value, but you can check it against None
  # to detect illegal arguments on the command line.
  args = opt.parse_args()
  if not args:
     print('oops')
     sys.exit(-1)

  # read in a command passed to the invoke> prompt:
  opts = opt.parse_cmd('do androids dream of electric sheep? -H256 -W1024 -n4')

  # The Args object acts like a namespace object
  print(opt.model)

You can set attributes in the usual way, use vars(), etc.:

  opt.model = 'something-else'
  do_something(**vars(a))

It is helpful in saving metadata:

  # To get a json representation of all the values, allowing
  # you to override any values dynamically
  j = opt.json(seed=42)

  # To get the prompt string with the switches, allowing you
  # to override any values dynamically
  j = opt.dream_prompt_str(seed=42)

If you want to access the namespace objects from the shell args or the
parsed command directly, you may use the values returned from the
original calls to parse_args() and parse_cmd(), or get them later
using the _arg_switches and _cmd_switches attributes. This can be
useful if both the args and the command contain the same attribute and
you wish to apply logic as to which one to use. For example:

  a = Args()
  args    = a.parse_args()
  opts    = a.parse_cmd(string)
  do_grid = args.grid or opts.grid

To add new attributes, edit the _create_arg_parser() and
_create_dream_cmd_parser() methods.

**Generating and retrieving sd-metadata**

To generate a dict representing RFC266 metadata:

  metadata = metadata_dumps(opt,<seeds,model_hash,postprocesser>)

This will generate an RFC266 dictionary that can then be turned into a JSON
and written to the PNG file. The optional seeds, weights, model_hash and
postprocesser arguments are not available to the opt object and so must be
provided externally. See how invoke.py does it.

Note that this function was originally called format_metadata() and a wrapper
is provided that issues a deprecation notice.

To retrieve a (series of) opt objects corresponding to the metadata, do this:

 opt_list = metadata_loads(metadata)

The metadata should be pulled out of the PNG image. pngwriter has a method
retrieve_metadata that will do this, or you can do it in one swell foop
with metadata_from_png():

 opt_list = metadata_from_png('/path/to/image_file.png')
"""

import argparse
import base64
import copy
import functools
import hashlib
import json
import os
import pydoc
import re
import shlex
import sys
import ldm.invoke
import ldm.invoke.pngwriter

from ldm.invoke.globals import Globals
from ldm.invoke.prompt_parser import split_weighted_subprompts
from argparse import Namespace
from pathlib import Path

APP_ID = ldm.invoke.__app_id__
APP_NAME = ldm.invoke.__app_name__
APP_VERSION = ldm.invoke.__version__

SAMPLER_CHOICES = [
    'ddim',
    'k_dpm_2_a',
    'k_dpm_2',
    'k_dpmpp_2_a',
    'k_dpmpp_2',
    'k_euler_a',
    'k_euler',
    'k_heun',
    'k_lms',
    'plms',
    # diffusers:
    "pndm",
]

PRECISION_CHOICES = [
    'auto',
    'float32',
    'autocast',
    'float16',
]

class ArgFormatter(argparse.RawTextHelpFormatter):
        # use defined argument order to display usage
    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = 'usage: '

        # if usage is specified, use that
        if usage is not None:
            usage = usage % dict(prog=self._prog)

        # if no optionals or positionals are available, usage is just prog
        elif usage is None and not actions:
            usage = 'invoke>'
        elif usage is None:
            prog='invoke>'
            # build full usage string
            action_usage = self._format_actions_usage(actions, groups) # NEW
            usage = ' '.join([s for s in [prog, action_usage] if s])
            # omit the long line wrapping code
        # prefix with 'usage:'
        return '%s%s\n\n' % (prefix, usage)

class PagingArgumentParser(argparse.ArgumentParser):
    '''
    A custom ArgumentParser that uses pydoc to page its output.
    It also supports reading defaults from an init file.
    '''
    def print_help(self, file=None):
        text = self.format_help()
        pydoc.pager(text)

    def convert_arg_line_to_args(self, arg_line):
        return shlex.split(arg_line,comments=True)

class Args(object):
    def __init__(self,arg_parser=None,cmd_parser=None):
        '''
        Initialize new Args class. It takes two optional arguments, an argparse
        parser for switches given on the shell command line, and an argparse
        parser for switches given on the invoke> CLI line. If one or both are
        missing, it creates appropriate parsers internally.
        '''
        self._arg_parser   = arg_parser or self._create_arg_parser()
        self._cmd_parser   = cmd_parser or self._create_dream_cmd_parser()
        self._arg_switches = self.parse_cmd('')   # fill in defaults
        self._cmd_switches = self.parse_cmd('')   # fill in defaults

    def parse_args(self):
        '''Parse the shell switches and store.'''
        try:
            sysargs = sys.argv[1:]
            # pre-parse before we do any initialization to get root directory
            # and intercept --version request
            switches = self._arg_parser.parse_args(sysargs)
            if switches.version:
                print(f'{ldm.invoke.__app_name__} {ldm.invoke.__version__}')
                sys.exit(0)

            print('* Initializing, be patient...')
            Globals.root = Path(os.path.abspath(switches.root_dir or Globals.root))
            Globals.try_patchmatch = switches.patchmatch

            # now use root directory to find the init file
            initfile = os.path.expanduser(os.path.join(Globals.root,Globals.initfile))
            legacyinit = os.path.expanduser('~/.invokeai')
            if os.path.exists(initfile):
                print(f'>> Initialization file {initfile} found. Loading...',file=sys.stderr)
                sysargs.insert(0,f'@{initfile}')
            elif os.path.exists(legacyinit):
                print(f'>> WARNING: Old initialization file found at {legacyinit}. This location is deprecated. Please move it to {Globals.root}/invokeai.init.')
                sysargs.insert(0,f'@{legacyinit}')

            self._arg_switches = self._arg_parser.parse_args(sysargs)
            return self._arg_switches
        except Exception as e:
            print(f'An exception has occurred: {e}')
            return None

    def parse_cmd(self,cmd_string):
        '''Parse a invoke>-style command string '''
        # handle the case in which the first token is a switch
        if cmd_string.startswith('-'):
            prompt = ''
            switches = cmd_string
        # handle the case in which the prompt is enclosed by quotes
        elif cmd_string.startswith('"'):
            a = shlex.split(cmd_string,comments=True)
            prompt = a[0]
            switches = shlex.join(a[1:])
        else:
            # no initial quote, so get everything up to the first thing
            # that looks like a switch
            if cmd_string.startswith('-'):
                prompt = ''
                switches = cmd_string
            else:
                match = re.match('^(.+?)\s(--?[a-zA-Z].+)',cmd_string)
                if match:
                    prompt,switches = match.groups()
                else:
                    prompt = cmd_string
                    switches = ''
        try:
            self._cmd_switches = self._cmd_parser.parse_args(shlex.split(switches,comments=True))
            setattr(self._cmd_switches,'prompt',prompt)
            return self._cmd_switches
        except:
            return None

    def json(self,**kwargs):
        return json.dumps(self.to_dict(**kwargs))

    def to_dict(self,**kwargs):
        a = vars(self)
        a.update(kwargs)
        return a

    # Isn't there a more automated way of doing this?
    # Ideally we get the switch strings out of the argparse objects,
    # but I don't see a documented API for this.
    def dream_prompt_str(self,**kwargs):
        """Normalized dream_prompt."""
        a = vars(self)
        a.update(kwargs)
        switches = list()
        prompt = a['prompt']
        prompt.replace('"','\\"')
        switches.append(prompt)
        switches.append(f'-s {a["steps"]}')
        switches.append(f'-S {a["seed"]}')
        switches.append(f'-W {a["width"]}')
        switches.append(f'-H {a["height"]}')
        switches.append(f'-C {a["cfg_scale"]}')
        if a['karras_max'] is not None:
            switches.append(f'--karras_max {a["karras_max"]}')
        if a['perlin'] > 0:
            switches.append(f'--perlin {a["perlin"]}')
        if a['threshold'] > 0:
            switches.append(f'--threshold {a["threshold"]}')
        if a['grid']:
            switches.append('--grid')
        if a['seamless']:
            switches.append('--seamless')
        if a['hires_fix']:
            switches.append('--hires_fix')

        # img2img generations have parameters relevant only to them and have special handling
        if a['init_img'] and len(a['init_img'])>0:
            switches.append(f'-I {a["init_img"]}')
            switches.append(f'-A {a["sampler_name"]}')
            if a['fit']:
                switches.append('--fit')
            if a['init_mask'] and len(a['init_mask'])>0:
                switches.append(f'-M {a["init_mask"]}')
            if a['init_color'] and len(a['init_color'])>0:
                switches.append(f'--init_color {a["init_color"]}')
            if a['strength'] and a['strength']>0:
                switches.append(f'-f {a["strength"]}')
            if a['inpaint_replace']:
                switches.append('--inpaint_replace')
            if a['text_mask']:
                switches.append(f'-tm {" ".join([str(u) for u in a["text_mask"]])}')
        else:
            switches.append(f'-A {a["sampler_name"]}')

        # facetool-specific parameters, only print if running facetool
        if a['facetool_strength']:
            switches.append(f'-G {a["facetool_strength"]}')
            switches.append(f'-ft {a["facetool"]}')
            if a["facetool"] == "codeformer":
                switches.append(f'-cf {a["codeformer_fidelity"]}')

        if a['outcrop']:
            switches.append(f'-c {" ".join([str(u) for u in a["outcrop"]])}')

        # esrgan-specific parameters
        if a['upscale']:
            switches.append(f'-U {" ".join([str(u) for u in a["upscale"]])}')

        # embiggen parameters
        if a['embiggen']:
            switches.append(f'--embiggen {" ".join([str(u) for u in a["embiggen"]])}')
        if a['embiggen_tiles']:
            switches.append(f'--embiggen_tiles {" ".join([str(u) for u in a["embiggen_tiles"]])}')
        if a['embiggen_strength']:
            switches.append(f'--embiggen_strength {a["embiggen_strength"]}')

        # outpainting parameters
        if a['out_direction']:
            switches.append(f'-D {" ".join([str(u) for u in a["out_direction"]])}')

        # LS: slight semantic drift which needs addressing in the future:
        # 1. Variations come out of the stored metadata as a packed string with the keyword "variations"
        # 2. However, they come out of the CLI (and probably web) with the keyword "with_variations" and
        #    in broken-out form. Variation (1) should be changed to comply with (2)
        if a['with_variations'] and len(a['with_variations'])>0:
            formatted_variations = ','.join(f'{seed}:{weight}' for seed, weight in (a["with_variations"]))
            switches.append(f'-V {formatted_variations}')
        if 'variations' in a and len(a['variations'])>0:
            switches.append(f'-V {a["variations"]}')
        return ' '.join(switches)

    def __getattribute__(self,name):
        '''
        Returns union of command-line arguments and dream_prompt arguments,
        with the latter superseding the former.
        '''
        cmd_switches = None
        arg_switches = None
        try:
            cmd_switches = object.__getattribute__(self,'_cmd_switches')
            arg_switches = object.__getattribute__(self,'_arg_switches')
        except AttributeError:
            pass

        if cmd_switches and arg_switches and name=='__dict__':
            return self._merge_dict(
                arg_switches.__dict__,
                cmd_switches.__dict__,
            )
        try:
            return object.__getattribute__(self,name)
        except AttributeError:
            pass

        if not hasattr(cmd_switches,name) and not hasattr(arg_switches,name):
            raise AttributeError

        value_arg,value_cmd = (None,None)
        try:
            value_cmd = getattr(cmd_switches,name)
        except AttributeError:
            pass
        try:
            value_arg = getattr(arg_switches,name)
        except AttributeError:
            pass

        # here is where we can pick and choose which to use
        # default behavior is to choose the dream_command value over
        # the arg value. For example, the --grid and --individual options are a little
        # funny because of their push/pull relationship. This is how to handle it.
        if name=='grid':
            if cmd_switches.individual:
                return False
            else:
                return value_cmd or value_arg
        return value_cmd if value_cmd is not None else value_arg

    def __setattr__(self,name,value):
        if name.startswith('_'):
            object.__setattr__(self,name,value)
        else:
            self._cmd_switches.__dict__[name] = value

    def _merge_dict(self,dict1,dict2):
        new_dict  = {}
        for k in set(list(dict1.keys())+list(dict2.keys())):
            value1 = dict1.get(k,None)
            value2 = dict2.get(k,None)
            new_dict[k] = value2 if value2 is not None else value1
        return new_dict

    def _create_init_file(self,initfile:str):
        with open(initfile, mode='w', encoding='utf-8') as f:
            f.write('''# InvokeAI initialization file
# Put frequently-used startup commands here, one or more per line
# Examples:
# --web --host=0.0.0.0
# --steps 20
# -Ak_euler_a -C10.0
'''
            )

    def _create_arg_parser(self):
        '''
        This defines all the arguments used on the command line when you launch
        the CLI or web backend.
        '''
        parser = PagingArgumentParser(
            description=
            """
            Generate images using Stable Diffusion.
            Use --web to launch the web interface.
            Use --from_file to load prompts from a file path or standard input ("-").
            Otherwise you will be dropped into an interactive command prompt (type -h for help.)
            Other command-line arguments are defaults that can usually be overridden
            prompt the command prompt.
            """,
            fromfile_prefix_chars='@',
        )
        general_group    = parser.add_argument_group('General')
        model_group      = parser.add_argument_group('Model selection')
        file_group       = parser.add_argument_group('Input/output')
        web_server_group = parser.add_argument_group('Web server')
        render_group     = parser.add_argument_group('Rendering')
        postprocessing_group     = parser.add_argument_group('Postprocessing')
        deprecated_group = parser.add_argument_group('Deprecated options')

        deprecated_group.add_argument('--laion400m')
        deprecated_group.add_argument('--weights') # deprecated
        general_group.add_argument(
            '--version','-V',
            action='store_true',
            help='Print InvokeAI version number'
        )
        model_group.add_argument(
            '--root_dir',
            default=None,
            help='Path to directory containing "models", "outputs" and "configs". If not present will read from environment variable INVOKEAI_ROOT. Defaults to ~/invokeai.',
        )
        model_group.add_argument(
            '--config',
            '-c',
            '-config',
            dest='conf',
            default='./configs/models.yaml',
            help='Path to configuration file for alternate models.',
        )
        model_group.add_argument(
            '--model',
            help='Indicates which diffusion model to load (defaults to "default" stanza in configs/models.yaml)',
        )
        model_group.add_argument(
            '--weight_dirs',
            nargs='+',
            type=str,
            help='List of one or more directories that will be auto-scanned for new model weights to import',
        )
        model_group.add_argument(
            '--png_compression','-z',
            type=int,
            default=6,
            choices=range(0,9),
            dest='png_compression',
            help='level of PNG compression, from 0 (none) to 9 (maximum). Default is 6.'
        )
        model_group.add_argument(
            '-F',
            '--full_precision',
            dest='full_precision',
            action='store_true',
            help='Deprecated way to set --precision=float32',
        )
        model_group.add_argument(
            '--max_loaded_models',
            dest='max_loaded_models',
            type=int,
            default=2,
            help='Maximum number of models to keep in memory for fast switching, including the one in GPU',
        )
        model_group.add_argument(
            '--free_gpu_mem',
            dest='free_gpu_mem',
            action='store_true',
            help='Force free gpu memory before final decoding',
        )
        model_group.add_argument(
            '--xformers',
            action=argparse.BooleanOptionalAction,
            default=True,
            help='Enable/disable xformers support (default enabled if installed)',
        )
        model_group.add_argument(
            "--always_use_cpu",
            dest="always_use_cpu",
            action="store_true",
            help="Force use of CPU even if GPU is available"
        )
        model_group.add_argument(
            '--precision',
            dest='precision',
            type=str,
            choices=PRECISION_CHOICES,
            metavar='PRECISION',
            help=f'Set model precision. Defaults to auto selected based on device. Options: {", ".join(PRECISION_CHOICES)}',
            default='auto',
        )
        model_group.add_argument(
            '--internet',
            action=argparse.BooleanOptionalAction,
            dest='internet_available',
            default=True,
            help='Indicate whether internet is available for just-in-time model downloading (default: probe automatically).',
        )
        model_group.add_argument(
            '--nsfw_checker',
            '--safety_checker',
            action=argparse.BooleanOptionalAction,
            dest='safety_checker',
            default=False,
            help='Check for and blur potentially NSFW images. Use --no-nsfw_checker to disable.',
        )
        model_group.add_argument(
            '--autoconvert',
            default=None,
            type=str,
            help='Check the indicated directory for .ckpt weights files at startup and import as optimized diffuser models',
        )
        model_group.add_argument(
            '--patchmatch',
            action=argparse.BooleanOptionalAction,
            default=True,
            help='Load the patchmatch extension for outpainting. Use --no-patchmatch to disable.',
        )
        file_group.add_argument(
            '--from_file',
            dest='infile',
            type=str,
            help='If specified, load prompts from this file',
        )
        file_group.add_argument(
            '--outdir',
            '-o',
            type=str,
            help='Directory to save generated images and a log of prompts and seeds. Default: outputs/img-samples',
            default='outputs/img-samples',
        )
        file_group.add_argument(
            '--prompt_as_dir',
            '-p',
            action='store_true',
            help='Place images in subdirectories named after the prompt.',
        )
        render_group.add_argument(
            '--fnformat',
            default='{prefix}.{seed}.png',
            type=str,
            help='Overwrite the filename format. You can use any argument as wildcard enclosed in curly braces. Default is {prefix}.{seed}.png',
        )
        render_group.add_argument(
            '-s',
            '--steps',
            type=int,
            default=50,
            help='Number of steps'
        )
        render_group.add_argument(
            '-W',
            '--width',
            type=int,
            help='Image width, multiple of 64',
        )
        render_group.add_argument(
            '-H',
            '--height',
            type=int,
            help='Image height, multiple of 64',
        )
        render_group.add_argument(
            '-C',
            '--cfg_scale',
            default=7.5,
            type=float,
            help='Classifier free guidance (CFG) scale - higher numbers cause generator to "try" harder.',
        )
        render_group.add_argument(
            '--sampler',
            '-A',
            '-m',
            dest='sampler_name',
            type=str,
            choices=SAMPLER_CHOICES,
            metavar='SAMPLER_NAME',
            help=f'Set the default sampler. Supported samplers: {", ".join(SAMPLER_CHOICES)}',
            default='k_lms',
        )
        render_group.add_argument(
            '-f',
            '--strength',
            type=float,
            help='img2img strength for noising/unnoising. 0.0 preserves image exactly, 1.0 replaces it completely',
        )
        render_group.add_argument(
            '-T',
            '-fit',
            '--fit',
            action=argparse.BooleanOptionalAction,
            help='If specified, will resize the input image to fit within the dimensions of width x height (512x512 default)',
        )

        render_group.add_argument(
            '--grid',
            '-g',
            action=argparse.BooleanOptionalAction,
            help='generate a grid'
        )
        render_group.add_argument(
            '--embedding_directory',
            '--embedding_path',
            dest='embedding_path',
            default='embeddings',
            type=str,
            help='Path to a directory containing .bin and/or .pt files, or a single .bin/.pt file. You may use subdirectories. (default is ROOTDIR/embeddings)'
        )
        render_group.add_argument(
            '--embeddings',
            action=argparse.BooleanOptionalAction,
            default=True,
            help='Enable embedding directory (default). Use --no-embeddings to disable.',
        )
        render_group.add_argument(
            '--enable_image_debugging',
            action='store_true',
            help='Generates debugging image to display'
        )
        render_group.add_argument(
            '--karras_max',
            type=int,
            default=None,
            help="control the point at which the K* samplers will shift from using the Karras noise schedule (good for low step counts) to the LatentDiffusion noise schedule (good for high step counts). Set to 0 to use LatentDiffusion for all step values, and to a high value (e.g. 1000) to use Karras for all step values. [29]."
        )
        # Restoration related args
        postprocessing_group.add_argument(
            '--no_restore',
            dest='restore',
            action='store_false',
            help='Disable face restoration with GFPGAN or codeformer',
        )
        postprocessing_group.add_argument(
            '--no_upscale',
            dest='esrgan',
            action='store_false',
            help='Disable upscaling with ESRGAN',
        )
        postprocessing_group.add_argument(
            '--esrgan_bg_tile',
            type=int,
            default=400,
            help='Tile size for background sampler, 0 for no tile during testing. Default: 400.',
        )
        postprocessing_group.add_argument(
            '--gfpgan_model_path',
            type=str,
            default='./models/gfpgan/GFPGANv1.4.pth',
            help='Indicates the path to the GFPGAN model',
        )
        web_server_group.add_argument(
            '--web',
            dest='web',
            action='store_true',
            help='Start in web server mode.',
        )
        web_server_group.add_argument(
            '--web_develop',
            dest='web_develop',
            action='store_true',
            help='Start in web server development mode.',
        )
        web_server_group.add_argument(
            "--web_verbose",
            action="store_true",
            help="Enables verbose logging",
        )
        web_server_group.add_argument(
            "--cors",
            nargs="*",
            type=str,
            help="Additional allowed origins, comma-separated",
        )
        web_server_group.add_argument(
            '--host',
            type=str,
            default='127.0.0.1',
            help='Web server: Host or IP to listen on. Set to 0.0.0.0 to accept traffic from other devices on your network.'
        )
        web_server_group.add_argument(
            '--port',
            type=int,
            default='9090',
            help='Web server: Port to listen on'
        )
        web_server_group.add_argument(
            '--certfile',
            type=str,
            default=None,
            help='Web server: Path to certificate file to use for SSL. Use together with --keyfile'
        )
        web_server_group.add_argument(
            '--keyfile',
            type=str,
            default=None,
            help='Web server: Path to private key file to use for SSL. Use together with --certfile'
        )
        web_server_group.add_argument(
            '--gui',
            dest='gui',
            action='store_true',
            help='Start InvokeAI GUI',
        )
        return parser

    # This creates the parser that processes commands on the invoke> command line
    def _create_dream_cmd_parser(self):
        parser = PagingArgumentParser(
            formatter_class=ArgFormatter,
            description=
            """
            *Image generation*
                 invoke> a fantastic alien landscape -W576 -H512 -s60 -n4

            *postprocessing*
                !fix applies upscaling/facefixing to a previously-generated image.
                invoke> !fix 0000045.4829112.png -G1 -U4 -ft codeformer

            *History manipulation*
            !fetch retrieves the command used to generate an earlier image. Provide
            a directory wildcard and the name of a file to write and all the commands
            used to generate the images in the directory will be written to that file.
                invoke> !fetch 0000015.8929913.png
                invoke> a fantastic alien landscape -W 576 -H 512 -s 60 -A plms -C 7.5
                invoke> !fetch /path/to/images/*.png prompts.txt
 
            !replay /path/to/prompts.txt
            Replays all the prompts contained in the file prompts.txt.

            !history lists all the commands issued during the current session.

            !NN retrieves the NNth command from the history

            *Model manipulation*
            !models                                   -- list models in configs/models.yaml
            !switch <model_name>                      -- switch to model named <model_name>
            !import_model /path/to/weights/file.ckpt  -- adds a .ckpt model to your config
            !import_model http://path_to_model.ckpt   -- downloads and adds a .ckpt model to your config
            !import_model hakurei/waifu-diffusion     -- downloads and adds a diffusers model to your config
            !optimize_model <model_name>              -- converts a .ckpt model to a diffusers model
            !convert_model /path/to/weights/file.ckpt -- converts a .ckpt file path to a diffusers model
            !edit_model <model_name>                  -- edit a model's description
            !del_model <model_name>                   -- delete a model
            """
        )
        render_group     = parser.add_argument_group('General rendering')
        img2img_group    = parser.add_argument_group('Image-to-image and inpainting')
        inpainting_group    = parser.add_argument_group('Inpainting')
        outpainting_group    = parser.add_argument_group('Outpainting and outcropping')
        variation_group  = parser.add_argument_group('Creating and combining variations')
        postprocessing_group   = parser.add_argument_group('Post-processing')
        special_effects_group  = parser.add_argument_group('Special effects')
        deprecated_group = parser.add_argument_group('Deprecated options')
        render_group.add_argument(
            '--prompt',
            default='',
            help='prompt string',
        )
        render_group.add_argument(
            '-s',
            '--steps',
            type=int,
            help='Number of steps'
        )
        render_group.add_argument(
            '-S',
            '--seed',
            type=int,
            default=None,
            help='Image seed; a +ve integer, or use -1 for the previous seed, -2 for the one before that, etc',
        )
        render_group.add_argument(
            '-n',
            '--iterations',
            type=int,
            default=1,
            help='Number of samplings to perform (slower, but will provide seeds for individual images)',
        )
        render_group.add_argument(
            '-W',
            '--width',
            type=int,
            help='Image width, multiple of 64',
        )
        render_group.add_argument(
            '-H',
            '--height',
            type=int,
            help='Image height, multiple of 64',
        )
        render_group.add_argument(
            '-C',
            '--cfg_scale',
            type=float,
            help='Classifier free guidance (CFG) scale - higher numbers cause generator to "try" harder.',
        )
        render_group.add_argument(
            '--threshold',
            default=0.0,
            type=float,
            help='Latent threshold for classifier free guidance (CFG) - prevent generator from "trying" too hard. Use positive values, 0 disables.',
        )
        render_group.add_argument(
            '--perlin',
            default=0.0,
            type=float,
            help='Perlin noise scale (0.0 - 1.0) - add perlin noise to the initialization instead of the usual gaussian noise.',
        )
        render_group.add_argument(
            '--fnformat',
            default='{prefix}.{seed}.png',
            type=str,
            help='Overwrite the filename format. You can use any argument as wildcard enclosed in curly braces. Default is {prefix}.{seed}.png',
        )
        render_group.add_argument(
            '--grid',
            '-g',
            action=argparse.BooleanOptionalAction,
            help='generate a grid'
        )
        render_group.add_argument(
            '-i',
            '--individual',
            action='store_true',
            help='override command-line --grid setting and generate individual images'
        )
        render_group.add_argument(
            '-x',
            '--skip_normalize',
            action='store_true',
            help='Skip subprompt weight normalization',
        )
        render_group.add_argument(
            '-A',
            '-m',
            '--sampler',
            dest='sampler_name',
            type=str,
            choices=SAMPLER_CHOICES,
            metavar='SAMPLER_NAME',
            help=f'Switch to a different sampler. Supported samplers: {", ".join(SAMPLER_CHOICES)}',
        )
        render_group.add_argument(
            '-t',
            '--log_tokenization',
            action='store_true',
            help='shows how the prompt is split into tokens'
        )
        render_group.add_argument(
            '--outdir',
            '-o',
            type=str,
            help='Directory to save generated images and a log of prompts and seeds',
        )
        render_group.add_argument(
            '--hires_fix',
            action='store_true',
            dest='hires_fix',
            help='Create hires image using img2img to prevent duplicated objects'
        )
        render_group.add_argument(
            '--save_intermediates',
            type=int,
            default=0,
            dest='save_intermediates',
            help='Save every nth intermediate image into an "intermediates" directory within the output directory'
        )
        render_group.add_argument(
            '--png_compression','-z',
            type=int,
            default=6,
            choices=range(0,10),
            dest='png_compression',
            help='level of PNG compression, from 0 (none) to 9 (maximum). [6]'
        )
        render_group.add_argument(
            '--karras_max',
            type=int,
            default=None,
            help="control the point at which the K* samplers will shift from using the Karras noise schedule (good for low step counts) to the LatentDiffusion noise schedule (good for high step counts). Set to 0 to use LatentDiffusion for all step values, and to a high value (e.g. 1000) to use Karras for all step values. [29]."
        )
        img2img_group.add_argument(
            '-I',
            '--init_img',
            type=str,
            help='Path to input image for img2img mode (supersedes width and height)',
        )
        img2img_group.add_argument(
            '-tm',
            '--text_mask',
            nargs='+',
            type=str,
            help='Use the clipseg classifier to generate the mask area for inpainting. Provide a description of the area to mask ("a mug"), optionally followed by the confidence level threshold (0-1.0; defaults to 0.5).',
            default=None,
        )
        img2img_group.add_argument(
            '--init_color',
            type=str,
            help='Path to reference image for color correction (used for repeated img2img and inpainting)'
        )
        img2img_group.add_argument(
            '-T',
            '-fit',
            '--fit',
            action='store_true',
            help='If specified, will resize the input image to fit within the dimensions of width x height (512x512 default)',
        )
        img2img_group.add_argument(
            '-f',
            '--strength',
            type=float,
            help='img2img strength for noising/unnoising. 0.0 preserves image exactly, 1.0 replaces it completely',
        )
        inpainting_group.add_argument(
            '-M',
            '--init_mask',
            type=str,
            help='Path to input mask for inpainting mode (supersedes width and height)',
        )
        inpainting_group.add_argument(
            '--invert_mask',
            action='store_true',
            help='Invert the mask',
        )
        inpainting_group.add_argument(
            '-r',
            '--inpaint_replace',
            type=float,
            default=0.0,
            help='when inpainting, adjust how aggressively to replace the part of the picture under the mask, from 0.0 (a gentle merge) to 1.0 (replace entirely)',
        )
        outpainting_group.add_argument(
            '-c',
            '--outcrop',
            nargs='+',
            type=str,
            metavar=('direction','pixels'),
            help='Outcrop the image with one or more direction/pixel pairs: e.g. -c top 64 bottom 128 left 64 right 64',
        )
        outpainting_group.add_argument(
            '--force_outpaint',
            action='store_true',
            default=False,
            help='Force outpainting if you have no inpainting mask to pass',
        )
        outpainting_group.add_argument(
            '--seam_size',
            type=int,
            default=0,
            help='When outpainting, size of the mask around the seam between original and outpainted image',
        )
        outpainting_group.add_argument(
            '--seam_blur',
            type=int,
            default=0,
            help='When outpainting, the amount to blur the seam inwards',
        )
        outpainting_group.add_argument(
            '--seam_strength',
            type=float,
            default=0.7,
            help='When outpainting, the img2img strength to use when filling the seam. Values around 0.7 work well',
        )
        outpainting_group.add_argument(
            '--seam_steps',
            type=int,
            default=10,
            help='When outpainting, the number of steps to use to fill the seam. Low values (~10) work well',
        )
        outpainting_group.add_argument(
            '--tile_size',
            type=int,
            default=32,
            help='When outpainting, the tile size to use for filling outpaint areas',
        )
        postprocessing_group.add_argument(
            '--new_prompt',
            type=str,
            help='Change the text prompt applied during postprocessing (default, use original generation prompt)',
        )
        postprocessing_group.add_argument(
            '-ft',
            '--facetool',
            type=str,
            default='gfpgan',
            help='Select the face restoration AI to use: gfpgan, codeformer',
        )
        postprocessing_group.add_argument(
            '-G',
            '--facetool_strength',
            '--gfpgan_strength',
            type=float,
            help='The strength at which to apply the face restoration to the result.',
            default=0.0,
        )
        postprocessing_group.add_argument(
            '-cf',
            '--codeformer_fidelity',
            type=float,
            help='Used along with CodeFormer. Takes values between 0 and 1. 0 produces high quality but low accuracy. 1 produces high accuracy but low quality.',
            default=0.75
        )
        postprocessing_group.add_argument(
            '-U',
            '--upscale',
            nargs='+',
            type=float,
            help='Scale factor (1, 2, 3, 4, etc..) for upscaling final output followed by upscaling strength (0-1.0). If strength not specified, defaults to 0.75',
            default=None,
        )
        postprocessing_group.add_argument(
            '--save_original',
            '-save_orig',
            action='store_true',
            help='Save original. Use it when upscaling to save both versions.',
        )
        postprocessing_group.add_argument(
            '--embiggen',
            '-embiggen',
            nargs='+',
            type=float,
            help='Arbitrary upscaling using img2img. Provide scale factor (0.75), optionally followed by strength (0.75) and tile overlap proportion (0.25).',
            default=None,
        )
        postprocessing_group.add_argument(
            '--embiggen_tiles',
            '-embiggen_tiles',
            nargs='+',
            type=int,
            help='For embiggen, provide list of tiles to process and replace onto the image e.g. `1 3 5`.',
            default=None,
        )
        postprocessing_group.add_argument(
            '--embiggen_strength',
            '-embiggen_strength',
            type=float,
            help='The strength of the embiggen img2img step, defaults to 0.4',
            default=None,
        )
        special_effects_group.add_argument(
            '--seamless',
            action='store_true',
            help='Change the model to seamless tiling (circular) mode',
        )
        special_effects_group.add_argument(
            '--seamless_axes',
            default=['x', 'y'],
            type=list[str],
            help='Specify which axes to use circular convolution on.',
        )
        variation_group.add_argument(
            '-v',
            '--variation_amount',
            default=0.0,
            type=float,
            help='If > 0, generates variations on the initial seed instead of random seeds per iteration. Must be between 0 and 1. Higher values will be more different.'
        )
        variation_group.add_argument(
            '-V',
            '--with_variations',
            default=None,
            type=str,
            help='list of variations to apply, in the format `seed:weight,seed:weight,...'
        )
        render_group.add_argument(
            '--use_mps_noise',
            action='store_true',
            dest='use_mps_noise',
            help='Simulate noise on M1 systems to get the same results'
        )
        deprecated_group.add_argument(
            '-D',
            '--out_direction',
            nargs='+',
            type=str,
            metavar=('direction', 'pixels'),
            help='Older outcropping system. Direction to extend the given image (left|right|top|bottom). If a distance pixel value is not specified it defaults to half the image size'
        )
        return parser

def format_metadata(**kwargs):
    print('format_metadata() is deprecated. Please use metadata_dumps()')
    return metadata_dumps(kwargs)

def metadata_dumps(opt,
                   seeds=[],
                   model_hash=None,
                   postprocessing=None):
    '''
    Given an Args object, returns a dict containing the keys and
    structure of the proposed stable diffusion metadata standard
    https://github.com/lstein/stable-diffusion/discussions/392
    This is intended to be turned into JSON and stored in the
    "sd
    '''

    # top-level metadata minus `image` or `images`
    metadata = {
        'model'       : 'stable diffusion',
        'model_id'    : opt.model,
        'model_hash'  : model_hash,
        'app_id'      : ldm.invoke.__app_id__,
        'app_version' : ldm.invoke.__version__,
    }

    # # add some RFC266 fields that are generated internally, and not as
    # # user args
    image_dict = opt.to_dict(
         postprocessing=postprocessing
    )

    # remove any image keys not mentioned in RFC #266
    rfc266_img_fields = ['type','postprocessing','sampler','prompt','seed','variations','steps',
                         'cfg_scale','threshold','perlin','step_number','width','height','extra','strength','seamless'
                         'init_img','init_mask','facetool','facetool_strength','upscale']
    rfc_dict ={}

    for item in image_dict.items():
        key,value = item
        if key in rfc266_img_fields:
            rfc_dict[key] = value

    # semantic drift
    rfc_dict['sampler']  = image_dict.get('sampler_name',None)

    # display weighted subprompts (liable to change)
    if opt.prompt:
        subprompts = split_weighted_subprompts(opt.prompt)
        subprompts = [{'prompt':x[0],'weight':x[1]} for x in subprompts]
        rfc_dict['prompt'] = subprompts

    # 'variations' should always exist and be an array, empty or consisting of {'seed': seed, 'weight': weight} pairs
    rfc_dict['variations'] = [{'seed':x[0],'weight':x[1]} for x in opt.with_variations] if opt.with_variations else []

    # if variations are present then we need to replace 'seed' with 'orig_seed'
    if hasattr(opt,'first_seed'):
        rfc_dict['seed'] = opt.first_seed

    if opt.init_img:
        rfc_dict['type']            = 'img2img'
        rfc_dict['strength_steps']  = rfc_dict.pop('strength')
        rfc_dict['orig_hash']       = calculate_init_img_hash(opt.init_img)
        rfc_dict['inpaint_replace'] = opt.inpaint_replace
    else:
        rfc_dict['type']  = 'txt2img'
        rfc_dict.pop('strength')

    if len(seeds)==0 and opt.seed:
        seeds=[opt.seed]

    if opt.grid:
        images = []
        for seed in seeds:
            rfc_dict['seed'] = seed
            images.append(copy.copy(rfc_dict))
        metadata['images'] = images
    else:
        # there should only ever be a single seed if we did not generate a grid
        assert len(seeds) == 1, 'Expected a single seed'
        rfc_dict['seed'] = seeds[0]
        metadata['image'] = rfc_dict

    return metadata

@functools.lru_cache(maxsize=50)
def args_from_png(png_file_path) -> list[Args]:
    '''
    Given the path to a PNG file created by invoke.py,
    retrieves a list of Args objects containing the image
    data.
    '''
    try:
        meta = ldm.invoke.pngwriter.retrieve_metadata(png_file_path)
    except AttributeError:
        return [legacy_metadata_load({},png_file_path)]

    try:
        return metadata_loads(meta)
    except:
        return [legacy_metadata_load(meta,png_file_path)]

@functools.lru_cache(maxsize=50)
def metadata_from_png(png_file_path) -> Args:
    '''
    Given the path to a PNG file created by dream.py, retrieves
    an Args object containing the image metadata. Note that this
    returns a single Args object, not multiple.
    '''
    args_list = args_from_png(png_file_path)
    return args_list[0] if len(args_list)>0 else Args()  # empty args

def dream_cmd_from_png(png_file_path):
    opt = metadata_from_png(png_file_path)
    return opt.dream_prompt_str()

def metadata_loads(metadata) -> list:
    '''
    Takes the dictionary corresponding to RFC266 (https://github.com/lstein/stable-diffusion/issues/266)
    and returns a series of opt objects for each of the images described in the dictionary. Note that this
    returns a list, and not a single object. See metadata_from_png() for a more convenient function for
    files that contain a single image.
    '''
    results = []
    try:
        if 'images' in metadata['sd-metadata']:
            images = metadata['sd-metadata']['images']
        else:
            images = [metadata['sd-metadata']['image']]
        for image in images:
            # repack the prompt and variations
            if 'prompt' in image:
                image['prompt']     = repack_prompt(image['prompt'])
            if 'variations' in image:
                image['variations'] = ','.join([':'.join([str(x['seed']),str(x['weight'])]) for x in image['variations']])
            # fix a bit of semantic drift here
            image['sampler_name']=image.pop('sampler')
            opt = Args()
            opt._cmd_switches = Namespace(**image)
            results.append(opt)
    except Exception:
        import sys, traceback
        print('>> could not read metadata',file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
    return results

def repack_prompt(prompt_list:list)->str:
    # in the common case of no weighting syntax, just return the prompt as is
    if len(prompt_list) > 1:
        return ','.join([':'.join([x['prompt'], str(x['weight'])]) for x in prompt_list])
    else:
        return prompt_list[0]['prompt']

# image can either be a file path on disk or a base64-encoded
# representation of the file's contents
def calculate_init_img_hash(image_string):
    prefix = 'data:image/png;base64,'
    hash   = None
    if image_string.startswith(prefix):
        imagebase64 = image_string[len(prefix):]
        imagedata   = base64.b64decode(imagebase64)
        with open('outputs/test.png','wb') as file:
            file.write(imagedata)
        sha = hashlib.sha256()
        sha.update(imagedata)
        hash = sha.hexdigest()
    else:
        hash = sha256(image_string)
    return hash

# Bah. This should be moved somewhere else...
def sha256(path):
    sha = hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha.update(data)
    return sha.hexdigest()

def legacy_metadata_load(meta,pathname) -> Args:
    opt = Args()
    if 'Dream' in meta and len(meta['Dream']) > 0:
        dream_prompt = meta['Dream']
        opt.parse_cmd(dream_prompt)
    else:               # if nothing else, we can get the seed
        match = re.search('\d+\.(\d+)',pathname)
        if match:
            seed = match.groups()[0]
            opt.seed = seed
        else:
            opt.prompt = ''
            opt.seed = 0
    return opt

