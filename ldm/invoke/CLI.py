import os
import re
import sys
import shlex
import copy
import warnings
import time
import traceback
import yaml

from ldm.invoke.globals import Globals
from ldm.generate import Generate
from ldm.invoke.prompt_parser import PromptParser
from ldm.invoke.readline import get_completer, Completer
from ldm.invoke.args import Args, metadata_dumps, metadata_from_png, dream_cmd_from_png
from ldm.invoke.pngwriter import PngWriter, retrieve_metadata, write_metadata
from ldm.invoke.image_util import make_grid
from ldm.invoke.log import write_log
from ldm.invoke.concepts_lib import Concepts
from omegaconf import OmegaConf
from pathlib import Path
import pyparsing
import ldm.invoke

# global used in multiple functions (fix)
infile = None

def main():
    """Initialize command-line parsers and the diffusion model"""
    global infile

    opt  = Args()
    args = opt.parse_args()
    if not args:
        sys.exit(-1)

    if args.laion400m:
        print('--laion400m flag has been deprecated. Please use --model laion400m instead.')
        sys.exit(-1)
    if args.weights:
        print('--weights argument has been deprecated. Please edit ./configs/models.yaml, and select the weights using --model instead.')
        sys.exit(-1)
    if args.max_loaded_models is not None:
        if args.max_loaded_models <= 0:
            print('--max_loaded_models must be >= 1; using 1')
            args.max_loaded_models = 1

    if not args.conf:
        if not os.path.exists(os.path.join(Globals.root,'configs','models.yaml')):
            print(f"\n** Error. The file {os.path.join(Globals.root,'configs','models.yaml')} could not be found.")
            print(f'** Please check the location of your invokeai directory and use the --root_dir option to point to the correct path.')
            print(f'** This script will now exit.')
            sys.exit(-1)

    print(f'>> {ldm.invoke.__app_name__} {ldm.invoke.__version__}')
    print(f'>> InvokeAI runtime directory is "{Globals.root}"')

    # loading here to avoid long delays on startup
    from ldm.generate import Generate

    # these two lines prevent a horrible warning message from appearing
    # when the frozen CLIP tokenizer is imported
    import transformers
    transformers.logging.set_verbosity_error()

    # Loading Face Restoration and ESRGAN Modules
    gfpgan,codeformer,esrgan = load_face_restoration(opt)

    # normalize the config directory relative to root
    if not os.path.isabs(opt.conf):
        opt.conf = os.path.normpath(os.path.join(Globals.root,opt.conf))

    if opt.embeddings:
        if not os.path.isabs(opt.embedding_path):
            embedding_path = os.path.normpath(os.path.join(Globals.root,opt.embedding_path))
        else:
            embedding_path = opt.embedding_path
    else:
        embedding_path = None

    # load the infile as a list of lines
    if opt.infile:
        try:
            if os.path.isfile(opt.infile):
                infile = open(opt.infile, 'r', encoding='utf-8')
            elif opt.infile == '-':  # stdin
                infile = sys.stdin
            else:
                raise FileNotFoundError(f'{opt.infile} not found.')
        except (FileNotFoundError, IOError) as e:
            print(f'{e}. Aborting.')
            sys.exit(-1)

    # creating a Generate object:
    try:
        gen = Generate(
            conf = opt.conf,
            model = opt.model,
            sampler_name = opt.sampler_name,
            embedding_path = embedding_path,
            full_precision = opt.full_precision,
            precision = opt.precision,
            gfpgan=gfpgan,
            codeformer=codeformer,
            esrgan=esrgan,
            free_gpu_mem=opt.free_gpu_mem,
            safety_checker=opt.safety_checker,
            max_loaded_models=opt.max_loaded_models,
            )
    except (FileNotFoundError, TypeError, AssertionError):
        emergency_model_reconfigure(opt)
        sys.exit(-1)
    except (IOError, KeyError) as e:
        print(f'{e}. Aborting.')
        sys.exit(-1)

    if opt.seamless:
        print(">> changed to seamless tiling mode")

    # preload the model
    try:
        gen.load_model()
    except AssertionError:
        emergency_model_reconfigure(opt)
        sys.exit(-1)

    # web server loops forever
    if opt.web or opt.gui:
        invoke_ai_web_server_loop(gen, gfpgan, codeformer, esrgan)
        sys.exit(0)

    if not infile:
        print(
            "\n* Initialization done! Awaiting your command (-h for help, 'q' to quit)"
        )

    try:
        main_loop(gen, opt)
    except KeyboardInterrupt:
        print(f'\nGoodbye!\nYou can start InvokeAI again by running the "invoke.bat" (or "invoke.sh") script from {Globals.root}')

# TODO: main_loop() has gotten busy. Needs to be refactored.
def main_loop(gen, opt):
    """prompt/read/execute loop"""
    global infile
    done = False
    doneAfterInFile = infile is not None
    path_filter = re.compile(r'[<>:"/\\|?*]')
    last_results = list()
    model_config = OmegaConf.load(opt.conf)

    # The readline completer reads history from the .dream_history file located in the
    # output directory specified at the time of script launch. We do not currently support
    # changing the history file midstream when the output directory is changed.
    completer   = get_completer(opt, models=list(model_config.keys()))
    set_default_output_dir(opt, completer)
    add_embedding_terms(gen, completer)
    output_cntr = completer.get_current_history_length()+1

    # os.pathconf is not available on Windows
    if hasattr(os, 'pathconf'):
        path_max = os.pathconf(opt.outdir, 'PC_PATH_MAX')
        name_max = os.pathconf(opt.outdir, 'PC_NAME_MAX')
    else:
        path_max = 260
        name_max = 255

    while not done:

        operation = 'generate'

        try:
            command = get_next_command(infile)
        except EOFError:
            done = infile is None or doneAfterInFile
            infile = None
            continue

        # skip empty lines
        if not command.strip():
            continue

        if command.startswith(('#', '//')):
            continue

        if len(command.strip()) == 1 and command.startswith('q'):
            done = True
            break

        if not command.startswith('!history'):
            completer.add_history(command)

        if command.startswith('!'):
            command, operation = do_command(command, gen, opt, completer)

        if operation is None:
            continue

        if opt.parse_cmd(command) is None:
            continue

        if opt.init_img:
            try:
                if not opt.prompt:
                    oldargs    = metadata_from_png(opt.init_img)
                    opt.prompt = oldargs.prompt
                    print(f'>> Retrieved old prompt "{opt.prompt}" from {opt.init_img}')
            except (OSError, AttributeError, KeyError):
                pass

        if len(opt.prompt) == 0:
            opt.prompt = ''

        # width and height are set by model if not specified
        if not opt.width:
            opt.width = gen.width
        if not opt.height:
            opt.height = gen.height

        # retrieve previous value of init image if requested
        if opt.init_img is not None and re.match('^-\\d+$', opt.init_img):
            try:
                opt.init_img = last_results[int(opt.init_img)][0]
                print(f'>> Reusing previous image {opt.init_img}')
            except IndexError:
                print(
                    f'>> No previous initial image at position {opt.init_img} found')
                opt.init_img = None
                continue

        # the outdir can change with each command, so we adjust it here
        set_default_output_dir(opt,completer)

        # try to relativize pathnames
        for attr in ('init_img','init_mask','init_color'):
            if getattr(opt,attr) and not os.path.exists(getattr(opt,attr)):
                basename = getattr(opt,attr)
                path     = os.path.join(opt.outdir,basename)
                setattr(opt,attr,path)

        # retrieve previous value of seed if requested
        # Exception: for postprocess operations negative seed values
        # mean "discard the original seed and generate a new one"
        # (this is a non-obvious hack and needs to be reworked)
        if opt.seed is not None and opt.seed < 0 and operation != 'postprocess':
            try:
                opt.seed = last_results[opt.seed][1]
                print(f'>> Reusing previous seed {opt.seed}')
            except IndexError:
                print(f'>> No previous seed at position {opt.seed} found')
                opt.seed = None
                continue

        if opt.strength is None:
            opt.strength = 0.75 if opt.out_direction is None else 0.83

        if opt.with_variations is not None:
            opt.with_variations = split_variations(opt.with_variations)

        if opt.prompt_as_dir and operation == 'generate':
            # sanitize the prompt to a valid folder name
            subdir = path_filter.sub('_', opt.prompt)[:name_max].rstrip(' .')

            # truncate path to maximum allowed length
            # 39 is the length of '######.##########.##########-##.png', plus two separators and a NUL
            subdir = subdir[:(path_max - 39 - len(os.path.abspath(opt.outdir)))]
            current_outdir = os.path.join(opt.outdir, subdir)

            print('Writing files to directory: "' + current_outdir + '"')

            # make sure the output directory exists
            if not os.path.exists(current_outdir):
                os.makedirs(current_outdir)
        else:
            if not os.path.exists(opt.outdir):
                os.makedirs(opt.outdir)
            current_outdir = opt.outdir

        # Here is where the images are actually generated!
        last_results = []
        try:
            file_writer      = PngWriter(current_outdir)
            results          = []  # list of filename, prompt pairs
            grid_images      = dict()  # seed -> Image, only used if `opt.grid`
            prior_variations = opt.with_variations or []
            prefix = file_writer.unique_prefix()
            step_callback = make_step_callback(gen, opt, prefix) if opt.save_intermediates > 0 else None

            def image_writer(image, seed, upscaled=False, first_seed=None, use_prefix=None, prompt_in=None, attention_maps_image=None):
                # note the seed is the seed of the current image
                # the first_seed is the original seed that noise is added to
                # when the -v switch is used to generate variations
                nonlocal prior_variations
                nonlocal prefix

                path = None
                if opt.grid:
                    grid_images[seed] = image

                elif operation == 'mask':
                    filename = f'{prefix}.{use_prefix}.{seed}.png'
                    tm = opt.text_mask[0]
                    th = opt.text_mask[1] if len(opt.text_mask)>1 else 0.5
                    formatted_dream_prompt = f'!mask {opt.input_file_path} -tm {tm} {th}'
                    path = file_writer.save_image_and_prompt_to_png(
                        image           = image,
                        dream_prompt    = formatted_dream_prompt,
                        metadata        = {},
                        name      = filename,
                        compress_level = opt.png_compression,
                    )
                    results.append([path, formatted_dream_prompt])

                else:
                    if use_prefix is not None:
                        prefix = use_prefix
                    postprocessed = upscaled if upscaled else operation=='postprocess'
                    opt.prompt = gen.concept_lib().replace_triggers_with_concepts(opt.prompt or prompt_in)  # to avoid the problem of non-unique concept triggers
                    filename, formatted_dream_prompt = prepare_image_metadata(
                        opt,
                        prefix,
                        seed,
                        operation,
                        prior_variations,
                        postprocessed,
                        first_seed
                    )
                    path = file_writer.save_image_and_prompt_to_png(
                        image           = image,
                        dream_prompt    = formatted_dream_prompt,
                        metadata        = metadata_dumps(
                            opt,
                            seeds      = [seed if opt.variation_amount==0 and len(prior_variations)==0 else first_seed],
                            model_hash = gen.model_hash,
                        ),
                        name      = filename,
                        compress_level = opt.png_compression,
                    )

                    # update rfc metadata
                    if operation == 'postprocess':
                        tool = re.match('postprocess:(\w+)',opt.last_operation).groups()[0]
                        add_postprocessing_to_metadata(
                            opt,
                            opt.input_file_path,
                            filename,
                            tool,
                            formatted_dream_prompt,
                        )

                    if (not postprocessed) or opt.save_original:
                        # only append to results if we didn't overwrite an earlier output
                        results.append([path, formatted_dream_prompt])

                # so that the seed autocompletes (on linux|mac when -S or --seed specified
                if completer and operation == 'generate':
                    completer.add_seed(seed)
                    completer.add_seed(first_seed)
                last_results.append([path, seed])

            if operation == 'generate':
                catch_ctrl_c = infile is None # if running interactively, we catch keyboard interrupts
                opt.last_operation='generate'
                try:
                    gen.prompt2image(
                        image_callback=image_writer,
                        step_callback=step_callback,
                        catch_interrupts=catch_ctrl_c,
                        **vars(opt)
                    )
                except (PromptParser.ParsingException, pyparsing.ParseException) as e:
                    print('** An error occurred while processing your prompt **')
                    print(f'** {str(e)} **')
            elif operation == 'postprocess':
                print(f'>> fixing {opt.prompt}')
                opt.last_operation = do_postprocess(gen,opt,image_writer)

            elif operation == 'mask':
                print(f'>> generating masks from {opt.prompt}')
                do_textmask(gen, opt, image_writer)

            if opt.grid and len(grid_images) > 0:
                grid_img   = make_grid(list(grid_images.values()))
                grid_seeds = list(grid_images.keys())
                first_seed = last_results[0][1]
                filename   = f'{prefix}.{first_seed}.png'
                formatted_dream_prompt  = opt.dream_prompt_str(seed=first_seed,grid=True,iterations=len(grid_images))
                formatted_dream_prompt += f' # {grid_seeds}'
                metadata = metadata_dumps(
                    opt,
                    seeds      = grid_seeds,
                    model_hash = gen.model_hash
                    )
                path = file_writer.save_image_and_prompt_to_png(
                    image        = grid_img,
                    dream_prompt = formatted_dream_prompt,
                    metadata     = metadata,
                    name         = filename
                )
                results = [[path, formatted_dream_prompt]]

        except AssertionError as e:
            print(e)
            continue

        except OSError as e:
            print(e)
            continue

        print('Outputs:')
        log_path = os.path.join(current_outdir, 'invoke_log')
        output_cntr = write_log(results, log_path ,('txt', 'md'), output_cntr)
        print()


    print(f'\nGoodbye!\nYou can start InvokeAI again by running the "invoke.bat" (or "invoke.sh") script from {Globals.root}')

# TO DO: remove repetitive code and the awkward command.replace() trope
# Just do a simple parse of the command!
def do_command(command:str, gen, opt:Args, completer) -> tuple:
    global infile
    operation = 'generate'   # default operation, alternative is 'postprocess'

    if command.startswith('!dream'):   # in case a stored prompt still contains the !dream command
        command = command.replace('!dream ','',1)

    elif command.startswith('!fix'):
        command = command.replace('!fix ','',1)
        operation = 'postprocess'

    elif command.startswith('!mask'):
        command = command.replace('!mask ','',1)
        operation = 'mask'

    elif command.startswith('!switch'):
        model_name = command.replace('!switch ','',1)
        gen.set_model(model_name)
        add_embedding_terms(gen, completer)
        completer.add_history(command)
        operation = None

    elif command.startswith('!models'):
        gen.model_cache.print_models()
        completer.add_history(command)
        operation = None

    elif command.startswith('!import'):
        path = shlex.split(command)
        if len(path) < 2:
            print('** please provide a path to a .ckpt or .vae model file')
        elif not os.path.exists(path[1]):
            print(f'** {path[1]}: file not found')
        else:
            add_weights_to_config(path[1], gen, opt, completer)
        completer.add_history(command)
        operation = None

    elif command.startswith('!edit'):
        path = shlex.split(command)
        if len(path) < 2:
            print('** please provide the name of a model')
        else:
            edit_config(path[1], gen, opt, completer)
        completer.add_history(command)
        operation = None

    elif command.startswith('!del'):
        path = shlex.split(command)
        if len(path) < 2:
            print('** please provide the name of a model')
        else:
            del_config(path[1], gen, opt, completer)
        completer.add_history(command)
        operation = None

    elif command.startswith('!fetch'):
        file_path = command.replace('!fetch','',1).strip()
        retrieve_dream_command(opt,file_path,completer)
        completer.add_history(command)
        operation = None

    elif command.startswith('!replay'):
        file_path = command.replace('!replay','',1).strip()
        if infile is None and os.path.isfile(file_path):
            infile = open(file_path, 'r', encoding='utf-8')
        completer.add_history(command)
        operation = None

    elif command.startswith('!history'):
        completer.show_history()
        operation = None

    elif command.startswith('!search'):
        search_str = command.replace('!search','',1).strip()
        completer.show_history(search_str)
        operation = None

    elif command.startswith('!clear'):
        completer.clear_history()
        operation = None

    elif re.match('^!(\d+)',command):
        command_no = re.match('^!(\d+)',command).groups()[0]
        command    = completer.get_line(int(command_no))
        completer.set_line(command)
        operation = None

    else:  # not a recognized command, so give the --help text
        command = '-h'
    return command, operation

def set_default_output_dir(opt:Args, completer:Completer):
    '''
    If opt.outdir is relative, we add the root directory to it
    normalize the outdir relative to root and make sure it exists.
    '''
    if not os.path.isabs(opt.outdir):
        opt.outdir=os.path.normpath(os.path.join(Globals.root,opt.outdir))
    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)
    completer.set_default_dir(opt.outdir)


def add_weights_to_config(model_path:str, gen, opt, completer):
    print(f'>> Model import in process. Please enter the values needed to configure this model:')
    print()

    new_config = {}
    new_config['weights'] = model_path

    done = False
    while not done:
        model_name = input('Short name for this model: ')
        if not re.match('^[\w._-]+$',model_name):
            print('** model name must contain only words, digits and the characters [._-] **')
        else:
            done = True
    new_config['description'] = input('Description of this model: ')

    completer.complete_extensions(('.yaml','.yml'))
    completer.linebuffer = 'configs/stable-diffusion/v1-inference.yaml'

    done = False
    while not done:
        new_config['config'] = input('Configuration file for this model: ')
        done = os.path.exists(new_config['config'])

    done = False
    completer.complete_extensions(('.vae.pt','.vae','.ckpt'))
    while not done:
        vae = input('VAE autoencoder file for this model [None]: ')
        if os.path.exists(vae):
            new_config['vae'] = vae
            done = True
        else:
            done = len(vae)==0

    completer.complete_extensions(None)

    for field in ('width','height'):
        done = False
        while not done:
            try:
                completer.linebuffer = '512'
                value = int(input(f'Default image {field}: '))
                assert value >= 64 and value <= 2048
                new_config[field] = value
                done = True
            except:
                print('** Please enter a valid integer between 64 and 2048')

    make_default = input('Make this the default model? [n] ') in ('y','Y')

    if write_config_file(opt.conf, gen, model_name, new_config, make_default=make_default):
        completer.add_model(model_name)

def del_config(model_name:str, gen, opt, completer):
    current_model = gen.model_name
    if model_name == current_model:
        print("** Can't delete active model. !switch to another model first. **")
        return
    gen.model_cache.del_model(model_name)
    gen.model_cache.commit(opt.conf)
    print(f'** {model_name} deleted')
    completer.del_model(model_name)

def edit_config(model_name:str, gen, opt, completer):
    config = gen.model_cache.config

    if model_name not in config:
        print(f'** Unknown model {model_name}')
        return

    print(f'\n>> Editing model {model_name} from configuration file {opt.conf}')

    conf = config[model_name]
    new_config = {}
    completer.complete_extensions(('.yaml','.yml','.ckpt','.vae.pt'))
    for field in ('description', 'weights', 'vae', 'config', 'width','height'):
        completer.linebuffer = str(conf[field]) if field in conf else ''
        new_value = input(f'{field}: ')
        new_config[field] = int(new_value) if field in ('width','height') else new_value
    make_default = input('Make this the default model? [n] ') in ('y','Y')
    completer.complete_extensions(None)
    write_config_file(opt.conf, gen, model_name, new_config, clobber=True, make_default=make_default)

def write_config_file(conf_path, gen, model_name, new_config, clobber=False, make_default=False):
    current_model = gen.model_name

    op = 'modify' if clobber else 'import'
    print('\n>> New configuration:')
    if make_default:
        new_config['default'] = True
    print(yaml.dump({model_name:new_config}))
    if input(f'OK to {op} [n]? ') not in ('y','Y'):
        return False

    try:
        print('>> Verifying that new model loads...')
        gen.model_cache.add_model(model_name, new_config, clobber)
        assert gen.set_model(model_name) is not None, 'model failed to load'
    except AssertionError as e:
        print(f'** aborting **')
        gen.model_cache.del_model(model_name)
        return False

    if make_default:
        print('making this default')
        gen.model_cache.set_default_model(model_name)

    gen.model_cache.commit(conf_path)

    do_switch = input(f'Keep model loaded? [y]')
    if len(do_switch)==0 or do_switch[0] in ('y','Y'):
        pass
    else:
        gen.set_model(current_model)
    return True

def do_textmask(gen, opt, callback):
    image_path = opt.prompt
    if not os.path.exists(image_path):
        image_path = os.path.join(opt.outdir,image_path)
    assert os.path.exists(image_path), '** "{opt.prompt}" not found. Please enter the name of an existing image file to mask **'
    assert opt.text_mask is not None and len(opt.text_mask) >= 1, '** Please provide a text mask with -tm **'
    opt.input_file_path = image_path
    tm = opt.text_mask[0]
    threshold = float(opt.text_mask[1]) if len(opt.text_mask) > 1  else 0.5
    gen.apply_textmask(
        image_path = image_path,
        prompt = tm,
        threshold = threshold,
        callback = callback,
    )

def do_postprocess (gen, opt, callback):
    file_path = opt.prompt      # treat the prompt as the file pathname
    if opt.new_prompt is not None:
        opt.prompt = opt.new_prompt
    else:
        opt.prompt = None

    if os.path.dirname(file_path) == '': #basename given
        file_path = os.path.join(opt.outdir,file_path)

    opt.input_file_path = file_path

    tool=None
    if opt.facetool_strength > 0:
        tool = opt.facetool
    elif opt.embiggen:
        tool = 'embiggen'
    elif opt.upscale:
        tool = 'upscale'
    elif opt.out_direction:
        tool = 'outpaint'
    elif opt.outcrop:
        tool = 'outcrop'
    opt.save_original  = True # do not overwrite old image!
    opt.last_operation = f'postprocess:{tool}'
    try:
        gen.apply_postprocessor(
            image_path      = file_path,
            tool            = tool,
            facetool_strength = opt.facetool_strength,
            codeformer_fidelity = opt.codeformer_fidelity,
            save_original       = opt.save_original,
            upscale             = opt.upscale,
            out_direction       = opt.out_direction,
            outcrop             = opt.outcrop,
            callback            = callback,
            opt                 = opt,
        )
    except OSError:
        print(traceback.format_exc(), file=sys.stderr)
        print(f'** {file_path}: file could not be read')
        return
    except (KeyError, AttributeError):
        print(traceback.format_exc(), file=sys.stderr)
        return
    return opt.last_operation

def add_postprocessing_to_metadata(opt,original_file,new_file,tool,command):
    original_file = original_file if os.path.exists(original_file) else os.path.join(opt.outdir,original_file)
    new_file       = new_file     if os.path.exists(new_file)      else os.path.join(opt.outdir,new_file)
    try:
        meta = retrieve_metadata(original_file)['sd-metadata']
    except AttributeError:
        try:
            meta = retrieve_metadata(new_file)['sd-metadata']
        except AttributeError:
            meta = {}

    if 'image' not in meta:
        meta = metadata_dumps(opt,seeds=[opt.seed])['image']
        meta['image'] = {}
    img_data = meta.get('image')
    pp = img_data.get('postprocessing',[]) or []
    pp.append(
        {
            'tool':tool,
            'dream_command':command,
        }
    )
    meta['image']['postprocessing'] = pp
    write_metadata(new_file,meta)

def prepare_image_metadata(
        opt,
        prefix,
        seed,
        operation='generate',
        prior_variations=[],
        postprocessed=False,
        first_seed=None
):

    if postprocessed and opt.save_original:
        filename = choose_postprocess_name(opt,prefix,seed)
    else:
        wildcards = dict(opt.__dict__)
        wildcards['prefix'] = prefix
        wildcards['seed'] = seed
        try:
            filename = opt.fnformat.format(**wildcards)
        except KeyError as e:
            print(f'** The filename format contains an unknown key \'{e.args[0]}\'. Will use \'{{prefix}}.{{seed}}.png\' instead')
            filename = f'{prefix}.{seed}.png'
        except IndexError as e:
            print(f'** The filename format is broken or complete. Will use \'{{prefix}}.{{seed}}.png\' instead')
            filename = f'{prefix}.{seed}.png'

    if opt.variation_amount > 0:
        first_seed             = first_seed or seed
        this_variation         = [[seed, opt.variation_amount]]
        opt.with_variations    = prior_variations + this_variation
        formatted_dream_prompt = opt.dream_prompt_str(seed=first_seed)
    elif len(prior_variations) > 0:
        formatted_dream_prompt = opt.dream_prompt_str(seed=first_seed)
    elif operation == 'postprocess':
        formatted_dream_prompt = '!fix '+opt.dream_prompt_str(seed=seed,prompt=opt.input_file_path)
    else:
        formatted_dream_prompt = opt.dream_prompt_str(seed=seed)
    return filename,formatted_dream_prompt

def choose_postprocess_name(opt,prefix,seed) -> str:
    match      = re.search('postprocess:(\w+)',opt.last_operation)
    if match:
        modifier = match.group(1)   # will look like "gfpgan", "upscale", "outpaint" or "embiggen"
    else:
        modifier = 'postprocessed'

    counter   = 0
    filename  = None
    available = False
    while not available:
        if counter == 0:
            filename = f'{prefix}.{seed}.{modifier}.png'
        else:
            filename = f'{prefix}.{seed}.{modifier}-{counter:02d}.png'
        available = not os.path.exists(os.path.join(opt.outdir,filename))
        counter += 1
    return filename

def get_next_command(infile=None) -> str:  # command string
    if infile is None:
        command = input('invoke> ')
    else:
        command = infile.readline()
        if not command:
            raise EOFError
        else:
            command = command.strip()
        if len(command)>0:
            print(f'#{command}')
    return command

def invoke_ai_web_server_loop(gen: Generate, gfpgan, codeformer, esrgan):
    print('\n* --web was specified, starting web server...')
    from backend.invoke_ai_web_server import InvokeAIWebServer
    # Change working directory to the stable-diffusion directory
    os.chdir(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    )

    invoke_ai_web_server = InvokeAIWebServer(generate=gen, gfpgan=gfpgan, codeformer=codeformer, esrgan=esrgan)

    try:
        invoke_ai_web_server.run()
    except KeyboardInterrupt:
        pass

def add_embedding_terms(gen,completer):
    '''
    Called after setting the model, updates the autocompleter with
    any terms loaded by the embedding manager.
    '''
    completer.add_embedding_terms(gen.model.embedding_manager.list_terms())

def split_variations(variations_string) -> list:
    # shotgun parsing, woo
    parts = []
    broken = False  # python doesn't have labeled loops...
    for part in variations_string.split(','):
        seed_and_weight = part.split(':')
        if len(seed_and_weight) != 2:
            print(f'** Could not parse with_variation part "{part}"')
            broken = True
            break
        try:
            seed   = int(seed_and_weight[0])
            weight = float(seed_and_weight[1])
        except ValueError:
            print(f'** Could not parse with_variation part "{part}"')
            broken = True
            break
        parts.append([seed, weight])
    if broken:
        return None
    elif len(parts) == 0:
        return None
    else:
        return parts

def load_face_restoration(opt):
    try:
        gfpgan, codeformer, esrgan = None, None, None
        if opt.restore or opt.esrgan:
            from ldm.invoke.restoration import Restoration
            restoration = Restoration()
            if opt.restore:
                gfpgan, codeformer = restoration.load_face_restore_models(opt.gfpgan_model_path)
            else:
                print('>> Face restoration disabled')
            if opt.esrgan:
                esrgan = restoration.load_esrgan(opt.esrgan_bg_tile)
            else:
                print('>> Upscaling disabled')
        else:
            print('>> Face restoration and upscaling disabled')
    except (ModuleNotFoundError, ImportError):
        print(traceback.format_exc(), file=sys.stderr)
        print('>> You may need to install the ESRGAN and/or GFPGAN modules')
    return gfpgan,codeformer,esrgan

def make_step_callback(gen, opt, prefix):
    destination = os.path.join(opt.outdir,'intermediates',prefix)
    os.makedirs(destination,exist_ok=True)
    print(f'>> Intermediate images will be written into {destination}')
    def callback(img, step):
        if step % opt.save_intermediates == 0 or step == opt.steps-1:
            filename = os.path.join(destination,f'{step:04}.png')
            image = gen.sample_to_image(img)
            image.save(filename,'PNG')
    return callback

def retrieve_dream_command(opt,command,completer):
    '''
    Given a full or partial path to a previously-generated image file,
    will retrieve and format the dream command used to generate the image,
    and pop it into the readline buffer (linux, Mac), or print out a comment
    for cut-and-paste (windows)

    Given a wildcard path to a folder with image png files,
    will retrieve and format the dream command used to generate the images,
    and save them to a file commands.txt for further processing
    '''
    if len(command) == 0:
        return

    tokens = command.split()
    dir,basename = os.path.split(tokens[0])
    if len(dir) == 0:
        path = os.path.join(opt.outdir,basename)
    else:
        path = tokens[0]

    if len(tokens) > 1:
        return write_commands(opt, path, tokens[1])

    cmd = ''
    try:
        cmd = dream_cmd_from_png(path)
    except OSError:
        print(f'## {tokens[0]}: file could not be read')
    except (KeyError, AttributeError, IndexError):
        print(f'## {tokens[0]}: file has no metadata')
    except:
        print(f'## {tokens[0]}: file could not be processed')
    if len(cmd)>0:
        completer.set_line(cmd)

def write_commands(opt, file_path:str, outfilepath:str):
    dir,basename = os.path.split(file_path)
    try:
        paths = sorted(list(Path(dir).glob(basename)))
    except ValueError:
        print(f'## "{basename}": unacceptable pattern')
        return

    commands = []
    cmd = None
    for path in paths:
        try:
            cmd = dream_cmd_from_png(path)
        except (KeyError, AttributeError, IndexError):
            print(f'## {path}: file has no metadata')
        except:
            print(f'## {path}: file could not be processed')
        if cmd:
            commands.append(f'# {path}')
            commands.append(cmd)
    if len(commands)>0:
        dir,basename = os.path.split(outfilepath)
        if len(dir)==0:
            outfilepath = os.path.join(opt.outdir,basename)
        with open(outfilepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(commands))
        print(f'>> File {outfilepath} with commands created')

def emergency_model_reconfigure(opt):
    print()
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('   You appear to have a missing or misconfigured model file(s).                   ')
    print('   The script will now exit and run configure_invokeai.py to help fix the problem.')
    print('   After reconfiguration is done, please relaunch invoke.py.                      ')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('configure_invokeai is launching....\n')

    # Match arguments that were set on the CLI
    # only the arguments accepted by the configuration script are parsed
    root_dir = ["--root", opt.root_dir] if opt.root_dir is not None else []
    config = ["--config", opt.conf] if opt.conf is not None else []
    yes_to_all = os.environ.get('INVOKE_MODEL_RECONFIGURE')

    sys.argv = [ 'configure_invokeai' ]
    sys.argv.extend(root_dir)
    sys.argv.extend(config)
    if yes_to_all is not None:
        sys.argv.append(yes_to_all)

    import configure_invokeai
    configure_invokeai.main()
