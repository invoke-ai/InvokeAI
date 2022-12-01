import os
import sys
import traceback
from ....generate import Generate


# Temporary initializer for Generate until we migrate off of it
def get_generate(args, config) -> Generate:
    # TODO: Remove the need for globals
    from ldm.invoke.globals import Globals

    # alert - setting globals here
    Globals.root = os.path.expanduser(args.root_dir or os.environ.get('INVOKEAI_ROOT') or os.path.abspath('.'))
    Globals.try_patchmatch = args.patchmatch
    
    print(f'>> InvokeAI runtime directory is "{Globals.root}"')

    # these two lines prevent a horrible warning message from appearing
    # when the frozen CLIP tokenizer is imported
    import transformers
    transformers.logging.set_verbosity_error()

    # Loading Face Restoration and ESRGAN Modules
    gfpgan, codeformer, esrgan = None, None, None
    try:
        if config.restore or config.esrgan:
            from ldm.invoke.restoration import Restoration
            restoration = Restoration()
            if config.restore:
                gfpgan, codeformer = restoration.load_face_restore_models(config.gfpgan_model_path)
            else:
                print('>> Face restoration disabled')
            if config.esrgan:
                esrgan = restoration.load_esrgan(config.esrgan_bg_tile)
            else:
                print('>> Upscaling disabled')
        else:
            print('>> Face restoration and upscaling disabled')
    except (ModuleNotFoundError, ImportError):
        print(traceback.format_exc(), file=sys.stderr)
        print('>> You may need to install the ESRGAN and/or GFPGAN modules')

    # normalize the config directory relative to root
    if not os.path.isabs(config.conf):
        config.conf = os.path.normpath(os.path.join(Globals.root,config.conf))

    if config.embeddings:
        if not os.path.isabs(config.embedding_path):
            embedding_path = os.path.normpath(os.path.join(Globals.root,config.embedding_path))
    else:
        embedding_path = None


    # TODO: lazy-initialize this by wrapping it
    try:
        generate = Generate(
            conf              = config.conf,
            model             = config.model,
            sampler_name      = config.sampler_name,
            embedding_path    = embedding_path,
            full_precision    = config.full_precision,
            precision         = config.precision,
            gfpgan            = gfpgan,
            codeformer        = codeformer,
            esrgan            = esrgan,
            free_gpu_mem      = config.free_gpu_mem,
            safety_checker    = config.safety_checker,
            max_loaded_models = config.max_loaded_models,
        )
    except (FileNotFoundError, TypeError, AssertionError):
        #emergency_model_reconfigure() # TODO?
        sys.exit(-1)
    except (IOError, KeyError) as e:
        print(f'{e}. Aborting.')
        sys.exit(-1)

    generate.free_gpu_mem = config.free_gpu_mem

    return generate
