from modules.parse_seed_weights import parse_seed_weights

"""
Helper functions to convert the data received from frontend to
directly-consumable dicts for the backend.
"""


def make_generation_parameters(data):
    """
    Convert image generation parameters received from frontend to
    dict with the same key names the backend uses.
    """

    parameters = {
        # Common parameters
        'prompt': data['prompt'] if 'prompt' in data else None,
        'iterations': data['iterations'] if 'iterations' in data else None,
        'steps': data['steps'] if 'steps' in data else None,
        'width': data['width'] if 'width' in data else None,
        'height': data['height'] if 'height' in data else None,
        'fit': data['shouldFitToWidthHeight'] if 'shouldFitToWidthHeight' in data else None,
        'cfg_scale': data['cfgScale'] if 'cfgScale' in data else None,
        'sampler_name': data['sampler'] if 'sampler' in data else None,
        'seed': data['seed'] if 'seed' in data else None,
        'seamless': data['seamless'] if 'seamless' in data else None,

        # Init image only
        'init_img': data['initialImagePath'] if 'initialImagePath' in data else None,
        'init_mask': data['maskPath'] if 'maskPath' in data else None,
        'strength': data['img2imgStrength'] if 'img2imgStrength' in data else None,
        'fit': data['shouldFitToWidthHeight'] if 'shouldFitToWidthHeight' in data else None,

        # System
        'progress_images': data['shouldDisplayInProgress'] if 'shouldDisplayInProgress' in data else None,

        # # ESRGAN/GFPGAN - left in such a way that generate ignores them
        # 'gfpgan_strength': 0,
        # 'upscale': None,
    }

    if 'shouldGenerateVariations' in data and data['shouldGenerateVariations'] is True:
        seed_weights = parse_seed_weights(data['seedWeights'])
        parameters['with_variations'] = seed_weights if (data['shouldGenerateVariations'] and seed_weights) else None
        parameters['variation_amount'] = data['variantAmount'] if data['shouldGenerateVariations'] else 0

    return parameters

def make_esrgan_parameters(data):
    """
    Convert ESRGAN parameters received from frontend to
    dict with the same key names the backend uses.
    """
    return {
        # TODO: pending metadata RFC, we get seed from the image file itself
        'seed': data['imagePath'].split(".")[1] if 'imagePath' in data else data['seed'],
        'upsampler_scale': data['upscalingLevel'],
        'strength': data['upscalingStrength'],
    }


def make_gfpgan_parameters(data):
    """
    Convert GFPGAN parameters received from frontend to
    dict with the same key names the backend uses.
    """
    return {
        # TODO: pending metadata RFC, we get seed from the image file itself
        'seed': data['imagePath'].split(".")[1] if 'imagePath' in data else data['seed'],
        'strength': data['gfpganStrength']
    }


def parameters_to_command(data):
    switches = list()

    if 'prompt' in data:
        switches.append(f'"{data["prompt"]}"')
    if 'steps' in data:
        switches.append(f'-s {data["steps"]}')
    if 'seed' in data:
        switches.append(f'-S {data["seed"]}')
    if 'width' in data:
        switches.append(f'-W {data["width"]}')
    if 'height' in data:
        switches.append(f'-H {data["height"]}')
    if 'cfg_scale' in data:
        switches.append(f'-C {data["cfg_scale"]}')
    if 'sampler' in data:
        switches.append(f'-A {data["sampler"]}')
    if 'seamless' in data and data["seamless"] == True:
        switches.append(f'--seamless')
    if 'init_img' in data and len(data['init_img']) > 0:
        switches.append(f'-I {data["init_img"]}')
    if 'init_mask' in data and len(data['init_mask']) > 0:
        switches.append(f'-M {data["init_mask"]}')
    if 'strength' in data and 'init_img' in data:
        switches.append(f'-f {data["strength"]}')
        if 'fit' in data and data["fit"] == True:
            switches.append(f'--fit')
    if 'gfpgan_strength' in data and data["gfpgan_strength"]:
        switches.append(f'-G {data["gfpgan_strength"]}')
    if 'upscale' in data and data["upscale"]:
        switches.append(f'-U {data["upscale"][0]} {data["upscale"][1]}')
    if 'variation_amount' in data and data['variation_amount'] > 0:
        switches.append(f'-v {data["variation_amount"]}')
        if 'with_variations' in data:
            seed_weight_pairs = ','.join(f'{seed}:{weight}' for seed, weight in data["with_variations"])
            switches.append(f'-V {seed_weight_pairs}')

    return ' '.join(switches)
