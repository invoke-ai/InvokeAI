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
    print()
    return {
        # Common parameters
        'prompt': data['prompt'],
        'strength': data['img2imgStrength'],
        'iterations': data['iterations'],
        'steps': data['steps'],
        'width': data['width'],
        'height': data['height'],
        'fit': data['shouldFitToWidthHeight'],
        'cfgscale': data['cfgScale'],
        'sampler_name': data['sampler'],
        'seed': data['seed'],
        'seamless': data['seamless'],

        # Init image only
        'init_img': data['initialImagePath'],
        'init_mask': data['maskPath'],
        'fit': data['shouldFitToWidthHeight'],

        # Variations
        'variation_amount': data['variantAmount'],
        'seed_weights': parse_seed_weights(data['seedWeights']),

        # System
        'progress_images': data['shouldDisplayInProgress'],
    }


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
