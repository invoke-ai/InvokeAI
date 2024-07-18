import { Graph, GraphType } from 'features/nodes/util/graph/generation/Graph';
import { RootState } from '../../../../app/store/store';
import { assert } from 'tsafe';
import { ControlNetModelConfig, Invocation, NonNullableGraph } from '../../../../services/api/types';
import { ESRGAN, NEGATIVE_CONDITIONING, POSITIVE_CONDITIONING } from './constants';
import { isParamESRGANModelName } from '../../../parameters/store/postprocessingSlice';
import { ControlNetConfig } from '../../../controlAdapters/store/types';
import { MODEL_TYPES } from '../../types/constants';


export const buildMultidiffusionUpscsaleGraph = async (state: RootState): Promise<GraphType> => {
    const {
        model,
        cfgScale: cfg_scale,
        scheduler,
        steps,
        vaePrecision,
        seed,
        vae,
    } = state.generation;
    const { positivePrompt, negativePrompt } = state.controlLayers.present;
    const { upscaleModel, upscaleInitialImage, sharpness, structure, creativity } = state.upscale;

    assert(model, 'No model found in state');
    assert(upscaleModel, 'No upscale model found in state');
    assert(upscaleInitialImage, 'No initial image found in state');

    if (!isParamESRGANModelName(upscaleModel.name)) {
        throw new Error()
    }

    const g = new Graph()

    const unsharp_mask_1 = g.addNode({
        id: 'unsharp_mask_1',
        type: 'unsharp_mask',
        image: upscaleInitialImage,
        radius: 2,
        strength: ((sharpness + 10) * 3.75) + 25
    })

    const esrgan = g.addNode({
        id: ESRGAN,
        type: 'esrgan',
        model_name: upscaleModel.name,
        tile_size: 500
    })

    g.addEdge(unsharp_mask_1, 'image', esrgan, 'image')

    const unsharp_mask_2 = g.addNode({
        id: 'unsharp_mask_2',
        type: 'unsharp_mask',
        radius: 2,
        strength: 50
    })

    g.addEdge(esrgan, 'image', unsharp_mask_2, 'image',)

    const SCALE = 2

    const resizeNode = g.addNode({
        id: 'img_resize',
        type: 'img_resize',
        width: upscaleInitialImage.width * SCALE, //  TODO: handle floats
        height: upscaleInitialImage.height * SCALE, //  TODO: handle floats
        resample_mode: "lanczos"
    })

    g.addEdge(unsharp_mask_2, 'image', resizeNode, "image")



    const sharpnessNode: Invocation<'unsharp_mask'> = { //before and after esrgan
        id: 'unsharp_mask',
        type: 'unsharp_mask',
        image: upscaleInitialImage,
        radius: 2,
        strength: ((sharpness + 10) * 3.75) + 25
    };

    const creativityNode: Invocation<'tiled_multi_diffusion_denoise_latents'> = { //before and after esrgan
        id: 'tiled_multi_diffusion_denoise_latents',
        type: 'tiled_multi_diffusion_denoise_latents',
        tile_height: 1024,
        tile_width: 1024,
        tile_overlap: 128,
        steps,
        cfg_scale,
        scheduler,
        denoising_start: (((creativity * -1) + 10) * 4.99) / 100,
        denoising_end: 1
    };

    const controlnetModel = {
        key: "placeholder",
        hash: "placeholder",
        type: "controlnet" as any,
        name: "tile",
        base: model.base
    }

    const controlnet: Invocation<"controlnet"> = {
        id: "controlnet",
        type: "controlnet",
        control_model: controlnetModel,
        control_mode: "balanced",
        resize_mode: "just_resize",
        control_weight: ((((structure + 10) * 0.025) + 0.3) * 0.013) + 0.35
    }


    const noiseNode: Invocation<'noise'> = {
        id: "noise",
        type: "noise",
        seed,
        // width: resized output width
        // height: resized output height
    }

    const posPrompt: Invocation<"compel"> = {
        type: 'compel',
        id: POSITIVE_CONDITIONING,
        prompt: positivePrompt,
    }

    const negPrompt: Invocation<"compel"> = {
        type: 'compel',
        id: NEGATIVE_CONDITIONING,
        prompt: negativePrompt,
    }

    return g.getGraph();

}