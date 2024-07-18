import type { RootState } from 'app/store/store';
import type { GraphType } from 'features/nodes/util/graph/generation/Graph';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { isParamESRGANModelName } from 'features/parameters/store/postprocessingSlice';
import { assert } from 'tsafe';

import { CLIP_SKIP, CONTROL_NET_COLLECT, ESRGAN, IMAGE_TO_LATENTS, LATENTS_TO_IMAGE, MAIN_MODEL_LOADER, NEGATIVE_CONDITIONING, NOISE, POSITIVE_CONDITIONING, RESIZE, SDXL_MODEL_LOADER, TILED_MULTI_DIFFUSION_DENOISE_LATENTS, UNSHARP_MASK, VAE_LOADER } from './constants';
import { addLoRAs } from './generation/addLoRAs';
import { addSDXLLoRas } from './generation/addSDXLLoRAs';
import { getBoardField, getSDXLStylePrompts } from './graphBuilderUtils';


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
    const { upscaleModel, upscaleInitialImage, sharpness, structure, creativity, tiledVAE } = state.upscale;

    assert(model, 'No model found in state');
    assert(upscaleModel, 'No upscale model found in state');
    assert(upscaleInitialImage, 'No initial image found in state');
    assert(isParamESRGANModelName(upscaleModel.name), "")

    const g = new Graph()

    const unsharpMaskNode1 = g.addNode({
        id: `${UNSHARP_MASK}_1`,
        type: 'unsharp_mask',
        image: upscaleInitialImage,
        radius: 2,
        strength: ((sharpness + 10) * 3.75) + 25
    })

    const upscaleNode = g.addNode({
        id: ESRGAN,
        type: 'esrgan',
        model_name: upscaleModel.name,
        tile_size: 500
    })

    g.addEdge(unsharpMaskNode1, 'image', upscaleNode, 'image')

    const unsharpMaskNode2 = g.addNode({
        id: `${UNSHARP_MASK}_2`,
        type: 'unsharp_mask',
        radius: 2,
        strength: 50
    })

    g.addEdge(upscaleNode, 'image', unsharpMaskNode2, 'image',)

    const SCALE = 4

    const resizeNode = g.addNode({
        id: RESIZE,
        type: 'img_resize',
        width: ((upscaleInitialImage.width * SCALE) / 8) * 8,
        height: ((upscaleInitialImage.height * SCALE) / 8) * 8,
        resample_mode: "lanczos",
    })

    g.addEdge(unsharpMaskNode2, 'image', resizeNode, "image")

    const noiseNode = g.addNode({
        id: NOISE,
        type: "noise",
        seed,
    })

    g.addEdge(resizeNode, 'width', noiseNode, "width")
    g.addEdge(resizeNode, 'height', noiseNode, "height")

    const i2lNode = g.addNode({
        id: IMAGE_TO_LATENTS,
        type: "i2l",
        fp32: vaePrecision === "fp32",
        tiled: tiledVAE
    })

    g.addEdge(resizeNode, 'image', i2lNode, "image")

    const l2iNode = g.addNode({
        type: "l2i",
        id: LATENTS_TO_IMAGE,
        fp32: vaePrecision === "fp32",
        tiled: tiledVAE,
        board: getBoardField(state),
        is_intermediate: false,

    })

    const tiledMultidiffusionNode = g.addNode({
        id: TILED_MULTI_DIFFUSION_DENOISE_LATENTS,
        type: 'tiled_multi_diffusion_denoise_latents',
        tile_height: 1024, // is this dependent on base model
        tile_width: 1024, // is this dependent on base model
        tile_overlap: 128,
        steps,
        cfg_scale,
        scheduler,
        denoising_start: (((creativity * -1) + 10) * 4.99) / 100,
        denoising_end: 1
    });

    let posCondNode; let negCondNode; let modelNode;

    if (model.base === "sdxl") {
        const { positiveStylePrompt, negativeStylePrompt } = getSDXLStylePrompts(state);

        posCondNode = g.addNode({
            type: 'sdxl_compel_prompt',
            id: POSITIVE_CONDITIONING,
            prompt: positivePrompt,
            style: positiveStylePrompt
        });
        negCondNode = g.addNode({
            type: 'sdxl_compel_prompt',
            id: NEGATIVE_CONDITIONING,
            prompt: negativePrompt,
            style: negativeStylePrompt
        });
        modelNode = g.addNode({
            type: 'sdxl_model_loader',
            id: SDXL_MODEL_LOADER,
            model,
        });
        g.addEdge(modelNode, 'clip', posCondNode, 'clip');
        g.addEdge(modelNode, 'clip', negCondNode, 'clip');
        g.addEdge(modelNode, 'clip2', posCondNode, 'clip2');
        g.addEdge(modelNode, 'clip2', negCondNode, 'clip2');
        addSDXLLoRas(state, g, tiledMultidiffusionNode, modelNode, null, posCondNode, negCondNode);
    } else {
        posCondNode = g.addNode({
            type: 'compel',
            id: POSITIVE_CONDITIONING,
            prompt: positivePrompt,
        });
        negCondNode = g.addNode({
            type: 'compel',
            id: NEGATIVE_CONDITIONING,
            prompt: negativePrompt,
        });
        modelNode = g.addNode({
            type: 'main_model_loader',
            id: MAIN_MODEL_LOADER,
            model,
        });
        const clipSkipNode = g.addNode({
            type: 'clip_skip',
            id: CLIP_SKIP,
        });

        g.addEdge(modelNode, 'clip', clipSkipNode, 'clip');
        g.addEdge(clipSkipNode, 'clip', posCondNode, 'clip');
        g.addEdge(clipSkipNode, 'clip', negCondNode, 'clip');
        addLoRAs(state, g, tiledMultidiffusionNode, modelNode, null, clipSkipNode, posCondNode, negCondNode);
    }


    let vaeNode;
    if (vae) {
        vaeNode = g.addNode({
            id: VAE_LOADER,
            type: "vae_loader",
            vae_model: vae
        })
    }

    g.addEdge(vaeNode || modelNode, "vae", i2lNode, "vae")
    g.addEdge(vaeNode || modelNode, "vae", l2iNode, "vae")


    g.addEdge(noiseNode, "noise", tiledMultidiffusionNode, "noise")
    g.addEdge(i2lNode, "latents", tiledMultidiffusionNode, "latents")
    g.addEdge(posCondNode, 'conditioning', tiledMultidiffusionNode, 'positive_conditioning');
    g.addEdge(negCondNode, 'conditioning', tiledMultidiffusionNode, 'negative_conditioning');
    g.addEdge(modelNode, "unet", tiledMultidiffusionNode, "unet")
    g.addEdge(tiledMultidiffusionNode, "latents", l2iNode, "latents")


    const controlnetTileModel = { // TODO: figure out how to handle this, can't assume name is `tile` or that they have it installed
        key: "8fe0b31e-89bd-4db0-b305-df36732a9939", //mary's key
        "hash": "random:72b7863348e3b038c17d1a8c49fe6ebc91053cb5e1da69d122552e59b2495f2a", //mary's hash
        type: "controlnet" as any,
        name: "tile",
        base: model.base
    }

    const controlnetNode1 = g.addNode({
        id: 'controlnet_1',
        type: "controlnet",
        control_model: controlnetTileModel,
        control_mode: "balanced",
        resize_mode: "just_resize",
        control_weight: ((((structure + 10) * 0.025) + 0.3) * 0.013) + 0.35,
        begin_step_percent: 0,
        end_step_percent: ((structure + 10) * 0.025) + 0.3
    })

    g.addEdge(resizeNode, "image", controlnetNode1, "image")

    const controlnetNode2 = g.addNode({
        id: "controlnet_2",
        type: "controlnet",
        control_model: controlnetTileModel,
        control_mode: "balanced",
        resize_mode: "just_resize",
        control_weight: (((structure + 10) * 0.025) + 0.3) * 0.013,
        begin_step_percent: ((structure + 10) * 0.025) + 0.3,
        end_step_percent: 0.8
    })

    g.addEdge(resizeNode, "image", controlnetNode2, "image")

    const collectNode = g.addNode({
        id: CONTROL_NET_COLLECT,
        type: "collect",
    })
    g.addEdge(controlnetNode1, "control", collectNode, "item")
    g.addEdge(controlnetNode2, "control", collectNode, "item")

    g.addEdge(collectNode, "collection", tiledMultidiffusionNode, "control")


    return g.getGraph();

}