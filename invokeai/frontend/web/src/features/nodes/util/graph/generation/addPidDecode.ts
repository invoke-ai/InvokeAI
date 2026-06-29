import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { getOriginalAndScaledSizesForTextToImage } from 'features/nodes/util/graph/graphBuilderUtils';
import type { Invocation } from 'services/api/types';
import { assert } from 'tsafe';

type AddPidDecodeArg = {
  g: Graph;
  state: RootState;
  /** The FLUX denoise node producing the latents PiD will decode. */
  denoise: Invocation<'flux_denoise'>;
  /** The positive prompt node - PiD conditions its decode on the same caption. */
  positivePrompt: Invocation<'string'>;
  /** The seed node - reused for PiD's internal decode noise so results are reproducible. */
  seed: Invocation<'integer'>;
};

/**
 * Adds a PiD (Pixel Diffusion Decoder) decode in place of the regular FLUX VAE decode, in "fit" mode.
 *
 * PiD is a fixed 4x super-resolution decoder: it consumes the FLUX latent and emits an image at 4x the
 * generation resolution. In "fit" mode we downscale that output back to the requested size so the result
 * composites cleanly (canvas/inpaint) and matches the dimensions the user expects. The 4x detail gain is
 * partly traded away by the downscale - "native" mode (no downscale) is handled separately.
 *
 * The caller is responsible for having NOT wired a VAE decode for these latents (or for deleting it).
 *
 * @returns The terminal image node (the downscale), to be used as the canvas output.
 */
export const addPidDecode = ({
  g,
  state,
  denoise,
  positivePrompt,
  seed,
}: AddPidDecodeArg): Invocation<'img_resize'> => {
  const params = selectParamsSlice(state);
  const { pidDecoderModel, gemma2EncoderModel, pidSteps } = params;
  assert(pidDecoderModel, 'No PiD decoder model selected');
  assert(gemma2EncoderModel, 'No Gemma-2 encoder model selected');

  const { originalSize, scaledSize } = getOriginalAndScaledSizesForTextToImage(state);

  // Generate at the normal resolution; PiD will 4x it and we downscale back below.
  denoise.denoising_start = 0;
  denoise.denoising_end = 1;
  denoise.width = scaledSize.width;
  denoise.height = scaledSize.height;

  const gemma2Loader = g.addNode({
    type: 'gemma2_encoder_loader',
    id: getPrefixedId('gemma2_encoder_loader'),
    gemma2_model: gemma2EncoderModel,
  });
  const pidLoader = g.addNode({
    type: 'pid_decoder_loader',
    id: getPrefixedId('pid_decoder_loader'),
    pid_decoder_model: pidDecoderModel,
  });
  const pidDecode = g.addNode({
    type: 'flux_pid_decode',
    id: getPrefixedId('flux_pid_decode'),
    num_inference_steps: pidSteps,
  });

  g.addEdge(denoise, 'latents', pidDecode, 'latents');
  g.addEdge(positivePrompt, 'value', pidDecode, 'prompt');
  g.addEdge(gemma2Loader, 'gemma2_encoder', pidDecode, 'gemma2_encoder');
  g.addEdge(pidLoader, 'pid_decoder', pidDecode, 'pid_decoder');
  g.addEdge(seed, 'value', pidDecode, 'seed');

  // Fit mode: downscale PiD's 4x output back to the requested size.
  const resize = g.addNode({
    id: getPrefixedId('pid_fit_resize'),
    type: 'img_resize',
    ...originalSize,
  });
  g.addEdge(pidDecode, 'image', resize, 'image');

  g.upsertMetadata({
    width: originalSize.width,
    height: originalSize.height,
    pid_mode: 'fit',
    pid_decoder: pidDecoderModel,
    gemma2_encoder: gemma2EncoderModel,
    pid_steps: pidSteps,
  });

  return resize;
};
