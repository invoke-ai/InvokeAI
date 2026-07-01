import type { RootState } from 'app/store/store';
import { roundDownToMultiple } from 'common/util/roundDownToMultiple';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import {
  getDenoisingStartAndEnd,
  getOriginalAndScaledSizesForOtherModes,
  getOriginalAndScaledSizesForTextToImage,
} from 'features/nodes/util/graph/graphBuilderUtils';
import type { ImageToLatentsNodes, MainModelLoaderNodes, VaeSourceNodes } from 'features/nodes/util/graph/types';
import { PID_SCALE } from 'features/parameters/util/optimalDimension';
import type { Invocation } from 'services/api/types';
import { assert } from 'tsafe';

// FLUX works on a 16px grid (VAE /8 x 2x2 patches), so the generation resolution must be a multiple of 16.
const FLUX_GRID_SIZE = 16;

type Size = { width: number; height: number };

/**
 * The base-specific PiD decode node types. Each replaces its base's VAE decode with the PiD super-res decode.
 * Only bases whose graph builder actually wires PiD are listed; more are added as their builders gain support.
 */
type PidDecodeNodeType = 'flux_pid_decode' | 'flux2_pid_decode' | 'sd3_pid_decode';

/**
 * Denoise nodes whose latents PiD can decode. Narrower than `DenoiseLatentsNodes` so the shared
 * width/height/denoising_start/denoising_end fields (which only the FLUX-family denoise nodes have) are available.
 */
type PidDenoiseNodeType = 'flux_denoise' | 'flux2_denoise' | 'sd3_denoise';

/** PiD decode node types that expose a `vae` input (used to read the VAE's scaling constants at runtime). */
const PID_DECODE_NODES_WITH_VAE_INPUT = new Set<PidDecodeNodeType>(['flux2_pid_decode']);

type BuildPidDecodeChainArg = {
  g: Graph;
  state: RootState;
  /** The denoise node producing the latents PiD will decode. Its dimensions are set by the CALLER. */
  denoise: Invocation<PidDenoiseNodeType>;
  /** Which base-specific PiD decode node to build (e.g. `flux_pid_decode`, `flux2_pid_decode`). */
  decodeNodeType: PidDecodeNodeType;
  /**
   * Optional VAE source. If the chosen decode node has a `vae` input (e.g. `flux2_pid_decode`), it is wired so
   * the node can read the VAE's scaling/shift constants at runtime. Ignored for nodes without a `vae` input.
   */
  vaeSource?: Invocation<VaeSourceNodes | MainModelLoaderNodes>;
  /** The positive prompt node - PiD conditions its decode on the same caption. */
  positivePrompt: Invocation<'string'>;
  /** The seed node - reused for PiD's internal decode noise so results are reproducible. */
  seed: Invocation<'integer'>;
  /**
   * - 'fit':    PiD decodes 4x, then the output is downscaled to `fitSize` (compositing-safe; used everywhere).
   * - 'native': PiD's full 4x output is used directly (txt2img only; `fitSize` is ignored).
   */
  mode: 'fit' | 'native';
  /** The size to downscale the 4x output to in 'fit' mode (the bbox / region the result must fit). */
  fitSize: Size;
};

/**
 * Builds the PiD (Pixel Diffusion Decoder) decode chain: the Gemma-2 + PiD loaders, the `flux_pid_decode` node
 * wired to the given denoise latents, and (in 'fit' mode) an `img_resize` that downscales PiD's 4x output to
 * `fitSize`. Returns the terminal image node, which is a drop-in for the regular VAE decode (`l2i`) - downstream
 * nodes only consume its `.image` output.
 *
 * This does NOT modify the denoise node's dimensions or denoising start/end; the caller owns those (they differ
 * between txt2img and img2img/inpaint).
 */
export const buildPidDecodeChain = ({
  g,
  state,
  denoise,
  decodeNodeType,
  vaeSource,
  positivePrompt,
  seed,
  mode,
  fitSize,
}: BuildPidDecodeChainArg): Invocation<'img_resize' | PidDecodeNodeType> => {
  const params = selectParamsSlice(state);
  const { pidDecoderModel, gemma2EncoderModel, pidSteps } = params;
  assert(pidDecoderModel, 'No PiD decoder model selected');
  assert(gemma2EncoderModel, 'No Gemma-2 encoder model selected');

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
    type: decodeNodeType,
    id: getPrefixedId(decodeNodeType),
    num_inference_steps: pidSteps,
  });

  g.addEdge(denoise, 'latents', pidDecode, 'latents');
  g.addEdge(positivePrompt, 'value', pidDecode, 'prompt');
  g.addEdge(gemma2Loader, 'gemma2_encoder', pidDecode, 'gemma2_encoder');
  g.addEdge(pidLoader, 'pid_decoder', pidDecode, 'pid_decoder');
  g.addEdge(seed, 'value', pidDecode, 'seed');
  // Wire the VAE only for decode nodes that read scaling constants from it (currently just flux2_pid_decode).
  if (vaeSource && PID_DECODE_NODES_WITH_VAE_INPUT.has(decodeNodeType)) {
    g.addEdge(vaeSource, 'vae', pidDecode as Invocation<'flux2_pid_decode'>, 'vae');
  }

  const commonMetadata = {
    pid_decoder: pidDecoderModel,
    gemma2_encoder: gemma2EncoderModel,
    pid_steps: pidSteps,
  };

  if (mode === 'native') {
    // PiD's 4x output IS the result (the caller generated at target / 4) - no downscale.
    assert(
      denoise.width !== undefined && denoise.height !== undefined,
      'PiD native decode requires the denoise dimensions to be set by the caller'
    );
    g.upsertMetadata({
      ...commonMetadata,
      pid_mode: mode,
      width: denoise.width * PID_SCALE,
      height: denoise.height * PID_SCALE,
    });
    return pidDecode;
  }

  // Fit mode: downscale PiD's 4x output back to the requested size.
  const resize = g.addNode({
    id: getPrefixedId('pid_fit_resize'),
    type: 'img_resize',
    ...fitSize,
  });
  g.addEdge(pidDecode, 'image', resize, 'image');
  g.upsertMetadata({ ...commonMetadata, pid_mode: mode, width: fitSize.width, height: fitSize.height });

  return resize;
};

type AddPidDecodeArg = {
  g: Graph;
  state: RootState;
  mode: 'fit' | 'native';
  denoise: Invocation<PidDenoiseNodeType>;
  decodeNodeType: PidDecodeNodeType;
  vaeSource?: Invocation<VaeSourceNodes | MainModelLoaderNodes>;
  positivePrompt: Invocation<'string'>;
  seed: Invocation<'integer'>;
};

/**
 * Text-to-image PiD decode: sets up the denoise node (full denoise, generation dimensions) and replaces the VAE
 * decode with a PiD decode (see {@link buildPidDecodeChain}).
 *
 * - 'fit':    generate at the requested size, PiD decodes 4x, then downscale back to it.
 * - 'native': the requested dimensions are the 4x target; generate at target / 4 and use PiD's 4x output directly.
 *
 * The caller is responsible for having NOT wired a VAE decode for these latents (or for deleting it).
 *
 * @returns The terminal image node, to be used as the canvas output.
 */
export const addPidDecode = ({
  g,
  state,
  mode,
  denoise,
  decodeNodeType,
  vaeSource,
  positivePrompt,
  seed,
}: AddPidDecodeArg): Invocation<'img_resize' | PidDecodeNodeType> => {
  const { originalSize, scaledSize } = getOriginalAndScaledSizesForTextToImage(state);

  denoise.denoising_start = 0;
  denoise.denoising_end = 1;
  if (mode === 'native') {
    // The user-facing dimensions are the 4x target; generate at target / PID_SCALE (kept on the FLUX grid).
    denoise.width = Math.max(roundDownToMultiple(originalSize.width / PID_SCALE, FLUX_GRID_SIZE), FLUX_GRID_SIZE);
    denoise.height = Math.max(roundDownToMultiple(originalSize.height / PID_SCALE, FLUX_GRID_SIZE), FLUX_GRID_SIZE);
  } else {
    // Generate at the normal resolution; PiD will 4x it and we downscale back to it.
    denoise.width = scaledSize.width;
    denoise.height = scaledSize.height;
  }

  return buildPidDecodeChain({
    g,
    state,
    denoise,
    decodeNodeType,
    vaeSource,
    positivePrompt,
    seed,
    mode,
    fitSize: originalSize,
  });
};

type AddPidImageToImageNativeArg = {
  g: Graph;
  state: RootState;
  manager: CanvasManager;
  /** The denoise node. Its dimensions are set here to the 4x target / PID_SCALE. */
  denoise: Invocation<PidDenoiseNodeType>;
  /** Which base-specific PiD decode node to build. */
  decodeNodeType: PidDecodeNodeType;
  /** The VAE encode node for the init image. */
  i2l: Invocation<ImageToLatentsNodes>;
  /** The model loader / VAE source providing the VAE for encoding the init image (and, if applicable, the decode). */
  vaeSource: Invocation<VaeSourceNodes | MainModelLoaderNodes>;
  positivePrompt: Invocation<'string'>;
  seed: Invocation<'integer'>;
};

/**
 * Native-4x PiD image-to-image (Canvas only). The user-facing bbox IS the 4x target: generation runs at bbox /
 * PID_SCALE, the init image is downscaled to that resolution before encoding, and PiD decodes the latents straight
 * back up to the full bbox size - no post-decode downscale, so all of PiD's detail is preserved. Because the result
 * is exactly the bbox size it composites cleanly back onto the canvas region.
 *
 * Requires the bbox to be a multiple of the PiD-scaled grid (enforced by the UI grid snapping / readiness) so that
 * bbox / PID_SCALE lands on the FLUX grid and PiD's 4x output matches the bbox exactly.
 *
 * @returns The terminal `flux_pid_decode` node, to be used as the canvas output.
 */
export const addPidImageToImageNative = async ({
  g,
  state,
  manager,
  denoise,
  decodeNodeType,
  i2l,
  vaeSource,
  positivePrompt,
  seed,
}: AddPidImageToImageNativeArg): Promise<Invocation<'img_resize' | PidDecodeNodeType>> => {
  const { denoising_start, denoising_end } = getDenoisingStartAndEnd(state);
  denoise.denoising_start = denoising_start;
  denoise.denoising_end = denoising_end;

  const { originalSize, rect } = getOriginalAndScaledSizesForOtherModes(state);

  // The bbox is the 4x target; generate at target / PID_SCALE (kept on the FLUX grid).
  const genSize = {
    width: Math.max(roundDownToMultiple(originalSize.width / PID_SCALE, FLUX_GRID_SIZE), FLUX_GRID_SIZE),
    height: Math.max(roundDownToMultiple(originalSize.height / PID_SCALE, FLUX_GRID_SIZE), FLUX_GRID_SIZE),
  };
  denoise.width = genSize.width;
  denoise.height = genSize.height;

  const adapters = manager.compositor.getVisibleAdaptersOfType('raster_layer');
  const { image_name } = await manager.compositor.getCompositeImageDTO(adapters, rect, {
    is_intermediate: true,
    silent: true,
  });

  // Downscale the init image to the generation resolution before encoding.
  const resizeIn = g.addNode({
    type: 'img_resize',
    id: getPrefixedId('initial_image_resize_in'),
    image: { image_name },
    ...genSize,
  });
  g.addEdge(vaeSource, 'vae', i2l, 'vae');
  g.addEdge(resizeIn, 'image', i2l, 'image');
  g.addEdge(i2l, 'latents', denoise, 'latents');

  // PiD decodes the genSize latents straight up to 4x = the bbox. fitSize is ignored in native mode.
  return buildPidDecodeChain({
    g,
    state,
    denoise,
    decodeNodeType,
    vaeSource,
    positivePrompt,
    seed,
    mode: 'native',
    fitSize: originalSize,
  });
};
