import type { BackendGraphContract, BackendInvocationContract } from '@features/generation/core/contracts';

import type { GenerateSettings, PidMode } from './types';

import { addEdge, addNode, createId } from './graphBuilder';
import { getIsPidActive, getPidGenerationSize } from './pid';

/** pid.ts owns geometry; this wiring replaces VAE decode with caption-conditioned 4× output. */

/** The per-base PiD decode node types. One per base whose builder wires PiD. */
export const PID_DECODE_NODE_TYPES = {
  flux: 'flux_pid_decode',
  flux2: 'flux2_pid_decode',
  'qwen-image': 'qwen_image_pid_decode',
  'sd-3': 'sd3_pid_decode',
  sdxl: 'sdxl_pid_decode',
  'z-image': 'z_image_pid_decode',
} as const;

export type PidDecodeNodeType = (typeof PID_DECODE_NODE_TYPES)[keyof typeof PID_DECODE_NODE_TYPES];

/** Retain VAE edges for runtime scaling constants even though PiD replaces decode. */
const PID_DECODE_NODES_WITH_VAE_INPUT: ReadonlySet<string> = new Set([
  'flux2_pid_decode',
  'qwen_image_pid_decode',
  'sdxl_pid_decode',
  'z_image_pid_decode',
]);

export const getPidDecodeNodeType = (base: string | null | undefined): PidDecodeNodeType | null =>
  base && base in PID_DECODE_NODE_TYPES ? PID_DECODE_NODE_TYPES[base as keyof typeof PID_DECODE_NODE_TYPES] : null;

/** Readiness must reject incomplete PiD first; false is safe only after that validation. */
export const shouldUsePidDecode = (
  settings: Pick<GenerateSettings, 'pidMode' | 'pidDecoderModel' | 'gemma2EncoderModel'>,
  base: string | null | undefined
): boolean =>
  getIsPidActive(settings.pidMode, base) &&
  settings.pidDecoderModel !== null &&
  settings.gemma2EncoderModel !== null &&
  getPidDecodeNodeType(base) !== null;

/** The resolution to denoise at, which in native mode is the requested size / 4. */
export const getPidDenoiseSize = (
  settings: Pick<GenerateSettings, 'pidMode' | 'width' | 'height'>,
  base: string | null | undefined,
  modelGrid: number
): { width: number; height: number } =>
  getPidGenerationSize(
    { height: settings.height, width: settings.width },
    getIsPidActive(settings.pidMode, base) ? settings.pidMode : 'off',
    modelGrid
  );

interface AddPidDecodeArg {
  graph: BackendGraphContract;
  settings: GenerateSettings;
  base: string;
  /** The denoise node producing the latents to decode. */
  denoise: BackendInvocationContract;
  /** The positive prompt node — PiD conditions its decode on the same caption. */
  positivePrompt: BackendInvocationContract;
  /** The seed node, reused for PiD's internal decode noise so results reproduce. */
  seed: BackendInvocationContract;
  /** VAE source, wired only for decode nodes that read its constants. */
  vaeSource?: BackendInvocationContract;
  /** The VAE output field on `vaeSource` (model loaders expose `vae`). */
  vaeField?: string;
  outputIsIntermediate: boolean;
}

/** Result hydration requires canvas_output on the terminal node: resize for fit, decode for native. */
export const addPidDecode = ({
  graph,
  settings,
  base,
  denoise,
  positivePrompt,
  seed,
  vaeSource,
  vaeField = 'vae',
  outputIsIntermediate,
}: AddPidDecodeArg): BackendInvocationContract => {
  const decodeNodeType = getPidDecodeNodeType(base);

  if (!decodeNodeType || !settings.pidDecoderModel || !settings.gemma2EncoderModel) {
    throw new Error('PiD decoding needs a PiD decoder and a Gemma-2 encoder.');
  }

  const isNative = settings.pidMode === 'native';
  const gemma2Loader = addNode(graph, {
    gemma2_model: settings.gemma2EncoderModel,
    id: createId('gemma2_encoder_loader'),
    type: 'gemma2_encoder_loader',
  });
  const pidLoader = addNode(graph, {
    id: createId('pid_decoder_loader'),
    pid_decoder_model: settings.pidDecoderModel,
    type: 'pid_decoder_loader',
  });
  const pidDecode = addNode(graph, {
    // In fit mode, move output ownership from decode to resize and mark decode intermediate.
    id: isNative ? 'canvas_output' : createId('pid_decode'),
    is_intermediate: isNative ? outputIsIntermediate : true,
    num_inference_steps: settings.pidSteps,
    type: decodeNodeType,
    use_cache: false,
  });

  addEdge(graph, denoise, 'latents', pidDecode, 'latents');
  addEdge(graph, positivePrompt, 'value', pidDecode, 'prompt');
  addEdge(graph, gemma2Loader, 'gemma2_encoder', pidDecode, 'gemma2_encoder');
  addEdge(graph, pidLoader, 'pid_decoder', pidDecode, 'pid_decoder');
  addEdge(graph, seed, 'value', pidDecode, 'seed');

  if (vaeSource && PID_DECODE_NODES_WITH_VAE_INPUT.has(decodeNodeType)) {
    addEdge(graph, vaeSource, vaeField, pidDecode, 'vae');
  }

  if (isNative) {
    return pidDecode;
  }

  // Fit returns the exact requested dimensions after 4× decode.
  const resize = addNode(graph, {
    height: settings.height,
    id: 'canvas_output',
    is_intermediate: outputIsIntermediate,
    type: 'img_resize',
    use_cache: false,
    width: settings.width,
  });
  addEdge(graph, pidDecode, 'image', resize, 'image');

  return resize;
};

/** The PiD fields to record in image metadata. */
export const getPidMetadata = (settings: GenerateSettings): Record<string, unknown> & { pid_mode: PidMode } => ({
  gemma2_encoder: settings.gemma2EncoderModel,
  pid_decoder: settings.pidDecoderModel,
  pid_mode: settings.pidMode,
  pid_steps: settings.pidSteps,
});
