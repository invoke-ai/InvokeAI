import type { BackendGraphContract, BackendInvocationContract } from '@features/generation/core/contracts';
import type { GenerateSettings, MainModelConfig, VaeModelConfig } from '@features/generation/core/types';

/** A node output that feeds an input: the node and its output field. */
export interface GraphOutput {
  node: BackendInvocationContract;
  field: string;
}

/**
 * A family builder's graph with the facts callers compose over. Canvas grafts image encodes, masks, control and
 * regional conditioning onto the nodes the builder names here, and Upscale feeds its enlarged frame through `latents`
 * into `denoise`, so neither guesses node ids nor scans edges for these facts. The `seed` and output ids stay shared
 * contracts outside this record.
 */
export interface FamilyGraph {
  graph: BackendGraphContract;
  /** The base denoise every attachment feeds: latents, denoise mask, control, regional conditioning. */
  denoise: BackendInvocationContract;
  /** The size the denoise works at: the requested size, except PiD native denoises at a quarter of it. */
  denoiseSize: { width: number; height: number };
  /** The latent space Canvas encodes images into; null when the family has no encode node (Ideogram 4, ERNIE-Image). */
  latents: FamilyLatentSpace | null;
  /** The global conditioning regional guidance copies and joins; null where the family has no collectors. */
  conditioning: FamilyConditioning | null;
  /** The node a layout base places regions on: MOSS-Image's and FLUX.3's layout prompt, Ideogram 4's caption builder. */
  layoutPrompt: BackendInvocationContract | null;
}

export interface FamilyLatentSpace {
  /** The image-to-latents node type. */
  encodeType: string;
  /** The VAE the graph decodes with, override and seamless wrapping included; encodes and masks use the same one. */
  vae: GraphOutput;
  /** The UNet the gradient mask reads; SD families only. */
  unet?: GraphOutput;
  /** SD encode and gradient-mask nodes carry an fp32 switch; other families' nodes have none. */
  fp32?: boolean;
}

export interface FamilyConditioning {
  /** The global positive conditioning node regional conditioning copies encoder edges from. */
  positive: BackendInvocationContract;
  /** The collector feeding the denoise's positive conditioning, which regional positives join. */
  positiveCollector: BackendInvocationContract;
  /** The global negative conditioning and its collector; null where the denoise takes no negative collection. */
  negative: { node: BackendInvocationContract; collector: BackendInvocationContract } | null;
  /**
   * The collector the builder already feeds `denoise.ip_adapter` from, when it wired reference images. Only read where
   * regional references are IP-Adapters (SD families).
   */
  ipAdapterCollector: BackendInvocationContract | null;
  /**
   * The collector the builder already feeds `denoise.redux_conditioning` from, when it wired FLUX Redux. Only read where
   * regional references are FLUX Redux (FLUX).
   */
  fluxReduxCollector: BackendInvocationContract | null;
}

/**
 * The settings a family pipeline reads. Generate hands over its own settings; Upscale maps its widget fields onto
 * these, so widening this type fails Upscale's build until it decides what the new field is there.
 */
export type FamilyPipelineSettings = Pick<
  GenerateSettings,
  | 'cfgRescaleMultiplier'
  | 'cfgScale'
  | 'clipEmbedModel'
  | 'clipSkip'
  | 'componentSourceModel'
  | 'loras'
  | 'mistralEncoderModel'
  | 'qwen35EncoderModel'
  | 'qwen3EncoderModel'
  | 'qwen3VLEncoderModel'
  | 'qwenVLEncoderModel'
  | 'scheduler'
  | 'steps'
  | 't5EncoderModel'
  | 'vae'
  | 'vaePrecision'
>;

/**
 * What a family's image pipeline is asked to build: prompts and seed, model loader, LoRAs, text encoders, the denoise
 * and its decode. Nothing Generate adds on top (references, PiD, the refiner, enhancers, metadata) can be asked for
 * here; Generate passes those to its builders separately.
 */
export interface FamilyPipelineInput {
  model: MainModelConfig;
  settings: FamilyPipelineSettings;
  /** The size the denoise runs at, already on the family's grid. */
  denoiseSize: { width: number; height: number };
  /** Where the denoise enters its schedule, in the denoise node's own terms (see `getDenoisingStartClipping`). */
  denoisingStart: number;
  /** SD's noise node runs on the CPU, so a seed gives the same image on any device. */
  useCpuNoise: boolean;
  /** The decode node's id and whether its image is an intermediate. */
  output: { id: string; isIntermediate: boolean };
  /**
   * Tile the decode at this size where the family's decode node has the switch (`anima_l2i` tiles on its own); null
   * decodes whole. The caller tiles its own encode.
   */
  vaeTileSize: number | null;
  /**
   * SD only: denoise in overlapping tiles (`tiled_multi_diffusion_denoise_latents`) instead of whole. The tiled node
   * takes one conditioning per side, so the graph has no collectors and no `conditioning` to graft onto.
   */
  multiDiffusion: { tileSize: number; tileOverlap: number } | null;
}

/** A family's image pipeline: its graph, the facts callers compose over, and the node whose image is the result. */
export interface FamilyPipeline extends FamilyGraph {
  /** The decode node, `input.output.id`. */
  output: BackendInvocationContract;
  /** The VAE the graph loads over the main model's own; null where it decodes with the model's bundled VAE. */
  vaeModel: VaeModelConfig | null;
}
