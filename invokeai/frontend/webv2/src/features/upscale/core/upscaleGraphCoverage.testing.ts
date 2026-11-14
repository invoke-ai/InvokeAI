/** Upscale coverage fixtures, shared by the graph coverage and the image recall contract. */

import type { ModelConfig } from '@features/models';

import type { UpscaleWidgetValues } from './types';

import { createDefaultUpscaleWidgetValues } from './settings';

const model = (key: string, type: string, base: string, extra: Record<string, unknown> = {}): ModelConfig =>
  ({
    base,
    file_size: 1,
    format: 'checkpoint',
    hash: `${key}-hash`,
    key,
    name: key,
    path: key,
    source: key,
    source_type: 'path',
    type,
    ...extra,
  }) as ModelConfig;

/** What the widget picks components from, the way it does for a user. */
export const INSTALLED = [
  model('spandrel', 'spandrel_image_to_image', 'any'),
  model('t5', 't5_encoder', 'any'),
  model('clip', 'clip_embed', 'any', { variant: 'large' }),
  model('qwen3-4b', 'qwen3_encoder', 'any', { variant: 'qwen3_4b' }),
  model('qwen3-8b', 'qwen3_encoder', 'any', { variant: 'qwen3_8b' }),
  model('qwen3-06b', 'qwen3_encoder', 'any', { variant: 'qwen3_06b' }),
  model('qwen35', 'qwen3_5_encoder', 'any', { variant: 'qwen3_5_4b' }),
  model('qwen3vl-4b', 'qwen3_vl_encoder', 'any', { variant: 'qwen3_vl_4b' }),
  model('qwen-vl', 'qwen_vl_encoder', 'any'),
  model('mistral', 'mistral_encoder', 'any', { variant: 'mistral3_24b' }),
  model('flux-vae', 'vae', 'flux'),
  model('flux2-vae', 'vae', 'flux2'),
  model('qwen-vae', 'vae', 'qwen-image'),
  model('sd1-tile', 'controlnet', 'sd-1', { name: 'sd1 tile' }),
  model('sdxl-tile', 'controlnet', 'sdxl', { name: 'sdxl tile' }),
  model('z-tile', 'controlnet', 'z-image', { name: 'Z-Image ControlNet Tile' }),
];

/**
 * One case per path a builder can take: CFG above 1 where the negative branch is CFG-gated, the
 * optional Z-Image ControlNet, a bundled Z-Image, Krea-2, FLUX.2 or Qwen Image install, Anima-3.8B's
 * second encoder, and FLUX.2 Klein at either encoder size and [dev] with its own loaders.
 */
export const CASES: { label: string; main: ModelConfig; patch?: Partial<UpscaleWidgetValues> }[] = [
  { label: 'sd-1', main: model('sd1', 'main', 'sd-1') },
  { label: 'sdxl', main: model('sdxl', 'main', 'sdxl') },
  { label: 'flux', main: model('flux', 'main', 'flux') },
  {
    label: 'z-image',
    main: model('z', 'main', 'z-image'),
    patch: {
      cfgScale: 4,
      tileControlnetModel: INSTALLED.find(({ key }) => key === 'z-tile') as UpscaleWidgetValues['tileControlnetModel'],
    },
  },
  { label: 'z-image-diffusers', main: model('z-diffusers', 'main', 'z-image', { format: 'diffusers' }) },
  { label: 'anima', main: model('anima', 'main', 'anima', { variant: 'anima_qwen3' }), patch: { cfgScale: 4.5 } },
  {
    label: 'anima-3.8b',
    main: model('anima-38', 'main', 'anima', { variant: 'anima_qwen35' }),
    patch: { cfgScale: 6 },
  },
  { label: 'krea-2-turbo', main: model('krea2-turbo', 'main', 'krea-2', { variant: 'krea2_turbo' }) },
  {
    label: 'krea-2-raw',
    main: model('krea2-raw', 'main', 'krea-2', { format: 'gguf_quantized', variant: 'krea2_base' }),
    patch: { cfgScale: 5.5 },
  },
  {
    label: 'krea-2-diffusers',
    main: model('krea2-diffusers', 'main', 'krea-2', { format: 'diffusers', variant: 'krea2_turbo' }),
  },
  { label: 'flux2-klein-4b', main: model('klein-4b', 'main', 'flux2', { variant: 'klein_4b' }) },
  {
    label: 'flux2-klein-9b',
    main: model('klein-9b', 'main', 'flux2', { format: 'gguf_quantized', variant: 'klein_9b' }),
  },
  { label: 'flux2-klein-base', main: model('klein-base', 'main', 'flux2', { variant: 'klein_4b_base' }) },
  {
    label: 'flux2-dev',
    main: model('flux2-dev', 'main', 'flux2', { format: 'gguf_quantized', variant: 'dev' }),
    patch: { cfgScale: 3.5 },
  },
  {
    label: 'flux2-diffusers',
    main: model('klein-diffusers', 'main', 'flux2', { format: 'diffusers', variant: 'klein_9b' }),
  },
  {
    label: 'qwen-image',
    main: model('qwen', 'main', 'qwen-image', { format: 'gguf_quantized', variant: 'generate' }),
    patch: { cfgScale: 4 },
  },
  {
    label: 'qwen-image-diffusers',
    main: model('qwen-diffusers', 'main', 'qwen-image', { format: 'diffusers', variant: 'generate' }),
    patch: { cfgScale: 1 },
  },
];

/** The widget values a case compiles, picked from the installed models plus its main model and a LoRA for it. */
export const valuesForCase = ({ main, patch }: (typeof CASES)[number]) => {
  const installed = [main, ...INSTALLED, model(`${main.key}-lora`, 'lora', main.base)];
  const defaults = createDefaultUpscaleWidgetValues(installed);
  const values: UpscaleWidgetValues = {
    ...defaults,
    inputImage: { height: 101, image_name: 'input.png', width: 203 },
    // Every builder's concept loader, so its node and edges reach the contract too.
    loras: [{ isEnabled: true, model: installed.at(-1) as UpscaleWidgetValues['loras'][number]['model'], weight: 1 }],
    ...patch,
  };

  return { installed, values };
};
