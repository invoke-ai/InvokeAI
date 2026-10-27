// The model families on the home page, summarised from each family's guide and the hardware table in
// `start-here/system-requirements.mdx`, with each family's tagline in `src/content/i18n/en.json`
// (`modelFamilies.<slug>.tagline`). Keep them in step when a guide changes.
//
// A card shows a cover image above its text when `src/assets/model-families/<slug>.webp` exists, and is text only
// otherwise. Covers are cropped to a 16:9 frame, so keep the subject near the centre.

type LicenseKind = 'open' | 'community' | 'noncommercial' | 'research' | 'restricted' | 'mixed';

/** A name, such as a license or format name, or the UI string key of a generic label. */
type Label = string | { labelKey: string };

type ModelFamily = {
  slug: string;
  name: string;
  /** The family's guide, without the deployment base path. */
  link: string;
  kind: 'image' | 'video';
  /** The headline model's license, or `mixed` when the family's headline models carry different licenses. */
  license: { kind: LicenseKind; label: Label };
  /** Minimum VRAM in GB for the smallest build in the hardware table, when the table lists the family. */
  vram?: number;
  /** Quantized and single-file builds the guide documents, at most four. */
  formats: Label[];
};

const variesByModel = { labelKey: 'modelFamilies.license.variesByModel' };
const nonCommercial = { labelKey: 'modelFamilies.license.nonCommercial' };
const singleFile = { labelKey: 'modelFamilies.format.singleFile' };

const localModels = '/users-guide/models/local-models';
const video = '/users-guide/video-generation';

const modelFamilies: ModelFamily[] = [
  {
    slug: 'flux-2',
    name: 'FLUX.2',
    link: `${localModels}/flux-2/`,
    kind: 'image',
    license: { kind: 'mixed', label: variesByModel },
    vram: 8,
    formats: ['GGUF', 'FP8', 'int8', 'SDNQ'],
  },
  {
    slug: 'flux-1',
    name: 'FLUX.1',
    link: `${localModels}/flux-1/`,
    kind: 'image',
    license: { kind: 'mixed', label: variesByModel },
    vram: 10,
    formats: ['NF4', 'GGUF', 'int8', 'SDNQ'],
  },
  {
    slug: 'z-image',
    name: 'Z-Image',
    link: `${localModels}/z-image/`,
    kind: 'image',
    license: { kind: 'open', label: 'Apache 2.0' },
    vram: 8,
    formats: ['GGUF', 'NVFP4', 'SDNQ'],
  },
  {
    slug: 'qwen-image-2-1',
    name: 'Qwen-Image-2.1',
    link: `${localModels}/qwen-image-2-1/`,
    kind: 'image',
    license: { kind: 'research', label: 'Qwen Research' },
    vram: 16,
    formats: ['GGUF', 'FP8', 'int8'],
  },
  {
    slug: 'qwen-image',
    name: 'Qwen Image',
    link: `${localModels}/qwen-image/`,
    kind: 'image',
    license: { kind: 'open', label: 'Apache 2.0' },
    formats: ['GGUF', 'FP8', 'NVFP4'],
  },
  {
    slug: 'krea-2',
    name: 'Krea-2',
    link: `${localModels}/krea-2/`,
    kind: 'image',
    license: { kind: 'community', label: 'Krea 2 Community' },
    vram: 16,
    formats: ['NVFP4', 'GGUF', 'int8', 'FP8'],
  },
  {
    slug: 'ideogram4',
    name: 'Ideogram 4',
    link: `${localModels}/ideogram4/`,
    kind: 'image',
    license: { kind: 'noncommercial', label: nonCommercial },
    vram: 24,
    formats: ['NF4', 'FP8', 'int8', 'GGUF'],
  },
  {
    slug: 'ernie-image',
    name: 'ERNIE-Image',
    link: `${localModels}/ernie-image/`,
    kind: 'image',
    license: { kind: 'open', label: 'Apache 2.0' },
    vram: 24,
    formats: ['GGUF', singleFile],
  },
  {
    slug: 'cogview4',
    name: 'CogView4',
    link: `${localModels}/cogview4/`,
    kind: 'image',
    license: { kind: 'open', label: 'Apache 2.0' },
    formats: [],
  },
  {
    slug: 'anima',
    name: 'Anima',
    link: `${localModels}/anima/`,
    kind: 'image',
    license: { kind: 'noncommercial', label: nonCommercial },
    formats: [singleFile, 'int8'],
  },
  {
    slug: 'sd-3',
    name: 'SD 3.5',
    link: `${localModels}/sd-3/`,
    kind: 'image',
    license: { kind: 'community', label: 'Stability Community' },
    formats: [],
  },
  {
    slug: 'sdxl',
    name: 'SDXL',
    link: `${localModels}/sdxl/`,
    kind: 'image',
    license: { kind: 'open', label: 'OpenRAIL++-M' },
    vram: 8,
    formats: [singleFile],
  },
  {
    slug: 'stable-diffusion-1-2',
    name: 'SD 1.5 / 2.x',
    link: `${localModels}/stable-diffusion-1-2/`,
    kind: 'image',
    license: { kind: 'open', label: 'OpenRAIL-M' },
    vram: 4,
    formats: [singleFile],
  },
  {
    slug: 'wan-22',
    name: 'Wan 2.2',
    link: `${video}/wan-22/`,
    kind: 'video',
    license: { kind: 'open', label: 'Apache 2.0' },
    vram: 6,
    formats: ['GGUF'],
  },
  {
    slug: 'ltx-2',
    name: 'LTX-2',
    link: `${video}/ltx-2/`,
    kind: 'video',
    license: { kind: 'community', label: 'LTX-2 Community' },
    formats: ['int8', 'NVFP4'],
  },
  {
    slug: 'minimax-h3',
    name: 'MiniMax H3',
    link: `${video}/minimax-h3/`,
    kind: 'video',
    license: { kind: 'restricted', label: 'MiniMax Community' },
    formats: ['int8'],
  },
];

const externalProviders = [
  { name: 'OpenAI', link: '/users-guide/models/external-models/openai/' },
  { name: 'Google Gemini', link: '/users-guide/models/external-models/gemini/' },
  { name: 'BytePlus Seedream', link: '/users-guide/models/external-models/seedream/' },
  { name: 'Alibaba Cloud', link: '/users-guide/models/external-models/alibabacloud/' },
];

export { externalProviders, modelFamilies };
export type { Label, LicenseKind, ModelFamily };
