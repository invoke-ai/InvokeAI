// The model families on the home page, summarised from each family's guide and the hardware table in
// `start-here/system-requirements.mdx`. Keep them in step when a guide changes.
//
// A card shows a cover image when `src/assets/model-families/<slug>.webp` exists, and a generated backdrop in the
// family's `hue` otherwise. Covers are cropped to fill a wide card, so keep the subject near the centre.

type LicenseKind = 'open' | 'community' | 'noncommercial' | 'research' | 'restricted' | 'mixed';

type ModelFamily = {
  slug: string;
  name: string;
  /** The family's guide, without the deployment base path. */
  link: string;
  kind: 'image' | 'video';
  tagline: string;
  /** The headline model's license, or `mixed` when the family's headline models carry different licenses. */
  license: { kind: LicenseKind; label: string };
  /** Minimum VRAM in GB for the smallest build in the hardware table, when the table lists the family. */
  vram?: number;
  /** Quantized and single-file builds the guide documents, at most four. */
  formats: string[];
  /** Hue in degrees for the generated backdrop, spread so neighbouring cards differ. */
  hue: number;
};

const localModels = '/users-guide/models/local-models';
const video = '/users-guide/video-generation';

const modelFamilies: ModelFamily[] = [
  {
    slug: 'flux-2',
    name: 'FLUX.2',
    link: `${localModels}/flux-2/`,
    kind: 'image',
    tagline: 'Multi-reference image editing',
    license: { kind: 'mixed', label: 'Varies by model' },
    vram: 8,
    formats: ['GGUF', 'FP8', 'int8', 'SDNQ'],
    hue: 265,
  },
  {
    slug: 'flux-1',
    name: 'FLUX.1',
    link: `${localModels}/flux-1/`,
    kind: 'image',
    tagline: 'Generate and edit, with Fill and ControlNets',
    license: { kind: 'mixed', label: 'Varies by model' },
    vram: 10,
    formats: ['NF4', 'GGUF', 'int8', 'SDNQ'],
    hue: 42,
  },
  {
    slug: 'z-image',
    name: 'Z-Image',
    link: `${localModels}/z-image/`,
    kind: 'image',
    tagline: 'Fast text-to-image in English and Chinese',
    license: { kind: 'open', label: 'Apache 2.0' },
    vram: 8,
    formats: ['GGUF', 'NVFP4', 'SDNQ'],
    hue: 180,
  },
  {
    slug: 'qwen-image-2-1',
    name: 'Qwen-Image-2.1',
    link: `${localModels}/qwen-image-2-1/`,
    kind: 'image',
    tagline: 'Strong text rendering, transparent output',
    license: { kind: 'research', label: 'Qwen Research' },
    vram: 16,
    formats: ['GGUF', 'FP8', 'int8'],
    hue: 317,
  },
  {
    slug: 'qwen-image',
    name: 'Qwen Image',
    link: `${localModels}/qwen-image/`,
    kind: 'image',
    tagline: 'Edit with reference images',
    license: { kind: 'open', label: 'Apache 2.0' },
    formats: ['GGUF', 'FP8', 'NVFP4'],
    hue: 95,
  },
  {
    slug: 'krea-2',
    name: 'Krea-2',
    link: `${localModels}/krea-2/`,
    kind: 'image',
    tagline: 'Text-to-image with style reference',
    license: { kind: 'community', label: 'Krea 2 Community' },
    vram: 16,
    formats: ['NVFP4', 'GGUF', 'int8', 'FP8'],
    hue: 232,
  },
  {
    slug: 'ideogram4',
    name: 'Ideogram 4',
    link: `${localModels}/ideogram4/`,
    kind: 'image',
    tagline: 'Structured prompts with placed regions',
    license: { kind: 'noncommercial', label: 'Non-commercial' },
    vram: 24,
    formats: ['NF4', 'FP8', 'int8', 'GGUF'],
    hue: 10,
  },
  {
    slug: 'ernie-image',
    name: 'ERNIE-Image',
    link: `${localModels}/ernie-image/`,
    kind: 'image',
    tagline: 'Text-to-image with a prompt enhancer',
    license: { kind: 'open', label: 'Apache 2.0' },
    vram: 24,
    formats: ['GGUF', 'Single file'],
    hue: 147,
  },
  {
    slug: 'cogview4',
    name: 'CogView4',
    link: `${localModels}/cogview4/`,
    kind: 'image',
    tagline: 'Plain text-to-image from THUDM',
    license: { kind: 'open', label: 'Apache 2.0' },
    formats: [],
    hue: 285,
  },
  {
    slug: 'anima',
    name: 'Anima',
    link: `${localModels}/anima/`,
    kind: 'image',
    tagline: 'Anime, with regional prompting',
    license: { kind: 'noncommercial', label: 'Non-commercial' },
    formats: ['Single file', 'int8'],
    hue: 62,
  },
  {
    slug: 'sd-3',
    name: 'SD 3.5',
    link: `${localModels}/sd-3/`,
    kind: 'image',
    tagline: "Stability AI's diffusion transformer",
    license: { kind: 'community', label: 'Stability Community' },
    formats: [],
    hue: 200,
  },
  {
    slug: 'sdxl',
    name: 'SDXL',
    link: `${localModels}/sdxl/`,
    kind: 'image',
    tagline: '1024px generation with ControlNet and Refiner',
    license: { kind: 'open', label: 'OpenRAIL++-M' },
    vram: 8,
    formats: ['Single file'],
    hue: 337,
  },
  {
    slug: 'stable-diffusion-1-2',
    name: 'SD 1.5 / 2.x',
    link: `${localModels}/stable-diffusion-1-2/`,
    kind: 'image',
    tagline: 'The lightest local image model',
    license: { kind: 'open', label: 'OpenRAIL-M' },
    vram: 4,
    formats: ['Single file'],
    hue: 115,
  },
  {
    slug: 'wan-22',
    name: 'Wan 2.2',
    link: `${video}/wan-22/`,
    kind: 'video',
    tagline: 'Text- and image-to-video',
    license: { kind: 'open', label: 'Apache 2.0' },
    vram: 6,
    formats: ['GGUF'],
    hue: 252,
  },
  {
    slug: 'ltx-2',
    name: 'LTX-2',
    link: `${video}/ltx-2/`,
    kind: 'video',
    tagline: 'Text-to-video with sound',
    license: { kind: 'community', label: 'LTX-2 Community' },
    formats: ['int8', 'NVFP4'],
    hue: 30,
  },
  {
    slug: 'minimax-h3',
    name: 'MiniMax H3',
    link: `${video}/minimax-h3/`,
    kind: 'video',
    tagline: 'Video with sound from reference media',
    license: { kind: 'restricted', label: 'MiniMax Community' },
    formats: ['int8'],
    hue: 167,
  },
];

const externalProviders = [
  { name: 'OpenAI', link: '/users-guide/models/external-models/openai/' },
  { name: 'Google Gemini', link: '/users-guide/models/external-models/gemini/' },
  { name: 'BytePlus Seedream', link: '/users-guide/models/external-models/seedream/' },
  { name: 'Alibaba Cloud', link: '/users-guide/models/external-models/alibabacloud/' },
];

export { externalProviders, modelFamilies };
export type { LicenseKind, ModelFamily };
