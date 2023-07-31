import { components } from 'services/api/schema';

export const MODEL_TYPE_MAP = {
  'sd-1': 'Stable Diffusion 1.x',
  'sd-2': 'Stable Diffusion 2.x',
  sdxl: 'Stable Diffusion XL',
  'sdxl-refiner': 'Stable Diffusion XL Refiner',
};

export const MODEL_TYPE_SHORT_MAP = {
  'sd-1': 'SD1',
  'sd-2': 'SD2',
  sdxl: 'SDXL',
  'sdxl-refiner': 'SDXLR',
};

export const clipSkipMap = {
  'sd-1': {
    maxClip: 12,
    markers: [0, 1, 2, 3, 4, 8, 12],
  },
  'sd-2': {
    maxClip: 24,
    markers: [0, 1, 2, 3, 5, 10, 15, 20, 24],
  },
  sdxl: {
    maxClip: 24,
    markers: [0, 1, 2, 3, 5, 10, 15, 20, 24],
  },
  'sdxl-refiner': {
    maxClip: 24,
    markers: [0, 1, 2, 3, 5, 10, 15, 20, 24],
  },
};

type LoRAModelFormatMap = {
  [key in components['schemas']['LoRAModelFormat']]: string;
};

export const LORA_MODEL_FORMAT_MAP: LoRAModelFormatMap = {
  lycoris: 'LyCORIS',
  diffusers: 'Diffusers',
};
