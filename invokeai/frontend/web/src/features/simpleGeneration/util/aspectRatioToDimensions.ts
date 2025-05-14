import { roundToMultiple } from 'common/util/roundDownToMultiple';
import type { ChatGPT4oAspectRatioID, Dimensions } from 'features/controlLayers/store/types';
import type { AspectRatio } from 'features/simpleGeneration/store/types';

export const getDimensions = (ratio: number, area: number): Dimensions => {
  const exactWidth = Math.sqrt(area * ratio);
  const exactHeight = exactWidth / ratio;

  return {
    width: roundToMultiple(exactWidth, 64),
    height: roundToMultiple(exactHeight, 64),
  };
};

const FLUX_SDXL_AREA = 1024 * 1024;
export const FLUX_SDXL_ASPECT_RATIO_MAP: Record<AspectRatio, Dimensions> = {
  '16:9': getDimensions(16 / 9, FLUX_SDXL_AREA),
  '3:2': getDimensions(3 / 2, FLUX_SDXL_AREA),
  '4:3': getDimensions(4 / 3, FLUX_SDXL_AREA),
  '1:1': getDimensions(1, FLUX_SDXL_AREA),
  '3:4': getDimensions(3 / 4, FLUX_SDXL_AREA),
  '2:3': getDimensions(2 / 3, FLUX_SDXL_AREA),
  '9:16': getDimensions(9 / 16, FLUX_SDXL_AREA),
};

const SD_1_AREA = 768 * 768;
export const SD_1_ASPECT_RATIO_MAP: Record<AspectRatio, Dimensions> = {
  '16:9': getDimensions(16 / 9, SD_1_AREA),
  '3:2': getDimensions(3 / 2, SD_1_AREA),
  '4:3': getDimensions(4 / 3, SD_1_AREA),
  '1:1': getDimensions(1, SD_1_AREA),
  '3:4': getDimensions(3 / 4, SD_1_AREA),
  '2:3': getDimensions(2 / 3, SD_1_AREA),
  '9:16': getDimensions(9 / 16, SD_1_AREA),
};

export const CHATGPT_4O_ASPECT_RATIO_MAP: Record<ChatGPT4oAspectRatioID, Dimensions> = {
  '1:1': { width: 1024, height: 1024 },
  '2:3': { width: 1024, height: 1536 },
  '3:2': { width: 1536, height: 1024 },
};

export const ASPECT_RATIO_MAP = {
  flux: FLUX_SDXL_ASPECT_RATIO_MAP,
  sdxl: FLUX_SDXL_ASPECT_RATIO_MAP,
  'chatgpt-4o': CHATGPT_4O_ASPECT_RATIO_MAP,
  'sd-1': SD_1_ASPECT_RATIO_MAP,
} as const;
