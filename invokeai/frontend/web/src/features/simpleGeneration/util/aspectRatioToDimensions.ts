import type { AspectRatioID, ChatGPT4oAspectRatioID, Dimensions } from 'features/controlLayers/store/types';

const LOCAL_MODEL_ASPECT_RATIO_MAP: Record<Exclude<AspectRatioID, 'Free'>, Dimensions> = {
  '1:1': {
    width: 1024,
    height: 1024,
  },
  // TODO(psyche): fill in rest of aspect ratios
};

export const CHATGPT_4O_ASPECT_RATIO_MAP: Record<ChatGPT4oAspectRatioID, Dimensions> = {
  '1:1': {
    width: 1024,
    height: 1024,
  },
  '2:3': {
    height: 1024,
    width: 1536,
  },
  '3:2': {
    height: 1536,
    width: 1024,
  },
};
