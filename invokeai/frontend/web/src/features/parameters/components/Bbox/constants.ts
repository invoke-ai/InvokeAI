import type { AspectRatioID } from 'features/controlLayers/store/types';

export const ASPECT_RATIO_MAP: Record<Exclude<AspectRatioID, 'Free'>, { ratio: number; inverseID: AspectRatioID }> = {
  '21:9': { ratio: 21 / 9, inverseID: '9:21' },
  '16:9': { ratio: 16 / 9, inverseID: '9:16' },
  '3:2': { ratio: 3 / 2, inverseID: '2:3' },
  '4:3': { ratio: 4 / 3, inverseID: '4:3' },
  '1:1': { ratio: 1, inverseID: '1:1' },
  '3:4': { ratio: 3 / 4, inverseID: '4:3' },
  '2:3': { ratio: 2 / 3, inverseID: '3:2' },
  '9:16': { ratio: 9 / 16, inverseID: '16:9' },
  '9:21': { ratio: 9 / 21, inverseID: '21:9' },
};
