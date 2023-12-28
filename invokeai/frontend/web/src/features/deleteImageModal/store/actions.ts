import { createAction } from '@reduxjs/toolkit';
import type { ImageDTO } from 'services/api/types';

import type { ImageUsage } from './types';

export const imageDeletionConfirmed = createAction<{
  imageDTOs: ImageDTO[];
  imagesUsage: ImageUsage[];
}>('deleteImageModal/imageDeletionConfirmed');
