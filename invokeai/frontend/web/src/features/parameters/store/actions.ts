import { createAction } from '@reduxjs/toolkit';
import type { ImageDTO, MainModelField } from 'services/api/types';

export const initialImageSelected = createAction<ImageDTO | undefined>(
  'generation/initialImageSelected'
);

export const modelSelected = createAction<MainModelField>(
  'generation/modelSelected'
);

export const imageAdvancedOptionsExpanded = createAction(
  'parameters/imageAdvancedOptionsExpanded'
);
export const generationAdvancedOptionsExpanded = createAction(
  'parameters/generationAdvancedOptionsExpanded'
);

export const advancedPanelExpanded = createAction(
  'parameters/advancedPanelExpanded'
);
