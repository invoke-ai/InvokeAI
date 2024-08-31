import { atom } from 'nanostores';
import type { PresetType } from 'services/api/endpoints/stylePresets';

const initialState: StylePresetModalState = {
  isModalOpen: false,
  updatingStylePresetId: null,
  prefilledFormData: null,
};

/**
 * Tracks the state for the style preset modal.
 */
export const $stylePresetModalState = atom<StylePresetModalState>(initialState);

type StylePresetModalState = {
  isModalOpen: boolean;
  updatingStylePresetId: string | null;
  prefilledFormData: PrefilledFormData | null;
};

export type PrefilledFormData = {
  name: string;
  positivePrompt: string;
  negativePrompt: string;
  imageUrl: string | null;
  type: PresetType;
};
