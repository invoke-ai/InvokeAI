import type { StylePresetFormData } from 'features/stylePresets/components/StylePresetForm';
import type { StylePresetRecordWithImage } from 'services/api/endpoints/stylePresets';

export type StylePresetModalState = {
  isModalOpen: boolean;
  updatingStylePresetId: string | null;
  prefilledFormData: StylePresetFormData | null;
};

export type StylePresetState = {
  isMenuOpen: boolean;
  activeStylePreset: StylePresetRecordWithImage | null;
  searchTerm: string;
  viewMode: boolean;
};
