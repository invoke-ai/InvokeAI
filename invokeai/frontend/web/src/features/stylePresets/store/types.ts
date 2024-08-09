export type StylePresetModalState = {
  isModalOpen: boolean;
  updatingStylePresetId: string | null;
  prefilledFormData: PrefilledFormData | null;
};

export type PrefilledFormData = {
  name: string;
  positivePrompt: string;
  negativePrompt: string;
  imageUrl: string | null;
};

export type StylePresetState = {
  isMenuOpen: boolean;
  activeStylePresetId: string | null;
  searchTerm: string;
  viewMode: boolean;
};
