export type StylePresetState = {
  activeStylePresetId: string | null;
  searchTerm: string;
  viewMode: boolean;
  showPromptPreviews: boolean;
  collapsedSections: {
    myTemplates: boolean;
    sharedTemplates: boolean;
    defaultTemplates: boolean;
  };
};
