export type AddNewModelType = 'ckpt' | 'diffusers' | null;

export interface UIState {
  activeTab: number;
  currentTheme: string;
  parametersPanelScrollPosition: number;
  shouldHoldParametersPanelOpen: boolean;
  shouldPinParametersPanel: boolean;
  shouldShowParametersPanel: boolean;
  shouldShowDualDisplay: boolean;
  shouldShowImageDetails: boolean;
  shouldUseCanvasBetaLayout: boolean;
  shouldShowExistingModelsInSearch: boolean;
  shouldUseSliders: boolean;
  addNewModelUIOption: AddNewModelType;
}
