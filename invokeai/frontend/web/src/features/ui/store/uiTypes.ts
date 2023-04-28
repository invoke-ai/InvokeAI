import { InvokeTabName } from './tabMap';

export type AddNewModelType = 'ckpt' | 'diffusers' | null;

export interface UIState {
  activeTab: number;
  currentTheme: string;
  parametersPanelScrollPosition: number;
  shouldPinParametersPanel: boolean;
  shouldShowParametersPanel: boolean;
  shouldShowImageDetails: boolean;
  shouldUseCanvasBetaLayout: boolean;
  shouldShowExistingModelsInSearch: boolean;
  shouldUseSliders: boolean;
  addNewModelUIOption: AddNewModelType;
  shouldHidePreview: boolean;
  shouldPinGallery: boolean;
  shouldShowGallery: boolean;
  openLinearAccordionItems: number[];
  openGenerateAccordionItems: number[];
  openUnifiedCanvasAccordionItems: number[];
}
