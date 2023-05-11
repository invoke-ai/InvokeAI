export type AddNewModelType = 'ckpt' | 'diffusers' | null;

export type Coordinates = {
  x: number;
  y: number;
};

export type Dimensions = {
  width: number | string;
  height: number | string;
};

export type Rect = Coordinates & Dimensions;

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
  textTabAccordionState: number[];
  imageTabAccordionState: number[];
  canvasTabAccordionState: number[];
  floatingProgressImageRect: Rect;
  shouldShowProgressImages: boolean;
  shouldShowProgressInViewer: boolean;
  shouldShowImageParameters: boolean;
}
