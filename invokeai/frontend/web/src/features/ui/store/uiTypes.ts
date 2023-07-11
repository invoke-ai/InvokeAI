import { SchedulerParam } from 'features/parameters/store/parameterZodSchemas';

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
  shouldShowProgressInViewer: boolean;
  shouldShowEmbeddingPicker: boolean;
  shouldShowAdvancedOptions: boolean;
  aspectRatio: number | null;
  favoriteSchedulers: SchedulerParam[];
}
