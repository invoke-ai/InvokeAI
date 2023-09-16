import { SchedulerParam } from 'features/parameters/types/parameterSchemas';

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
  shouldShowImageDetails: boolean;
  shouldUseCanvasBetaLayout: boolean;
  shouldShowExistingModelsInSearch: boolean;
  shouldUseSliders: boolean;
  shouldHidePreview: boolean;
  shouldShowProgressInViewer: boolean;
  shouldShowEmbeddingPicker: boolean;
  shouldAutoChangeDimensions: boolean;
  favoriteSchedulers: SchedulerParam[];
  globalContextMenuCloseTrigger: number;
  panels: Record<string, string>;
}
