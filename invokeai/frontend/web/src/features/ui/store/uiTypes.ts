import type { InvokeTabName } from './tabMap';

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
  activeTab: InvokeTabName;
  shouldShowImageDetails: boolean;
  shouldShowExistingModelsInSearch: boolean;
  shouldHidePreview: boolean;
  shouldShowProgressInViewer: boolean;
  shouldAutoChangeDimensions: boolean;
  globalMenuCloseTrigger: number;
  panels: Record<string, string>;
}
