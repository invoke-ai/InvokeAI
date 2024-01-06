import type { InvokeTabName } from './tabMap';

export interface UIState {
  activeTab: InvokeTabName;
  shouldShowImageDetails: boolean;
  shouldShowExistingModelsInSearch: boolean;
  shouldHidePreview: boolean;
  shouldShowProgressInViewer: boolean;
  panels: Record<string, string>;
}
