import type { InvokeTabName } from './tabMap';

export interface UIState {
  _version: 1;
  activeTab: InvokeTabName;
  shouldShowImageDetails: boolean;
  shouldHidePreview: boolean;
  shouldShowProgressInViewer: boolean;
  panels: Record<string, string>;
}
