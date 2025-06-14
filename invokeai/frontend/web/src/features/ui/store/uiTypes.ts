import type { Dimensions } from 'features/controlLayers/store/types';

export type TabName = 'generate' | 'canvas' | 'upscaling' | 'workflows' | 'models' | 'queue';
export type CanvasRightPanelTabName = 'layers' | 'gallery';

export interface UIState {
  /**
   * Slice schema version.
   */
  _version: 3;
  /**
   * The currently active tab.
   */
  activeTab: TabName;
  /**
   * The currently active right panel canvas tab
   */
  activeTabCanvasRightPanel: CanvasRightPanelTabName;
  /**
   * Whether or not to show image details, e.g. metadata, workflow, etc.
   */
  shouldShowImageDetails: boolean;
  /**
   * Whether or not to show progress in the viewer.
   */
  shouldShowProgressInViewer: boolean;
  /**
   * The state of accordions. The key is the id of the accordion, and the value is a boolean representing the open state.
   */
  accordions: Record<string, boolean>;
  /**
   * The state of expanders. The key is the id of the expander, and the value is a boolean representing the open state.
   */
  expanders: Record<string, boolean>;
  /**
   * The size of textareas. The key is the id of the text area, and the value is an object representing its width and/or height.
   */
  textAreaSizes: Record<string, Partial<Dimensions>>;
  /**
   * Whether or not to show the user the open notification. Bump version to reset users who may have closed previous version.
   */
  shouldShowNotificationV2: boolean;
}
