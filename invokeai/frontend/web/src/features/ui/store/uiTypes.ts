export type TabName = 'canvas' | 'upscaling' | 'workflows' | 'models' | 'queue';
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
   * Whether or not to show the user the open notification.
   */
  shouldShowNotification: boolean;
}
