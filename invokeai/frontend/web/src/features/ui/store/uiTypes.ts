import type { InvokeTabName } from './tabMap';

export interface UIState {
  /**
   * Slice schema version.
   */
  _version: 1;
  /**
   * The currently active tab.
   */
  activeTab: InvokeTabName;
  /**
   * Whether or not to show image details, e.g. metadata, workflow, etc.
   */
  shouldShowImageDetails: boolean;
  /**
   * Whether or not to show progress in the viewer.
   */
  shouldShowProgressInViewer: boolean;
  /**
   * The react-resizable-panels state. The shape is managed by react-resizable-panels.
   */
  panels: Record<string, string>;
  /**
   * The state of accordions. The key is the id of the accordion, and the value is a boolean representing the open state.
   */
  accordions: Record<string, boolean>;
  /**
   * The state of expanders. The key is the id of the expander, and the value is a boolean representing the open state.
   */
  expanders: Record<string, boolean>;
}
