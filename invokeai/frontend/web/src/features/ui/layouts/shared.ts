import type { GridviewApi } from 'dockview';

export const LEFT_PANEL_ID = 'left';
export const MAIN_PANEL_ID = 'main';
export const RIGHT_PANEL_ID = 'right';

export const LAUNCHPAD_PANEL_ID = 'launchpad';
export const WORKSPACE_PANEL_ID = 'workspace';
export const VIEWER_PANEL_ID = 'viewer';

export const BOARDS_PANEL_ID = 'boards';
export const GALLERY_PANEL_ID = 'gallery';
export const LAYERS_PANEL_ID = 'layers';

export const SETTINGS_PANEL_ID = 'settings';

export const MODELS_PANEL_ID = 'models';
export const CUSTOM_NODES_PANEL_ID = 'customNodes';
export const QUEUE_PANEL_ID = 'queue';

export const DOCKVIEW_TAB_ID = 'tab-default';
export const DOCKVIEW_TAB_PROGRESS_ID = 'tab-progress';
export const DOCKVIEW_TAB_LAUNCHPAD_ID = 'tab-launchpad';
export const DOCKVIEW_TAB_CANVAS_VIEWER_ID = 'tab-canvas-viewer';
export const DOCKVIEW_TAB_CANVAS_WORKSPACE_ID = 'tab-canvas-workspace';

export const LEFT_PANEL_MIN_SIZE_PX = 420;
export const RIGHT_PANEL_MIN_SIZE_PX = 420;
// Keeps the main panel wide enough to fit the floating left/right toggle button
// groups on small screens, so the user can always grab them to expand the side panels.
export const MAIN_PANEL_MIN_SIZE_PX = 128;

export const BOARD_PANEL_MIN_HEIGHT_PX = 36;
export const BOARD_PANEL_MIN_EXPANDED_HEIGHT_PX = 128;
export const BOARD_PANEL_DEFAULT_HEIGHT_PX = 232;

export const GALLERY_PANEL_MIN_HEIGHT_PX = 36;
export const GALLERY_PANEL_MIN_EXPANDED_HEIGHT_PX = 128;
export const GALLERY_PANEL_DEFAULT_HEIGHT_PX = 232;

export const LAYERS_PANEL_MIN_HEIGHT_PX = 36;

export const CANVAS_BOARD_PANEL_DEFAULT_HEIGHT_PX = 36; // Collapsed by default on Canvas

export const SWITCH_TABS_FAKE_DELAY_MS = 300;

/**
 * Enforce the main panel's minimum width on the root gridview after the
 * container has been (re)constructed. The panel's `minimumWidth` set at
 * `addPanel` time only applies on a fresh layout — when `registerContainer`
 * restores from persisted JSON, the constraints come from that JSON, which
 * may pre-date `MAIN_PANEL_MIN_SIZE_PX`. Re-apply it here, and grow the panel
 * if its restored size violates the new minimum.
 */
export const enforceMainPanelMinWidth = (api: GridviewApi): void => {
  const main = api.getPanel(MAIN_PANEL_ID);
  if (!main) {
    return;
  }
  main.api.setConstraints({ maximumWidth: Number.MAX_SAFE_INTEGER, minimumWidth: MAIN_PANEL_MIN_SIZE_PX });
  if (main.api.width < MAIN_PANEL_MIN_SIZE_PX) {
    main.api.setSize({ width: MAIN_PANEL_MIN_SIZE_PX });
  }
};
