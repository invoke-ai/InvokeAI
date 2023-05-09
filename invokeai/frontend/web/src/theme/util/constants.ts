// Calc Values
export const APP_CUTOFF = '0px';
export const APP_CONTENT_HEIGHT_CUTOFF = 'calc(70px + 1rem)'; // default: 7rem
export const PROGRESS_BAR_THICKNESS = 1.5;
export const APP_WIDTH = `calc(100vw - ${APP_CUTOFF})`;
export const APP_HEIGHT = `calc(100vh - ${PROGRESS_BAR_THICKNESS * 4}px)`;
export const APP_CONTENT_HEIGHT = `calc(100vh - ${APP_CONTENT_HEIGHT_CUTOFF})`;
export const APP_GALLERY_HEIGHT_PINNED = `calc(100vh - (${APP_CONTENT_HEIGHT_CUTOFF} + 6rem))`;
export const APP_GALLERY_HEIGHT = 'calc(100vw - 0.3rem + 5rem)';
export const APP_GALLERY_POPOVER_HEIGHT = `calc(100vh - (${APP_CONTENT_HEIGHT_CUTOFF} + 6rem))`;
export const APP_METADATA_HEIGHT = `calc(100vh - (${APP_CONTENT_HEIGHT_CUTOFF} + 4.4rem))`;

// this is in pixels
// export const PARAMETERS_PANEL_WIDTH = 384;

// do not touch ffs
export const APP_TEXT_TO_IMAGE_HEIGHT =
  'calc(100vh - 9.4375rem - 1.925rem - 1.15rem)';

// option bar
export const OPTIONS_BAR_MAX_WIDTH = '22.5rem';

export const PARAMETERS_PANEL_WIDTH = '28rem';
