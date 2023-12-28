// When the aspect ratio is between these two values, we show the icon (experimentally determined)
export const ICON_LOW_CUTOFF = 0.23;
export const ICON_HIGH_CUTOFF = 1 / ICON_LOW_CUTOFF;
export const ICON_SIZE_PX = 48;
export const ICON_PADDING_PX = 16;
export const BOX_SIZE_CSS_CALC = `min(${ICON_SIZE_PX}px, calc(100% - ${ICON_PADDING_PX}px))`;
export const MOTION_ICON_INITIAL = {
  opacity: 0,
};
export const MOTION_ICON_ANIMATE = {
  opacity: 1,
  transition: { duration: 0.1 },
};
export const MOTION_ICON_EXIT = {
  opacity: 0,
  transition: { duration: 0.1 },
};
export const ICON_CONTAINER_STYLES = {
  width: '100%',
  height: '100%',
  alignItems: 'center',
  justifyContent: 'center',
};
