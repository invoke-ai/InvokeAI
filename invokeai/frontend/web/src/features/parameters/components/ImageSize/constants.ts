import type { ComboboxOption } from '@invoke-ai/ui-library';

import type { AspectRatioID, AspectRatioState } from './types';

// When the aspect ratio is between these two values, we show the icon (experimentally determined)
export const ICON_LOW_CUTOFF = 0.23;
export const ICON_HIGH_CUTOFF = 1 / ICON_LOW_CUTOFF;
export const ICON_SIZE_PX = 64;
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

export const ASPECT_RATIO_OPTIONS: ComboboxOption[] = [
  { label: 'Free' as const, value: 'Free' },
  { label: '16:9' as const, value: '16:9' },
  { label: '3:2' as const, value: '3:2' },
  { label: '4:3' as const, value: '4:3' },
  { label: '1:1' as const, value: '1:1' },
  { label: '3:4' as const, value: '3:4' },
  { label: '2:3' as const, value: '2:3' },
  { label: '9:16' as const, value: '9:16' },
] as const;

export const ASPECT_RATIO_MAP: Record<Exclude<AspectRatioID, 'Free'>, { ratio: number; inverseID: AspectRatioID }> = {
  '16:9': { ratio: 16 / 9, inverseID: '9:16' },
  '3:2': { ratio: 3 / 2, inverseID: '2:3' },
  '4:3': { ratio: 4 / 3, inverseID: '4:3' },
  '1:1': { ratio: 1, inverseID: '1:1' },
  '3:4': { ratio: 3 / 4, inverseID: '4:3' },
  '2:3': { ratio: 2 / 3, inverseID: '3:2' },
  '9:16': { ratio: 9 / 16, inverseID: '16:9' },
};

export const initialAspectRatioState: AspectRatioState = {
  id: '1:1',
  value: 1,
  isLocked: false,
};
