import type { ComboboxOption } from '@invoke-ai/ui-library';

import type { AspectRatioID, AspectRatioState } from './types';

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
