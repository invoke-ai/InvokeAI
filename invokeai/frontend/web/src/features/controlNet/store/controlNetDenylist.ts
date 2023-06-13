import { ControlNetState } from './controlNetSlice';

/**
 * ControlNet slice persist denylist
 */
export const controlNetDenylist: (keyof ControlNetState)[] = [
  'pendingControlImages',
];
