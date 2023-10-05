import { ControlAdaptersState } from './types';

/**
 * ControlNet slice persist denylist
 */
export const controlAdaptersPersistDenylist: (keyof ControlAdaptersState)[] = [
  'pendingControlImages',
];
