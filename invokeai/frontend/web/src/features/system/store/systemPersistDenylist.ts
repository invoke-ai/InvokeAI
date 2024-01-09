import type { SystemState } from './types';

export const systemPersistDenylist: (keyof SystemState)[] = [
  'isConnected',
  'denoiseProgress',
  'status',
];
