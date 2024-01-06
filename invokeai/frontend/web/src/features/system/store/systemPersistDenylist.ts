import type { SystemState } from './types';

export const systemPersistDenylist: (keyof SystemState)[] = [
  'isInitialized',
  'isConnected',
  'denoiseProgress',
  'status',
];
