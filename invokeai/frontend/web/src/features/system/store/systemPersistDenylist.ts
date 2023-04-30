import { SystemState } from './systemSlice';

/**
 * System slice persist denylist
 */
const itemsToDenylist: (keyof SystemState)[] = [
  'currentIteration',
  'currentStatus',
  'currentStep',
  'isCancelable',
  'isConnected',
  'isESRGANAvailable',
  'isGFPGANAvailable',
  'isProcessing',
  'socketId',
  'totalIterations',
  'totalSteps',
  'openModel',
  'isCancelScheduled',
  'progressImage',
  'wereModelsReceived',
  'wasSchemaParsed',
];

export const systemDenylist = itemsToDenylist.map(
  (denylistItem) => `system.${denylistItem}`
);
