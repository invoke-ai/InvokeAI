import { SystemState } from './systemSlice';

/**
 * System slice persist blacklist
 */
const itemsToBlacklist: (keyof SystemState)[] = [
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
  'sessionId',
  'progressImage',
];

export const systemBlacklist = itemsToBlacklist.map(
  (blacklistItem) => `system.${blacklistItem}`
);
