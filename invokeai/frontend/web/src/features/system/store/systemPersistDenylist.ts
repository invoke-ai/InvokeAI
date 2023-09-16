import { SystemState } from './systemSlice';

/**
 * System slice persist denylist
 */
export const systemPersistDenylist: (keyof SystemState)[] = [
  'currentIteration',
  'currentStep',
  'isCancelable',
  'isConnected',
  'isESRGANAvailable',
  'isGFPGANAvailable',
  'isProcessing',
  'totalIterations',
  'totalSteps',
  'isCancelScheduled',
  'progressImage',
  'wereModelsReceived',
  'isPersisted',
  'isUploading',
];
