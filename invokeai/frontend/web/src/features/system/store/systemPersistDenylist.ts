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
  'openModel',
  'isCancelScheduled',
  'progressImage',
  'wereModelsReceived',
  'wasSchemaParsed',
  'isPersisted',
  'isUploading',
];
