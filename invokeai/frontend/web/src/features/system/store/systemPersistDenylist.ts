import { SystemState } from './systemSlice';

/**
 * System slice persist denylist
 */
export const systemPersistDenylist: (keyof SystemState)[] = [
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
  'isPersisted',
  'isUploading',
];
