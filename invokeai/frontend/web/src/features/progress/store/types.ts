import type { GeneratorProgressEvent } from 'services/events/types';

export type SystemStatus = 'CONNECTED' | 'DISCONNECTED' | 'PROCESSING' | 'ERROR' | 'LOADING_MODEL';

export type DenoiseProgress = GeneratorProgressEvent & {
  percentage: number;
};
