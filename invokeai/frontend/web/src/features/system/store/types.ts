import { UseToastOptions } from '@chakra-ui/react';
import { InvokeLogLevel } from 'app/logging/logger';
import { ProgressImage } from 'services/events/types';
import { LANGUAGES } from './constants';

export type SystemStatus =
  | 'CONNECTED'
  | 'DISCONNECTED'
  | 'PROCESSING'
  | 'ERROR'
  | 'LOADING_MODEL';

export type DenoiseProgress = {
  session_id: string;
  progress_image: ProgressImage | null | undefined;
  step: number;
  total_steps: number;
  order: number;
  percentage: number;
};

export interface SystemState {
  isConnected: boolean;
  shouldConfirmOnDelete: boolean;
  enableImageDebugging: boolean;
  toastQueue: UseToastOptions[];
  denoiseProgress: DenoiseProgress | null;
  consoleLogLevel: InvokeLogLevel;
  shouldLogToConsole: boolean;
  shouldAntialiasProgressImage: boolean;
  language: keyof typeof LANGUAGES;
  shouldUseNSFWChecker: boolean;
  shouldUseWatermarker: boolean;
  status: SystemStatus;
}
