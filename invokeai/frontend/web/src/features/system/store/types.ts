import type { UseToastOptions } from '@invoke-ai/ui-library';
import type { LogLevel } from 'app/logging/logger';
import type { ProgressImage } from 'services/events/types';
import { z } from 'zod';

type SystemStatus = 'CONNECTED' | 'DISCONNECTED' | 'PROCESSING' | 'ERROR' | 'LOADING_MODEL';

type DenoiseProgress = {
  session_id: string;
  batch_id: string;
  progress_image: ProgressImage | null | undefined;
  step: number;
  total_steps: number;
  order: number;
  percentage: number;
};

const zLanguage = z.enum([
  'ar',
  'az',
  'de',
  'en',
  'es',
  'fi',
  'fr',
  'he',
  'hu',
  'it',
  'ja',
  'ko',
  'nl',
  'pl',
  'pt',
  'pt_BR',
  'ru',
  'sv',
  'tr',
  'ua',
  'zh_CN',
  'zh_Hant',
]);
export type Language = z.infer<typeof zLanguage>;
export const isLanguage = (v: unknown): v is Language => zLanguage.safeParse(v).success;

export interface SystemState {
  _version: 1;
  isConnected: boolean;
  shouldConfirmOnDelete: boolean;
  enableImageDebugging: boolean;
  toastQueue: UseToastOptions[];
  denoiseProgress: DenoiseProgress | null;
  consoleLogLevel: LogLevel;
  shouldLogToConsole: boolean;
  shouldAntialiasProgressImage: boolean;
  language: Language;
  shouldUseNSFWChecker: boolean;
  shouldUseWatermarker: boolean;
  status: SystemStatus;
  shouldEnableInformationalPopovers: boolean;
}
