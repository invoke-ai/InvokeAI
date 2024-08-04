import type { LogLevel } from 'app/logging/logger';
import type { InvocationProgressEvent } from 'services/events/types';
import { z } from 'zod';

type SystemStatus = 'CONNECTED' | 'DISCONNECTED' | 'PROCESSING' | 'ERROR' | 'LOADING_MODEL';

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
  denoiseProgress: Pick<InvocationProgressEvent, 'session_id' | 'batch_id' | 'image' | 'percentage' | 'message'> | null;
  consoleLogLevel: LogLevel;
  shouldLogToConsole: boolean;
  shouldAntialiasProgressImage: boolean;
  language: Language;
  shouldUseNSFWChecker: boolean;
  shouldUseWatermarker: boolean;
  status: SystemStatus;
  shouldEnableInformationalPopovers: boolean;
  cancellations: string[];
}
