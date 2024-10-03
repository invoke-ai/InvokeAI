import type { LogLevel, LogNamespace } from 'app/logging/logger';
import { z } from 'zod';

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
  shouldConfirmOnDelete: boolean;
  shouldAntialiasProgressImage: boolean;
  shouldConfirmOnNewSession: boolean;
  language: Language;
  shouldUseNSFWChecker: boolean;
  shouldUseWatermarker: boolean;
  shouldEnableInformationalPopovers: boolean;
  logIsEnabled: boolean;
  logLevel: LogLevel;
  logNamespaces: LogNamespace[];
}
