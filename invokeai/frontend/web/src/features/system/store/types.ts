import type { LogLevel, LogNamespace } from 'app/logging/logger';
import { z } from 'zod/v4';

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
  'pt-BR',
  'ru',
  'sv',
  'tr',
  'ua',
  'vi',
  'zh-CN',
  'zh-Hant',
]);
export type Language = z.infer<typeof zLanguage>;
export const isLanguage = (v: unknown): v is Language => zLanguage.safeParse(v).success;

export interface SystemState {
  _version: 2;
  shouldConfirmOnDelete: boolean;
  shouldAntialiasProgressImage: boolean;
  shouldConfirmOnNewSession: boolean;
  language: Language;
  shouldUseNSFWChecker: boolean;
  shouldUseWatermarker: boolean;
  shouldEnableInformationalPopovers: boolean;
  shouldEnableModelDescriptions: boolean;
  logIsEnabled: boolean;
  logLevel: LogLevel;
  logNamespaces: LogNamespace[];
  shouldShowInvocationProgressDetail: boolean;
  shouldHighlightFocusedRegions: boolean;
}
