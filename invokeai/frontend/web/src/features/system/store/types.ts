import { UseToastOptions } from '@chakra-ui/react';
import { InvokeLogLevel } from 'app/logging/logger';
import i18n from 'i18n';
import { ProgressImage } from 'services/events/types';

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
  shouldDisableInformationalPopovers: boolean;
}

export const LANGUAGES = {
  ar: i18n.t('common.langArabic', { lng: 'ar' }),
  nl: i18n.t('common.langDutch', { lng: 'nl' }),
  en: i18n.t('common.langEnglish', { lng: 'en' }),
  fr: i18n.t('common.langFrench', { lng: 'fr' }),
  de: i18n.t('common.langGerman', { lng: 'de' }),
  he: i18n.t('common.langHebrew', { lng: 'he' }),
  it: i18n.t('common.langItalian', { lng: 'it' }),
  ja: i18n.t('common.langJapanese', { lng: 'ja' }),
  ko: i18n.t('common.langKorean', { lng: 'ko' }),
  pl: i18n.t('common.langPolish', { lng: 'pl' }),
  pt_BR: i18n.t('common.langBrPortuguese', { lng: 'pt_BR' }),
  pt: i18n.t('common.langPortuguese', { lng: 'pt' }),
  ru: i18n.t('common.langRussian', { lng: 'ru' }),
  zh_CN: i18n.t('common.langSimplifiedChinese', { lng: 'zh_CN' }),
  es: i18n.t('common.langSpanish', { lng: 'es' }),
  uk: i18n.t('common.langUkranian', { lng: 'ua' }),
};

export const STATUS_TRANSLATION_KEYS: Record<SystemStatus, string> = {
  CONNECTED: 'common.statusConnected',
  DISCONNECTED: 'common.statusDisconnected',
  PROCESSING: 'common.statusProcessing',
  ERROR: 'common.statusError',
  LOADING_MODEL: 'common.statusLoadingModel',
};
