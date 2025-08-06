import { zLogLevel, zLogNamespace } from 'app/logging/logger';
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

export const zSystemState = z.object({
  _version: z.literal(3),
  shouldConfirmOnDelete: z.boolean(),
  shouldAntialiasProgressImage: z.boolean(),
  shouldConfirmOnNewSession: z.boolean(),
  shouldProtectStarredImages: z.boolean(),
  language: zLanguage,
  shouldUseNSFWChecker: z.boolean(),
  shouldUseWatermarker: z.boolean(),
  shouldEnableInformationalPopovers: z.boolean(),
  shouldEnableModelDescriptions: z.boolean(),
  logIsEnabled: z.boolean(),
  logLevel: zLogLevel,
  logNamespaces: z.array(zLogNamespace),
  shouldShowInvocationProgressDetail: z.boolean(),
  shouldHighlightFocusedRegions: z.boolean(),
});
export type SystemState = z.infer<typeof zSystemState>;
