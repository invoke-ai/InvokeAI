/** Supported application languages and their document direction. */
export const WORKBENCH_LANGUAGE_OPTIONS = [
  { value: 'ar', label: 'العربية', direction: 'rtl' },
  { value: 'az', label: 'Azərbaycan dili', direction: 'ltr' },
  { value: 'bg', label: 'Български', direction: 'ltr' },
  { value: 'de', label: 'Deutsch', direction: 'ltr' },
  { value: 'en', label: 'English', direction: 'ltr' },
  { value: 'en-GB', label: 'English (UK)', direction: 'ltr' },
  { value: 'es', label: 'Español', direction: 'ltr' },
  { value: 'fi', label: 'Suomi', direction: 'ltr' },
  { value: 'fr', label: 'Français', direction: 'ltr' },
  { value: 'he', label: 'עִבְֿרִית', direction: 'rtl' },
  { value: 'hu', label: 'Magyar Nyelv', direction: 'ltr' },
  { value: 'it', label: 'Italiano', direction: 'ltr' },
  { value: 'ja', label: '日本語', direction: 'ltr' },
  { value: 'ko', label: '한국어', direction: 'ltr' },
  { value: 'mn', label: 'Монгол', direction: 'ltr' },
  { value: 'nl', label: 'Nederlands', direction: 'ltr' },
  { value: 'pl', label: 'Polski', direction: 'ltr' },
  { value: 'pt', label: 'Português', direction: 'ltr' },
  { value: 'pt-BR', label: 'Português do Brasil', direction: 'ltr' },
  { value: 'ro', label: 'Română', direction: 'ltr' },
  { value: 'ru', label: 'Русский', direction: 'ltr' },
  { value: 'sv', label: 'Svenska', direction: 'ltr' },
  { value: 'tr', label: 'Türkçe', direction: 'ltr' },
  { value: 'uk', label: 'Українська', direction: 'ltr' },
  { value: 'vi', label: 'Tiếng Việt', direction: 'ltr' },
  { value: 'zh-CN', label: '简体中文', direction: 'ltr' },
  { value: 'zh-Hant', label: '漢語', direction: 'ltr' },
] as const;

export type WorkbenchLanguage = (typeof WORKBENCH_LANGUAGE_OPTIONS)[number]['value'];
export type WorkbenchLanguageDirection = (typeof WORKBENCH_LANGUAGE_OPTIONS)[number]['direction'];

export const WORKBENCH_LANGUAGES: WorkbenchLanguage[] = WORKBENCH_LANGUAGE_OPTIONS.map((option) => option.value);

export const normalizeWorkbenchLanguage = (value: unknown): WorkbenchLanguage | null => {
  if (value === 'ua') {
    return 'uk';
  }

  return typeof value === 'string' && WORKBENCH_LANGUAGES.includes(value as WorkbenchLanguage)
    ? (value as WorkbenchLanguage)
    : null;
};

export const getWorkbenchLanguageDirection = (language: WorkbenchLanguage): WorkbenchLanguageDirection =>
  WORKBENCH_LANGUAGE_OPTIONS.find((option) => option.value === language)?.direction ?? 'ltr';
